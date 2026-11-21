"""Маска на живом кадре — строго по геометрии съёмки, без подгонки.

Почему маска «съезжает» (физика):
  1. Люфт столика. Шаговый столик после смены направления теряет несколько шагов: при «змейке» соседние ряды
     подъезжают к кадрам с разных сторон, и настоящее положение отличается от заданного на ±люфт/2. Сшивка
     это видит (оценивает люфт), а привязка маски к столику раньше считала по ЗАДАННЫМ положениям — ошибка
     зависела от того, с какой стороны столик подъехал сейчас.
  2. Общая аффинная подгонка «мозаика -> столик» по всем кадрам сглаживает местные неровности винта и сшивки.
  3. Матрица камеры (пиксели на шаг) из калибровки и та, что даёт сшивка, немного различаются (поворот, масштаб).

Что делается здесь:
  - у каждого кадра съёмки известна сторона подхода (dir_x, dir_y — пишется при съёмке; у старых — по порядку
    кадров), положение в мозаике (сшивка) и заданное положение (шаги);
  - по ним одной системой: pos = Mo·p + Bl·s + c  ->  Mo (камера в пикселях на шаг, как её видит сшивка),
    B = Mo⁻¹·Bl (люфт в шагах на знак подхода);
  - настоящее положение = заданное + B·s; для живого кадра s — сторона последнего движения столика;
  - живой кадр ложится в мозаику рядом с ближайшими кадрами съёмки: L0 = pos_i + Mo·(P − T_i) (локально, с весами),
    а квадраты маски (пиксели мозаики) переносятся в пиксели камеры вычитанием L0 — та же камера, тот же
    масштаб и поворот, никакой подгонки.
"""
import os
import numpy as np
import cv2

_GEO, _SCENES = {}, {}
MIN_RESP = 0.12


def _sg_from_sequence(p):
    """Сторона подхода по порядку кадров (для съёмок без dir_x, dir_y)."""
    sg, last = np.zeros_like(p), np.zeros(2)
    for k in range(1, len(p)):
        d = np.sign(p[k] - p[k - 1])
        last = np.where(d != 0, d, last)
        sg[k] = last
    if len(p) > 1:
        sg[0] = sg[1]
    return sg


def geometry(folder):
    """Строгая геометрия съёмки (кэш, пока не изменилась сшивка). dict:
    pos — левый верхний угол кадров в мозаике (px, как у сетки); p — заданное положение (шаги); sg — сторона
    подхода; Mo — px мозаики на шаг; B — люфт (шаги на знак); T — настоящее положение (шаги); W, H — кадр (px
    хранения); ms — мкм на шаг; resid — невязка модели (px, медиана); backlash_um."""
    from .storage import ScanData
    from .stitch import load_positions
    jp = os.path.join(folder, 'join', 'positions_all.csv')
    key = os.path.getmtime(jp) if os.path.exists(jp) else 0
    if folder in _GEO and _GEO[folder][0] == key:
        return _GEO[folder][1]
    st = ScanData(folder)
    try:
        H, W = st.shape[:2]
        p = np.array([[r['x_p'], r['y_p']] for r in st.index], float)
        sg = np.array([[r.get('dir_x', 0) or 0, r.get('dir_y', 0) or 0] for r in st.index], float)
        if not np.abs(sg).any():
            sg = _sg_from_sequence(p)
        pos = load_positions(folder, st)
        ms = float(st.meta.get('motor_step_um') or 1.0)
    finally:
        st.close()
    pos = pos - pos.min(0)
    use_s = [k for k in range(2) if np.ptp(sg[:, k]) > 0]
    X = np.c_[p, sg[:, use_s], np.ones(len(p))]
    if len(p) >= 3 and np.linalg.matrix_rank(p - p.mean(0)) == 2:
        coef = np.linalg.lstsq(X, pos, rcond=None)[0]
        Mo = coef[:2].T
        Bl = np.zeros((2, 2))
        for j, k in enumerate(use_s):
            Bl[:, k] = coef[2 + j]
        c = coef[-1]
    else:                                                   # один ряд / кадр: камера из калибровки
        from .storage import meta_A
        st = ScanData(folder)
        try:
            Mo = -meta_A(st.meta, W, H)
        finally:
            st.close()
        Bl = np.zeros((2, 2))
        c = (pos - p @ Mo.T).mean(0)
    B = np.linalg.solve(Mo, Bl)
    T = p + sg @ B.T
    resid = pos - (T @ Mo.T + c)
    g = dict(pos=pos, p=p, sg=sg, Mo=Mo, B=B, T=T, c=c, W=W, H=H, ms=ms,
             resid=float(np.median(np.hypot(*resid.T))) if len(p) else 0.0,
             backlash_um=(2 * np.abs(np.diag(B)) * ms).round(2).tolist())
    _GEO[folder] = (key, g)
    return g


def live_origin(g, P_cmd, s_now):
    """Где в мозаике левый верхний угол живого кадра: столик задан в P_cmd (шаги), подъехал со стороны s_now."""
    P = np.asarray(P_cmd, float) + g['B'] @ np.asarray(s_now, float)
    d = np.hypot(*(g['T'] - P).T)
    k = np.argsort(d)[:4]                                   # ближайшие кадры съёмки, с весами
    w = np.exp(-(d[k] / max(1.0, 0.5 * np.hypot(*np.linalg.solve(g['Mo'], (g['W'], g['H']))))) ** 2) + 1e-9
    L = g['pos'][k] + (P - g['T'][k]) @ g['Mo'].T
    return (L * w[:, None]).sum(0) / w.sum(), float(d[k[0]])


def live_polys(folder, P_cmd, s_now, live_shape, extra_um=(0.0, 0.0)):
    """Квадраты маски съёмки в пикселях живого кадра: [(4x2, номер выделения)] или []."""
    from .selections import marked_squares
    mk = marked_squares(folder)
    if not mk or not mk['px']:
        return []
    g = geometry(folder)
    L0, _ = live_origin(g, P_cmd, s_now)
    k = live_shape[1] / g['W']                              # кадр хранится уменьшенным? -> пиксели камеры
    sh = (np.asarray(mk.get('shift_um', (0, 0)), float) + np.asarray(extra_um, float)) / g['ms']
    dq = g['Mo'] @ sh
    return [((q + dq - L0) * k, mk['num']) for q in mk['px']]


def _scene(folder):
    from .viewer import _Scene
    sc = _SCENES.get(folder)
    if sc is None:
        if len(_SCENES) > 4:
            _SCENES.pop(next(iter(_SCENES))).st.close()
        sc = _SCENES[folder] = _Scene(folder)
    return sc


def _hp(g):
    g = g.astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), max(2.0, min(g.shape) / 60))


def offset(folder, img, P_cmd, s_now):
    """Проверка по картинке: насколько живой кадр расходится с мозаикой там, куда его кладёт геометрия.
    -> (каким должен быть сдвиг маски (dx, dy) мкм, надёжность, расхождение px) или (None, причина, None).
    Геометрия верна — расхождение ~0 и сдвиг маски должен быть ~0."""
    g = geometry(folder)
    L0, dist = live_origin(g, P_cmd, s_now)
    if dist > 1.5 * max(np.abs(np.linalg.solve(g['Mo'], (g['W'], g['H'])))):
        return None, "outside the scan", None
    W, H = g['W'], g['H']
    sc = _scene(folder)
    ren = sc.render(L0[0] + W / 2, L0[1] + H / 2, 1.0, W, H)
    live = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if img.shape[1] != W else img
    a = _hp(cv2.cvtColor(live, cv2.COLOR_BGR2GRAY) if live.ndim == 3 else live)
    b = _hp(cv2.cvtColor(ren, cv2.COLOR_BGR2GRAY))
    val = (cv2.cvtColor(ren, cv2.COLOR_BGR2GRAY) > 20)
    if val.mean() < 0.5:
        return None, "the frame is mostly outside the scan", None
    a[~val] = 0
    b[~val] = 0
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(b, a, win)          # живой = мозаика, сдвинутая на (dx, dy)
    if resp < MIN_RESP:
        return None, f"weak match ({resp:.2f})", None
    d_px = np.array([dx, dy])
    return np.linalg.solve(g['Mo'], d_px) * g['ms'], float(resp), d_px
