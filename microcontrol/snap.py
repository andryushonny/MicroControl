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

_GEO = {}


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
    from .storage import meta_A
    st = ScanData(folder)
    try:
        Mo_cal = -meta_A(st.meta, W, H)                     # камера из калибровки (px на шаг)
    finally:
        st.close()
    Mo = None
    if len(p) >= X.shape[1] + 2 and np.linalg.matrix_rank(X) == X.shape[1] \
            and np.linalg.matrix_rank(p - p.mean(0)) == 2:
        coef = np.linalg.lstsq(X, pos, rcond=None)[0]
        Mo = coef[:2].T
        Bl = np.zeros((2, 2))
        for j, k in enumerate(use_s):
            Bl[:, k] = coef[2 + j]
        c = coef[-1]
        if np.linalg.norm(Mo - Mo_cal) > 0.05 * np.linalg.norm(Mo_cal):   # мало кадров: люфт и сдвиг ряда
            Mo = None                                       # неразличимы (2 ряда) — подгонке не верить
    if Mo is None:                                          # камера из калибровки; люфт и сдвиг — по остатку
        Mo = Mo_cal
        r_ = pos - p @ Mo.T
        Xs = np.c_[sg[:, use_s], np.ones(len(p))]
        Bl = np.zeros((2, 2))
        if use_s and len(p) >= Xs.shape[1] + 1 and np.linalg.matrix_rank(Xs) == Xs.shape[1]:
            cf = np.linalg.lstsq(Xs, r_, rcond=None)[0]
            for j, k in enumerate(use_s):
                Bl[:, k] = cf[j]
            c = cf[-1]
        else:
            c = r_.mean(0)
    B = np.linalg.solve(Mo, Bl)
    T = p + sg @ B.T
    resid = pos - (T @ Mo.T + c)
    g = dict(pos=pos, p=p, sg=sg, Mo=Mo, B=B, T=T, c=c, W=W, H=H, ms=ms,
             resid=float(np.median(np.hypot(*resid.T))) if len(p) else 0.0,
             backlash_um=(2 * np.abs(np.diag(B)) * ms).round(2).tolist())
    _GEO[folder] = (key, g)
    return g


def mosaic_to_stage(folder, q):
    """Пиксели мозаики (как у сетки и просмотра) -> настоящие мкм столика (строго по геометрии съёмки)."""
    g = geometry(folder)
    q = np.atleast_2d(np.asarray(q, float))
    return (q - g['c'] - (g['W'] / 2, g['H'] / 2)) @ np.linalg.inv(g['Mo']).T * g['ms']


def stage_to_mosaic(folder, S):
    """Настоящие мкм столика -> пиксели мозаики."""
    g = geometry(folder)
    S = np.atleast_2d(np.asarray(S, float))
    return (S / g['ms']) @ g['Mo'].T + g['c'] + (g['W'] / 2, g['H'] / 2)


def live_origin(g, P_cmd, s_now):
    """Где в мозаике левый верхний угол живого кадра: столик задан в P_cmd (шаги), подъехал со стороны s_now."""
    P = np.asarray(P_cmd, float) + g['B'] @ np.asarray(s_now, float)
    d = np.hypot(*(g['T'] - P).T)
    k = np.argsort(d)[:4]                                   # ближайшие кадры съёмки, с весами
    w = np.exp(-(d[k] / max(1.0, 0.5 * np.hypot(*np.linalg.solve(g['Mo'], (g['W'], g['H']))))) ** 2) + 1e-9
    L = g['pos'][k] + (P - g['T'][k]) @ g['Mo'].T
    return (L * w[:, None]).sum(0) / w.sum(), float(d[k[0]])


_SCENES = {}
MIN_RESP = 0.12


def _scene(folder):
    from .viewer import _Scene
    sc = _SCENES.get(folder)
    if sc is None:
        if len(_SCENES) > 4:
            _SCENES.pop(next(iter(_SCENES))).st.close()
        sc = _SCENES[folder] = _Scene(folder)
    return sc


def _hp(g):
    import cv2
    g = g.astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), max(2.0, min(g.shape) / 60))


def offset(folder, img, P_cmd, s_now):
    """Проверка геометрии по картинке: насколько живой кадр расходится с мозаикой там, куда его кладёт геометрия.
    -> (сдвиг в мкм, надёжность, расхождение px) или (None, причина, None). Геометрия верна — расхождение ~0."""
    import cv2
    g = geometry(folder)
    L0, dist = live_origin(g, P_cmd, s_now)
    if dist > 1.5 * max(np.abs(np.linalg.solve(g['Mo'], (g['W'], g['H'])))):
        return None, "outside the scan", None
    W, H = g['W'], g['H']
    ren = _scene(folder).render(L0[0] + W / 2, L0[1] + H / 2, 1.0, W, H)
    live = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if img.shape[1] != W else img
    a = _hp(cv2.cvtColor(live, cv2.COLOR_BGR2GRAY) if live.ndim == 3 else live)
    b = _hp(cv2.cvtColor(ren, cv2.COLOR_BGR2GRAY))
    val = cv2.cvtColor(ren, cv2.COLOR_BGR2GRAY) > 20
    if val.mean() < 0.5:
        return None, "the frame is mostly outside the scan", None
    a[~val] = 0
    b[~val] = 0
    (dx, dy), resp = cv2.phaseCorrelate(b, a, cv2.createHanningWindow((W, H), cv2.CV_32F))
    if resp < MIN_RESP:
        return None, f"weak match ({resp:.2f})", None
    d_px = np.array([dx, dy])
    return np.linalg.solve(g['Mo'], d_px) * g['ms'], float(resp), d_px
