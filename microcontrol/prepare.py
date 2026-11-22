"""Подготовка перед съёмкой (m.prepare()) и калибровка дисторсии объектива (m.calibrate_distortion()).

Подготовка — на месте, куда смотрит камера (лучше: крест или край чешуйки рядом с пустой подложкой):
  1. свет: лампа прогрелась? (яркость за 3 с не плывёт больше 1 %);
  2. фокус: автофокус здесь;
  3. экспозиция: самые яркие детали (алюминиевые кресты) — около 80 % шкалы, пересвета нет: так у краёв
     креста больше всего уровней яркости, центр находится точнее;
  4. баланс белого: если в кадре в основном пустая подложка (камера сама, однократно);
  5. успокоение столика: после остановки кадры снимаются подряд, пока картинка не перестанет дрожать
     (< 0.1 px) — это время и становится паузой перед снимком (move_pause).

Калибровка дисторсии — без крестов и без точности столика: 3×3 кадра с перекрытием 50 % вокруг места
с деталями (кресты, чешуйки, грязь). Одна и та же деталь видна в разных местах поля зрения; по сдвигам
мелких участков (до сотни на пару кадров) находятся положения кадров и дисторсия k1, k2, p1, p2 вместе
(как в anchor.py, только швы — плотные). Результат — для этого объектива: применяется ко всем сканам.
"""
import json
import os
import time
import numpy as np
import cv2
from . import log as L
from .util import Result, Progress


def _g(img):
    from .mask import meas_gray
    return meas_gray(img)


# ------------------------------------------------------------------ экспозиция
def auto_exposure(m, target=0.8, q=99.5, tries=7):
    """Экспозиция: q-й процентиль яркости (зелёный канал) — target шкалы, пересвет < 0.05 % пикселей.
    -> (экспозиция мс, процентиль 0..1, доля пересвеченных)."""
    m.cam.set(exposure_ms=float(m.exposure))
    e = float(m.exposure)
    p = sat = 0.0
    for _ in range(tries):
        g = _g(m.image(2))
        top = 255.0 if g.max() <= 255 else 65535.0
        p = float(np.percentile(g, q)) / top
        sat = float((g >= 0.995 * top).mean())
        if sat > 5e-4:                                     # пересвет: сразу вдвое меньше
            e_new = e * 0.5
        elif abs(p - target) < 0.03:
            break
        else:
            e_new = e * target / max(p, 0.02)
        e_new = float(np.clip(e_new, 0.05, 1000.0))
        if abs(e_new - e) < 1e-3:
            break
        e = e_new
        m.exposure = round(e, 3)
    return e, p, sat


# ------------------------------------------------------------------ свет
def light_drift(m, seconds=3.0):
    """Насколько плывёт яркость за seconds, % (лампа не прогрелась — плывёт)."""
    t0 = time.monotonic()
    vals = []
    while time.monotonic() - t0 < seconds:
        vals.append(float(_g(m.image(0)).mean()))
        time.sleep(0.15)
    if len(vals) < 3:
        return 0.0
    return float(100.0 * (max(vals) - min(vals)) / max(np.mean(vals), 1e-6))


# ------------------------------------------------------------------ успокоение
def settle_time(m, steps=20, limit_s=2.0, tol_px=0.1):
    """После переезда — сколько секунд картинка ещё дрожит (сдвиг кадра к последнему > tol_px).
    Столик едет туда и обратно на steps шагов по X; кадры снимаются подряд limit_s секунд."""
    p0 = m.position
    m.move(dx=steps * m.motor_step)
    m.goto(x=p0.x, y=p0.y)
    t0 = time.monotonic()
    fr, ts = [], []
    while time.monotonic() - t0 < limit_s:
        fr.append(_g(m.image(0)))
        ts.append(time.monotonic() - t0)
    if len(fr) < 3:
        return 0.0
    ref = fr[-1]
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    sh = [np.hypot(*cv2.phaseCorrelate(f * win, ref * win)[0]) for f in fr]
    moving = [t for t, s in zip(ts, sh) if s > tol_px]
    return float(moving[-1]) if moving else 0.0


# ------------------------------------------------------------------ подготовка
def prepare(m, light=True, focus=True, exposure=True, white=True, settle=True, verbose=True):
    """Подготовка перед съёмкой (см. описание модуля). Возвращает Result со всеми шагами."""
    m._need_cam()
    out = {}
    steps = [n for n, on in (('light', light), ('focus', focus), ('exposure', exposure), ('white', white),
                             ('settle', settle)) if on]
    pr = Progress(len(steps), "Prepare", unit='steps') if verbose else None

    def step(txt):
        if pr:
            pr.update(1, txt)
    if light:
        step("light: is the lamp steady")
        d = light_drift(m)
        out["Light"] = f"steady ({d:.1f} % in 3 s)" if d < 1.0 else f"still changing: {d:.1f} % in 3 s - wait a few minutes"
    if focus:
        step("focus here")
        try:
            m.focus()
            out["Focus"] = f"Z {m.position.z:.1f} um, sharpness {m.sharpness}"
        except Exception as e:
            out["Focus"] = f"not found: {e}"
    if exposure:
        step("exposure: brightest details at 80 %")
        e, p, sat = auto_exposure(m)
        out["Exposure"] = (f"{e:.2f} ms: brightest details at {100 * p:.0f} %, overexposed {100 * sat:.2f} %"
                           + ("" if p > 0.5 else " (too dark even so: more light, L)"))
    if white:
        step("white balance")
        g = _g(m.image(0))
        hp = cv2.GaussianBlur(g, (0, 0), 2) - cv2.GaussianBlur(g, (0, 0), 12)
        busy = float((np.abs(hp) > 6).mean())
        if busy < 0.08:
            m.white_balance()
            out["White balance"] = "done on the empty substrate"
        else:
            out["White balance"] = (f"skipped: {100 * busy:.0f} % of the picture has details - aim at an empty place "
                                    "and press W (crosses are measured in the green channel anyway)")
    if settle:
        step("stage settling after a stop")
        t = settle_time(m)
        mp = round(max(0.05, 1.3 * t), 2)
        old = m.move_pause
        m.move_pause = mp
        out["Stage settles"] = f"in {t:.2f} s: pause before a picture {old} -> {mp} s"
    if pr:
        pr.close()
    L.event('PRE-001', f"Prepared: {out}")
    return Result("Prepared", **out)


# ------------------------------------------------------------------ дисторсия
def _patch_shifts(a, b, d, W, H, patch, min_resp=0.25):
    """Сдвиги участков: кадр b сдвинут относительно a на d (px, точка a = точка b + d). Перекрытие режется
    на участки patch×patch; для каждого — точный сдвиг (фазовая корреляция). -> [(центр в a, сдвиг r_a − r_b)]."""
    x0, x1 = max(0, d[0]), min(W, W + d[0])
    y0, y1 = max(0, d[1]), min(H, H + d[1])
    out = []
    step = int(patch * 0.75)
    win = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
    for y in range(int(y0) + 4, int(y1) - patch - 4, step):
        for x in range(int(x0) + 4, int(x1) - patch - 4, step):
            pa = a[y:y + patch, x:x + patch]
            xb, yb = int(round(x - d[0])), int(round(y - d[1]))
            if xb < 0 or yb < 0 or xb + patch > W or yb + patch > H:
                continue
            pb = b[yb:yb + patch, xb:xb + patch]
            if pa.std() < 3 or pb.std() < 3:
                continue
            ha = pa - cv2.GaussianBlur(pa, (0, 0), patch / 8)
            hb = pb - cv2.GaussianBlur(pb, (0, 0), patch / 8)
            (sx, sy), resp = cv2.phaseCorrelate(hb * win, ha * win)   # pa = pb сдвинутый на (sx, sy)
            if resp < min_resp or abs(sx) > 6 or abs(sy) > 6:
                continue
            ca = np.array([x + patch / 2, y + patch / 2])
            out.append((ca, np.array([x - xb + sx, y - yb + sy])))
    return out


def _overlap_shift(a, b, d0, W, H):
    """Сдвиг пары по общей части (предсказание d0: точка a = точка b + d0): фазовая корреляция полос. -> d или None."""
    dx, dy = int(round(d0[0])), int(round(d0[1]))
    x0, x1, y0, y1 = max(0, dx), min(W, W + dx), max(0, dy), min(H, H + dy)
    if min(x1 - x0, y1 - y0) < 32:
        return None
    pa = a[y0:y1, x0:x1]
    pb = b[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    ha = pa - cv2.GaussianBlur(pa, (0, 0), 8)
    hb = pb - cv2.GaussianBlur(pb, (0, 0), 8)
    win = cv2.createHanningWindow(pa.shape[::-1], cv2.CV_32F)
    (sx, sy), resp = cv2.phaseCorrelate(hb * win, ha * win)
    if resp < 0.05 or np.hypot(sx, sy) > 0.3 * min(x1 - x0, y1 - y0):
        return None
    return np.array([dx + sx, dy + sy])


def calibrate_distortion(m, n=3, overlap=0.5, patch=None, save=True, verbose=True):
    """Дисторсия объектива по n×n кадрам с перекрытием overlap (см. описание модуля). Наведите на место
    с деталями. -> Result; r.theta — (k1, k2, p1, p2)."""
    from .optics import dist_basis, dist_corner, DIST_KEYS, _dist_path
    m._need_cam()
    p0 = m.position
    fw, fh = m.frame_size
    pr = Progress(n * n + 2, "Distortion", unit='frames') if verbose else None
    imgs, S = [], []
    old_os = m.one_side
    m.one_side = True                                    # люфт: все кадры — с одной стороны
    try:
        for i in range(n):
            for j in range(n):
                dx = (i - (n - 1) / 2) * (1 - overlap) * fw
                dy = (j - (n - 1) / 2) * (1 - overlap) * fh
                m.goto(x=p0.x + dx, y=p0.y + dy)
                time.sleep(float(m.move_pause))
                imgs.append(_g(m.image()))
                S.append((dx, dy))
                if pr:
                    pr.update(1, f"frame {len(imgs)} of {n * n}")
    finally:
        m.goto(x=p0.x, y=p0.y)
        m.one_side = old_os
    H, W = imgs[0].shape
    c = np.array([W / 2, H / 2])
    R = float(np.hypot(W / 2, H / 2))
    patch = patch or int(np.clip(min(W, H) / 10, 32, 128)) // 2 * 2
    K = m._pixels_per_step() / m.motor_step              # сдвиг картинки (px) на 1 мкм столика
    pred = -np.array(S) @ K.T                            # положение кадра в мозаике, px (по столику)
    nI = len(imgs)
    rows, rhs = [], []
    if pr:
        pr.update(1, "matching small pieces of the overlaps")
    pairs = 0
    for a in range(nI):
        for b in range(a + 1, nI):
            d0 = pred[b] - pred[a]                       # точка a = точка b + d0
            if abs(d0[0]) > 0.8 * W or abs(d0[1]) > 0.8 * H:
                continue
            d = _overlap_shift(imgs[a], imgs[b], d0, W, H)  # точно по общей полосе (столик лишь подсказка)
            if d is None:
                continue
            sh = _patch_shifts(imgs[a], imgs[b], d, W, H, patch)
            if len(sh) < 6:
                continue
            pairs += 1
            for ca, dr in sh:
                cb = ca - dr
                G = dist_basis(np.array([ca - c]), R)[0] - dist_basis(np.array([cb - c]), R)[0]   # (2, 4)
                for ax in (0, 1):
                    row = np.zeros(2 * nI + 4)
                    row[2 * b + ax], row[2 * a + ax] = 1.0, -1.0
                    row[2 * nI:] = G[ax]
                    rows.append(row)
                    rhs.append(dr[ax])
    if len(rows) < 40:
        raise ValueError(f"Too little detail to match ({len(rows) // 2} pieces): aim at crosses or a flake edge")
    A = np.array(rows)
    y = np.array(rhs)
    gauge = np.zeros((2, 2 * nI + 4))
    gauge[0, 0] = gauge[1, 1] = 1e3                      # первый кадр — в нуле
    A = np.vstack([A, gauge])
    y = np.r_[y, 0, 0]
    w = np.ones(len(y))
    for _ in range(5):                                   # выбросы (участок без деталей, грязь сдвинулась) — прочь
        x = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        res = A @ x - y
        sd = 1.48 * np.median(np.abs(res[:-2])) + 1e-3
        w = np.r_[1.0 / (1.0 + (res[:-2] / (2.5 * sd)) ** 2), 1.0, 1.0]
    th = x[2 * nI:]
    D = {k: float(v) for k, v in zip(DIST_KEYS, th)}
    corner = dist_corner(D, W, H)
    rms0 = float(np.sqrt(np.mean((np.linalg.lstsq(A[:, :2 * nI], y, rcond=None)[0] @ A[:, :2 * nI].T - y)[:-2] ** 2)))
    rms1 = float(np.sqrt(np.mean(res[:-2] ** 2)))
    dd = dict(D, corner_px=corner, apply=bool(corner >= 0.2), frame_px=[W, H], objective=m.objective,
              pieces=len(rows) // 2, pairs=pairs, rms_px=round(rms1, 3), rms_px_without=round(rms0, 3),
              calibrated=True, source="calibration: overlapping frames (prepare.calibrate_distortion)",
              time=time.strftime('%Y-%m-%d %H:%M'))
    if save:
        p = _dist_path({'device': dict(getattr(m, 'device', {}) or {}), 'objective': m.objective})
        m.__dict__.setdefault('_dist_cal', {})[int(m.objective)] = dd
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(dd, f, indent=1)
    if pr:
        pr.close()
    L.event('OPT-010', f"Distortion {m.objective}x: {dd}")
    r = Result(f"Distortion of {m.objective}x measured",
               **{"Pieces matched": f"{len(rows) // 2} in {pairs} overlaps",
                  "Edge of the frame": f"{corner:.2f} px off" + ("" if dd['apply'] else " (too small: not applied)"),
                  "Error of the pieces": f"{rms0:.3f} px -> {rms1:.3f} px with the correction",
                  "k1, k2, p1, p2": ", ".join(f"{v:+.5f}" for v in th)})
    r.theta, r.info = th, dd
    return r
