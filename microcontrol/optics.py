"""Оптические поправки кадров: неравномерность освещения (flat-field) и дисторсия объектива.

Освещение. Объектив и камера темнят края кадра (виньетирование), поэтому границы кадров видны в мозаике
по яркости. Поле освещения F(x, y) — медиана по многим кадрам (детали образца у разных кадров разные
и в медиане пропадают), сглаженная и нормированная на среднее; исправленный кадр I' = I / F.

Дисторсия (радиальная, один коэффициент). Точка кадра на расстоянии r от центра смещена:
    x_d = c + (x_u - c)·(1 + k1·ρ²),   ρ = r / R,  R — половина диагонали кадра.
Смещение в углу кадра = k1·R, px. k1 измеряется по крестам (решётка с точным шагом — линейка):
mc.distortion(folder) после разметки крестов в mc.view. Значение хранится для объектива и применяется
при сшивке и выводе.

Поправки подключаются к ScanData автоматически (attach), если в папке скана есть join/flatfield.npy
или для объектива измерена дисторсия.
"""
import json, os
import numpy as np
import cv2

HOME = os.path.join(os.path.expanduser('~'), '.microcontrol')


# ------------------------------------------------------------------ освещение
def flatfield(st, n=40, down=4):
    """Поле освещения скана: медиана до n кадров (по яркости), гауссово сглаживание, среднее = 1.
    Возвращает уменьшенное в down раз поле (float32) или None, если кадров мало."""
    N = len(st)
    if N < 12:
        return None
    ids = np.linspace(0, N - 1, min(n, N)).astype(int)
    fix, st.fix = getattr(st, 'fix', None), None              # считать по сырым кадрам
    try:
        stack = []
        for i in ids:
            g = st.frame(int(i))
            g = g if g.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(g), cv2.COLOR_BGR2GRAY)
            stack.append(cv2.resize(g, None, fx=1 / down, fy=1 / down, interpolation=cv2.INTER_AREA).astype(np.float32))
    finally:
        st.fix = fix
    F = np.median(np.array(stack), 0)
    F = cv2.GaussianBlur(F, (0, 0), min(F.shape) / 12)
    F /= F.mean()
    return np.clip(F, 0.2, 5).astype(np.float32)


# ------------------------------------------------------------------ дисторсия
def _dist_path(meta):
    """Файл дисторсии для микроскопа (по серийному номеру) и объектива; у симулятора — нет."""
    serial = str((meta.get('device') or {}).get('Serial', 'unknown'))
    if serial.upper().startswith('SIM') or not meta.get('objective'):
        return None
    safe = ''.join(ch if ch.isalnum() else '_' for ch in serial)
    return os.path.join(HOME, 'devices', f"distortion_{safe}_{meta['objective']}x.json")


def load_k1(folder, meta):
    """k1 для скана: join/distortion.json в папке скана, иначе измеренный для этого микроскопа и объектива."""
    for p in (os.path.join(folder, 'join', 'distortion.json'), _dist_path(meta)):
        if p and os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                d = json.load(f)
            if d.get('apply', True):
                return float(d['k1'])
    return 0.0


def undistort_maps(W, H, k1):
    """Карты cv2.remap: для пикселя выпрямленного кадра — где он в снятом кадре."""
    c = np.array([W / 2, H / 2], np.float32)
    R = float(np.hypot(W / 2, H / 2))
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = x - c[0], y - c[1]
    s = 1 + k1 * (dx * dx + dy * dy) / (R * R)
    return (c[0] + dx * s).astype(np.float32), (c[1] + dy * s).astype(np.float32)


# ------------------------------------------------------------------ подключение к ScanData
def attach(st, folder):
    """Поправить чтение кадров скана: I / F, затем выпрямление дисторсии (если они есть)."""
    W, H = st.shape[1], st.shape[0]
    fp = os.path.join(folder, 'join', 'flatfield.npy')
    F = None
    if os.path.exists(fp):
        F = cv2.resize(np.load(fp), (W, H), interpolation=cv2.INTER_LINEAR)
    k1 = load_k1(folder, st.meta)
    maps = undistort_maps(W, H, k1) if abs(k1) * np.hypot(W / 2, H / 2) >= 0.3 else None
    if F is None and maps is None:
        st.fix = None
        return st

    def fix(img):
        out = img
        if F is not None:
            f = F if img.ndim == 2 else F[..., None]
            out = np.clip(img.astype(np.float32) / f, 0, 255).astype(np.uint8)
        if maps is not None:
            out = cv2.remap(out, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out

    st.fix = fix
    st.fix_info = dict(flatfield=F is not None, k1=k1 if maps is not None else 0.0)
    return st


def measure_distortion(folder, save=True):
    """Дисторсия по крестам. Нужны: mc.stitch(folder) и решётка крестов из mc.view (G).
    В каждом кадре кресты ищутся около мест, где их ставит решётка (центр — по симметрии креста);
    отклонения за вычетом среднего по кадру (сдвиг кадра) сравниваются с моделью k1·(u - c)·ρ²."""
    from .mask import Grid, snap_cross
    from .storage import ScanData
    from .stitch import load_positions
    from .util import Result
    g = Grid(folder)
    if not g.ready():
        raise ValueError("Mark the crosses first: mc.view(folder), key G.")
    st = ScanData(folder)
    st.fix = None                                             # мерить по невыпрямленным кадрам
    H, W = st.shape[:2]
    P = load_positions(folder, st)
    P = P - P.min(0)
    px = st.meta.get('pixel_size_um') or 1.0
    r = int(min(150, max(8, 15.0 / px)))
    c = np.array([W / 2, H / 2])
    Rd = float(np.hypot(W / 2, H / 2))
    E, X = [], []
    for k in range(len(st)):
        lo, hi = P[k] + r * 2, P[k] + (W, H) - r * 2
        pts = [(i, j) for i, j in g.cells_range(lo, hi)
               if np.all(g.cr.point(i, j) >= lo) and np.all(g.cr.point(i, j) <= hi)]
        if len(pts) < 2:
            continue
        img = None
        d_, u_ = [], []
        for i, j in pts:
            u = g.cr.point(i, j) - P[k]
            x0, y0 = int(round(u[0])) - 2 * r, int(round(u[1])) - 2 * r
            if img is None:
                img = st.frame(k)
                img = img if img.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2GRAY)
            win = img[y0:y0 + 4 * r + 1, x0:x0 + 4 * r + 1]
            if win.shape != (4 * r + 1, 4 * r + 1):
                continue
            d, resp = snap_cross(win, r, arm=g.arm)
            if resp < 0.1:
                continue
            d_.append(np.array([x0 + 2 * r, y0 + 2 * r]) + d - u)      # где крест в кадре - где по решётке
            u_.append(u)
        if len(d_) < 2:
            continue
        d_, u_ = np.array(d_), np.array(u_)
        v = u_ - c
        f = v * (np.sum(v ** 2, 1) / Rd ** 2)[:, None]            # смещение при k1 = 1 (кубическое)
        E.append(d_ - d_.mean(0))
        X.append(np.stack([v - v.mean(0), np.c_[-v[:, 1], v[:, 0]] - np.c_[-v[:, 1], v[:, 0]].mean(0),
                           f - f.mean(0)], -1))             # масштаб и поворот решётки — отдельно от дисторсии
    st.close()
    if not E:
        raise ValueError("No frame with 2 or more crosses found.")
    E, X = np.vstack(E), np.vstack(X)                           # E: (n, 2); X: (n, 2, 3)
    coef = np.linalg.lstsq(X.reshape(-1, 3), E.reshape(-1), rcond=None)[0]
    k1 = float(coef[2])
    lin = X[..., :2] @ coef[:2]
    before = float(np.sqrt(((E - lin) ** 2).sum(1).mean()))
    after = float(np.sqrt(((E - X @ coef) ** 2).sum(1).mean()))
    corner = k1 * Rd
    worth = abs(corner) >= 0.5 and after < 0.8 * before
    d = dict(k1=k1, corner_px=corner, crosses=int(len(E)), rms_before_px=before, rms_after_px=after,
             frame_px=[W, H], apply=bool(worth))
    if save:
        os.makedirs(os.path.join(folder, 'join'), exist_ok=True)
        gp = _dist_path(st.meta) if worth else None
        for p in (os.path.join(folder, 'join', 'distortion.json'),) + ((gp,) if gp else ()):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(d, f, indent=1)
    return Result("Distortion measured", **{"Shift in the corner, px": round(corner, 2), "k1": f"{k1:.2e}",
                  "Crosses": len(E), "Crosses off, px rms": f"{before:.2f} -> {after:.2f}",
                  "Used": "yes: run mc.stitch(folder) again" if worth else "no: too small to matter"})


# ------------------------------------------------------------------ многополосное смешивание
def multiband(images, weights, levels=5):
    """Смешивание по частотам (пирамида Лапласа): мелкие детали — от одного кадра (без двоения),
    плавная яркость — смешивается широко (без ступеньки на шве). images — список HxWx3 float32,
    weights — список HxW (>=0). Каждый пиксель отдаётся кадру с наибольшим весом, затем маски сглаживаются
    по уровням пирамиды."""
    h, w = weights[0].shape
    levels = max(1, min(levels, int(np.log2(max(2, min(h, w)))) - 3))
    W = np.array(weights)
    valid = W.max(0) > 1e-6
    win = np.argmax(W, 0)
    out = None
    masks = [((win == k) & (W[k] > 1e-6)).astype(np.float32) for k in range(len(images))]
    msum = None
    for img, mk in zip(images, masks):
        gp_m = [mk]
        for _ in range(levels):
            gp_m.append(cv2.pyrDown(gp_m[-1]))
        gp = [img]
        for _ in range(levels):
            gp.append(cv2.pyrDown(gp[-1]))
        lp = [gp[i] - cv2.pyrUp(gp[i + 1], dstsize=gp[i].shape[1::-1]) for i in range(levels)] + [gp[levels]]
        part = [l * m[..., None] for l, m in zip(lp, gp_m)]
        if out is None:
            out, msum = part, [m.copy() for m in gp_m]
        else:
            out = [a + b for a, b in zip(out, part)]
            msum = [a + b for a, b in zip(msum, gp_m)]
    out = [o / np.maximum(s, 1e-6)[..., None] for o, s in zip(out, msum)]
    res = out[-1]
    for i in range(levels - 1, -1, -1):
        res = cv2.pyrUp(res, dstsize=out[i].shape[1::-1]) + out[i]
    return np.clip(res, 0, 255), valid
