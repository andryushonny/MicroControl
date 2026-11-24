"""Оптические поправки кадров: неравномерность освещения (flat-field) и дисторсия объектива.

Освещение. Объектив и камера темнят края кадра (виньетирование), поэтому границы кадров видны в мозаике
по яркости. Поле освещения F(x, y) — медиана по многим кадрам (детали образца у разных кадров разные
и в медиане пропадают), сглаженная и нормированная на среднее; исправленный кадр I' = I / F.

Дисторсия. Точка ровного кадра u (от центра c: v = u - c, ρ² = |v|²/R², R — половина диагонали) в снятом
кадре стоит в u + Δ(v):
    Δ = k1·v·ρ² + k2·v·ρ⁴                                   — радиальная (бочка < 0 < подушка),
    Δx += (p1·2xy + p2·(3x² + y²)) / R,  Δy += (p1·(x² + 3y²) + p2·2xy) / R   — тангенциальная (децентровка).
k1..p2 находятся по крестам (решётка с точным шагом — линейка) вместе с положениями кадров: сшивка по
крестам (anchor.py) после разметки крестов в mc.view. Значение хранится для объектива и применяется
при сшивке и выводе.

Поправки подключаются к ScanData автоматически (attach), если в папке скана есть join/flatfield.npy
или для объектива измерена дисторсия.
"""
import json, os
import numpy as np
from .stats import timed_fn
import cv2

HOME = os.path.join(os.path.expanduser('~'), '.microcontrol')


# ------------------------------------------------------------------ освещение
@timed_fn('Stitch: light field')
def flatfield(st, n=60, down=4, verbose=False):
    """Поле освещения скана — отдельно по каждому цвету (края кадра бывают не только темнее, но и другого
    оттенка): медиана до n кадров, каждый сначала поделён на свою медиану (кадры с другой общей яркостью
    не портят поле), гауссово сглаживание, среднее каждого цвета = 1. Возвращает уменьшенное в down раз
    поле (h, w, 3) или (h, w) float32, или None, если кадров мало (< 8)."""
    N = len(st)
    if N < 8:
        return None
    ids = np.linspace(0, N - 1, min(n, N)).astype(int)
    fix, st.fix = getattr(st, 'fix', None), None              # считать по сырым кадрам
    gains, st.gains = getattr(st, 'gains', None), None
    from .util import Progress
    pr = Progress(len(ids), "Stitch: light", unit='frames') if verbose else None
    try:
        stack = []
        for i in ids:
            g = np.ascontiguousarray(st.frame(int(i)))
            if pr:
                pr.update(1, f"light field from frame {int(i) + 1}")
            g = cv2.resize(g, None, fx=1 / down, fy=1 / down, interpolation=cv2.INTER_AREA).astype(np.float32)
            med = np.median(g.reshape(-1, g.shape[2]) if g.ndim == 3 else g.ravel(), 0)
            stack.append(g / np.maximum(med, 1.0))
    finally:
        st.fix, st.gains = fix, gains
        if pr:
            pr.close()
    F = np.median(np.array(stack), 0)
    sig = min(F.shape[:2]) / 12
    F = cv2.GaussianBlur(F, (0, 0), sig)
    F = F / F.reshape(-1, F.shape[2]).mean(0) if F.ndim == 3 else F / F.mean()
    return np.clip(F, 0.2, 5).astype(np.float32)


@timed_fn('Stitch: frame brightness')
def frame_gains(st, pos, W, H, down=8, lam=0.05, verbose=False):
    """Выровнять яркость (и оттенок) кадров по перекрытиям, как экспозицию в панорамах: для каждой пары
    соседей медиана общего участка в обоих кадрах; множители кадров a_i — по МНК в логарифмах
    ln a_i − ln a_j = ln m_j − ln m_i, со слабой привязкой ln a_i → 0. Возвращает (n, каналы) или None."""
    n = len(st)
    if n < 2:
        return None
    from .util import Progress
    pr = Progress(n, "Stitch: brightness", unit='frames') if verbose else None
    small = []
    for i in range(n):
        f = np.ascontiguousarray(st.frame(i))
        small.append(cv2.resize(f, (max(1, W // down), max(1, H // down)), interpolation=cv2.INTER_AREA))
        if pr:
            pr.update(1, "reading overlaps")
    if pr:
        pr.stage("solving frame gains")
    C = small[0].shape[2] if small[0].ndim == 3 else 1
    pos = np.asarray(pos, float)
    rows, rhs = [], []
    for a in range(n):
        for b in range(a + 1, n):
            d = pos[b] - pos[a]
            ox, oy = W - abs(d[0]), H - abs(d[1])
            if ox < 0.1 * W or oy < 0.1 * H:
                continue
            # общий участок в координатах кадра a и кадра b (с полем 10 % от края перекрытия)
            ax0, ay0 = max(0.0, d[0]), max(0.0, d[1])
            bx0, by0 = max(0.0, -d[0]), max(0.0, -d[1])
            sl = lambda x0, y0: (slice(int(y0 / down) + 1, int((y0 + oy) / down) - 1),
                                 slice(int(x0 / down) + 1, int((x0 + ox) / down) - 1))
            ra, rb = small[a][sl(ax0, ay0)], small[b][sl(bx0, by0)]
            if ra.size == 0 or rb.size == 0:
                continue
            ma = np.median(ra.reshape(-1, C), 0).astype(float)
            mb = np.median(rb.reshape(-1, C), 0).astype(float)
            if np.any(ma < 5) or np.any(mb < 5):
                continue
            rows.append((a, b))
            rhs.append(np.log(mb) - np.log(ma))
    if not rows:
        if pr:
            pr.close()
        return None
    M = np.zeros((len(rows) + n, n))
    for k, (a, b) in enumerate(rows):
        M[k, a], M[k, b] = 1.0, -1.0
    M[len(rows):] = np.eye(n) * lam
    R = np.vstack([np.array(rhs), np.zeros((n, C))])
    lg, *_ = np.linalg.lstsq(M, R, rcond=None)
    if pr:
        pr.stage(f"{len(rows)} overlaps, gains solved")
        pr.close()
    lg -= np.median(lg, 0)                                     # средний кадр — без изменений
    return np.exp(np.clip(lg, -0.7, 0.7)).astype(np.float32)


# ------------------------------------------------------------------ дисторсия
def _dist_path(meta):
    """Файл дисторсии для микроскопа (по серийному номеру) и объектива; у симулятора — нет."""
    serial = str((meta.get('device') or {}).get('Serial', 'unknown'))
    if serial.upper().startswith('SIM') or not meta.get('objective'):
        return None
    safe = ''.join(ch if ch.isalnum() else '_' for ch in serial)
    return os.path.join(HOME, 'devices', f"distortion_{safe}_{meta['objective']}x.json")


DIST_KEYS = ('k1', 'k2', 'p1', 'p2')


def dist_basis(v, R):
    """Смещения от единичных k1, k2, p1, p2 в точках v (от центра кадра, px): массив (n, 2, 4)."""
    v = np.atleast_2d(np.asarray(v, float))
    x, y = v[:, 0], v[:, 1]
    q = (x * x + y * y) / (R * R)
    return np.stack([v * q[:, None], v * (q * q)[:, None],
                     np.c_[2 * x * y, x * x + 3 * y * y] / R, np.c_[3 * x * x + y * y, 2 * x * y] / R], -1)


def dist_vec(D):
    return np.array([float((D or {}).get(k, 0.0)) for k in DIST_KEYS])


def dist_shift(v, D, R):
    """Δ(v): где точка ровного кадра оказывается в снятом (минус где она должна быть), px."""
    return dist_basis(v, R) @ dist_vec(D)


def dist_corner(D, W, H):
    """Наибольшее смещение по краю кадра, px (насколько дисторсия заметна)."""
    t = np.linspace(-1, 1, 9)
    e = np.r_[np.c_[t * W / 2, np.full(9, -H / 2)], np.c_[t * W / 2, np.full(9, H / 2)],
              np.c_[np.full(9, -W / 2), t * H / 2], np.c_[np.full(9, W / 2), t * H / 2]]
    return float(np.hypot(*dist_shift(e, D, float(np.hypot(W / 2, H / 2))).T).max())


def _dist_path(meta):
    """Файл дисторсии для микроскопа (по серийному номеру) и объектива; у симулятора — нет."""
    serial = str((meta.get('device') or {}).get('Serial', 'unknown'))
    if serial.upper().startswith('SIM') or not meta.get('objective'):
        return None
    safe = ''.join(ch if ch.isalnum() else '_' for ch in serial)
    return os.path.join(HOME, 'devices', f"distortion_{safe}_{meta['objective']}x.json")


def _pix_path(device, objective):
    serial = str((device or {}).get('Serial', 'unknown'))
    if not objective or (serial.upper().startswith('SIM') and not os.environ.get('MC_SIM_PIXEL_CAL')):
        return None                                   # симулятор: только для проверки калибровки (MC_SIM_PIXEL_CAL=1)
    safe = ''.join(ch if ch.isalnum() else '_' for ch in serial)
    return os.path.join(HOME, 'devices', f"pixel_{safe}_{objective}x.json")


def pixel_cal(device, objective):
    """Размер пикселя объектива, откалиброванный по крестам маски (anchor: шаг маски / шаг решётки в px),
    или None."""
    p = _pix_path(device, objective)
    if p and os.path.exists(p):
        try:
            return json.load(open(p, encoding='utf-8'))
        except (OSError, ValueError):
            return None
    return None


def measured_flat(meta, W, H):
    """Плоское поле и тёмный кадр, снятые кнопками Flat field / Dark frame для объектива скана (meta['references']):
    -> (F — поле (кадр − тёмный), среднее каждого цвета = 1; D — тёмный кадр) под размер кадра скана, или (None, None)."""
    refs = meta.get('references') or {}
    obj = meta.get('objective')
    fp, dp = refs.get(f"empty_{obj}x"), refs.get(f"dark_{obj}x")
    if not fp or not os.path.exists(fp):
        return None, None
    rd = lambda p: cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
    E = rd(fp)
    Dk = rd(dp) if dp and os.path.exists(dp) else None
    if Dk is not None and Dk.shape != E.shape:
        Dk = None
    F = E - Dk if Dk is not None else E.copy()
    F = cv2.GaussianBlur(F, (0, 0), 1.0)                   # шум усреднённого кадра — прочь, пыль и виньетка остаются
    F = np.maximum(F, 1e-3)
    F /= F.reshape(-1, F.shape[2]).mean(0) if F.ndim == 3 else F.mean()
    F = cv2.resize(F, (W, H), interpolation=cv2.INTER_AREA)
    if Dk is not None:
        Dk = cv2.resize(Dk, (W, H), interpolation=cv2.INTER_AREA)
    return F.astype(np.float32), (Dk.astype(np.float32) if Dk is not None else None)


def load_dist(folder, meta):
    """Дисторсия скана {k1, k2, p1, p2}: join/distortion.json в папке скана (сшивка по крестам), иначе
    калибровка объектива на момент съёмки (meta), иначе измеренная для этого микроскопа и объектива; нет — нули."""
    for p in (os.path.join(folder, 'join', 'distortion.json'), 'meta', _dist_path(meta)):
        if p == 'meta':                                     # калибровка объектива на момент съёмки
            d = meta.get('distortion') or None
        elif p and os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                d = json.load(f)
        else:
            d = None
        if d and d.get('apply', True):
            return {k: float(d.get(k, 0.0)) for k in DIST_KEYS}
    return {k: 0.0 for k in DIST_KEYS}


def undistort_maps(W, H, D):
    """Карты cv2.remap: для пикселя выпрямленного кадра — где он в снятом кадре."""
    c = np.array([W / 2, H / 2], np.float32)
    R = float(np.hypot(W / 2, H / 2))
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    v = np.c_[(x - c[0]).ravel(), (y - c[1]).ravel()]
    d = dist_shift(v, D, R).astype(np.float32)
    return (x + d[:, 0].reshape(H, W)).astype(np.float32), (y + d[:, 1].reshape(H, W)).astype(np.float32)


# ------------------------------------------------------------------ подключение к ScanData
def attach(st, folder):
    """Поправить чтение кадров скана: I / F, затем выпрямление дисторсии (если они есть)."""
    W, H = st.shape[1], st.shape[0]
    fp = os.path.join(folder, 'join', 'flatfield.npy')
    F = None
    if os.path.exists(fp):
        F = cv2.resize(np.load(fp), (W, H), interpolation=cv2.INTER_LINEAR)
    dkp = os.path.join(folder, 'join', 'dark.npy')
    Dk = cv2.resize(np.load(dkp), (W, H), interpolation=cv2.INTER_LINEAR) if os.path.exists(dkp) and F is not None else None
    D = load_dist(folder, st.meta)
    maps = undistort_maps(W, H, D) if dist_corner(D, W, H) >= 0.2 else None
    gp = os.path.join(folder, 'join', 'gains.npy')
    st.gains = np.load(gp) if os.path.exists(gp) else None     # яркость кадров, выровненная по перекрытиям
    if st.gains is not None and len(st.gains) != len(st):
        st.gains = None
    if F is None and maps is None and st.gains is None:
        st.fix = None
        return st

    def fix(img):
        out = img
        if F is not None:
            if img.ndim == 2:
                f = F if F.ndim == 2 else F.mean(2)
            else:
                f = F if F.ndim == 3 else F[..., None]
            src = img.astype(np.float32)
            if Dk is not None:                                     # тёмный кадр: шум и горячие пиксели матрицы
                src = src - (Dk if Dk.ndim == src.ndim else (Dk.mean(2) if src.ndim == 2 else Dk[..., None]))
            out = np.clip(src / f, 0, 255).astype(np.uint8)
        if maps is not None:
            out = cv2.remap(out, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out

    st.fix = fix
    st.fix_info = dict(flatfield=F is not None, dist=D if maps is not None else None,
                       corner_px=dist_corner(D, W, H) if maps is not None else 0.0, gains=st.gains is not None)
    return st


def measure_distortion(folder, save=True):
    """Дисторсия объектива по крестам — это сшивка по крестам (anchor.anchor): положения кадров, решётка
    и k1, k2, p1, p2 подбираются вместе. Нужны: mc.stitch(folder) и кресты, отмеченные в mc.view (C)."""
    from .anchor import anchor
    return anchor(folder, save=save)


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
