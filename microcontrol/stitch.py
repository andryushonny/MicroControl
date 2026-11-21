"""Сшивка скана (по идее ASHLAR: Muhlich et al., Bioinformatics 2022, MIT).

1) соседи — по координатам столика: кадры, чьи предсказанные прямоугольники перекрываются;
2) сдвиг каждой пары — фазовой корреляцией по полосе перекрытия (с запасом max_shift);
   качество — ошибка e = 1 - NCC совмещённых полос;
3) порог ошибки — тест перестановок: те же полосы, но от НЕ соседних кадров (заведомо ложные пары);
   пара принимается, если её ошибка меньше 1-го процентиля ложных (alpha = 0.01);
4) модель «столик -> пиксели» (матрица 2x2: поворот камеры, масштаб, перекос) — по принятым парам;
5) положения — МНК по принятым парам + слабая привязка к модели; пары с невязкой > 3 px отбрасываются,
   подгонка повторяется. Кадры без принятых пар ставятся по модели (как у ASHLAR).

Результат: <скан>/join/positions_all.csv (x_px, y_px верхнего левого угла кадра) и stitch.json;
mc.view(folder) показывает скан по этим положениям, мозаика целиком в память не собирается.
engine='ashlar' — настоящий ASHLAR (pip install ashlar; нужна Java).
"""
from .samples import per_sample as _per_sample
import math, csv, json, os, time
import numpy as np
import cv2
from . import log as L
from .storage import ScanData, meta_A, _thumb
from .util import Result, Progress


def _gray(a):
    a = a if a.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_BGR2GRAY)
    return a.astype(np.float32)


def _whiten(a, sigma):
    """Как в ASHLAR: подавить плавный фон (лапласиан гауссиана) — корреляция по деталям, не по яркости."""
    return -cv2.Laplacian(cv2.GaussianBlur(a, (0, 0), sigma), cv2.CV_32F) if sigma > 0 else a - a.mean()


def _pair(st, ia, ib, d, W, H, pad, sigma, max_shift, other=None):
    """Сдвиг кадра ib относительно ia (px, левый верхний угол b минус a) по полосам перекрытия.
    d — предсказание по столику; полосы берутся с запасом pad в каждом кадре отдельно (у краёв —
    сколько есть), дополняются до одного размера. other — подставной кадр вместо ib (тест перестановок).
    Возвращает (сдвиг, ошибка = 1 - NCC совмещённых полос) или (None, 2.0)."""
    dx, dy = int(round(d[0])), int(round(d[1]))
    # как в ASHLAR: общая часть по предсказанию, в обоих кадрах одного размера
    ax0, ax1 = max(0, dx), min(W, W + dx)
    ay0, ay1 = max(0, dy), min(H, H + dy)
    bx0, bx1, by0, by1 = ax0 - dx, ax1 - dx, ay0 - dy, ay1 - dy
    if min(ax1 - ax0, ay1 - ay0) < 16:
        return None, 2.0
    a = _whiten(_gray(st.frame(ia, (ax0, ay0, ax1, ay1))), sigma)
    b = _whiten(_gray(st.frame(ib if other is None else other, (bx0, by0, bx1, by1))), sigma)
    # окно Ханна на каждую полосу, затем дополнение нулями вдвое: корреляция без «заворота»
    h, w = 2 * max(a.shape[0], b.shape[0]), 2 * max(a.shape[1], b.shape[1])
    A_, B_ = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    A_[:a.shape[0], :a.shape[1]] = (a - a.mean()) * cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    B_[:b.shape[0], :b.shape[1]] = (b - b.mean()) * cv2.createHanningWindow(b.shape[::-1], cv2.CV_32F)
    (sx, sy), _ = cv2.phaseCorrelate(A_, B_)                   # B_(x) = A_(x - s)
    dt = np.array([ax0 - bx0 - sx, ay0 - by0 - sy])            # истинный сдвиг b относительно a
    if np.hypot(*(dt - (dx, dy))) > max_shift:
        return None, 2.0
    # NCC на общей части: точка a-окна (i) <-> b-окна (i + ax0 - bx0 - dt)
    ox, oy = ax0 - bx0 - dt[0], ay0 - by0 - dt[1]
    M = np.float32([[1, 0, ox], [0, 1, oy]])                    # bb(i) = b(i + o)
    bb = cv2.warpAffine(b, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderValue=np.nan)
    ok = ~np.isnan(bb)
    if ok.sum() < 256:
        return None, 2.0
    va, vb = a[ok] - a[ok].mean(), bb[ok] - bb[ok].mean()
    ncc = float((va * vb).sum() / (np.sqrt((va ** 2).sum() * (vb ** 2).sum()) + 1e-9))
    return dt, 1.0 - ncc


@_per_sample
def stitch(folder, engine='builtin', max_shift_um=40.0, sigma=None, alpha=0.01, preview=True, verbose=True,
           flat=True):
    """Сшить скан: найти точные положения кадров. engine='auto' | 'builtin' | 'ashlar'.
    max_shift_um — насколько столик может ошибиться между соседними кадрами."""
    t0 = time.monotonic()
    st = ScanData(folder)
    H, W = st.shape[:2]
    from .optics import flatfield, attach
    os.makedirs(os.path.join(folder, 'join'), exist_ok=True)
    fp = os.path.join(folder, 'join', 'flatfield.npy')
    gp = os.path.join(folder, 'join', 'gains.npy')
    if os.path.exists(gp):                                       # яркость кадров считается заново
        os.remove(gp)
    st.gains = None
    if flat:                                                     # поле освещения — по сырым кадрам
        F = flatfield(st)
        if F is not None:
            np.save(fp, F)
    elif os.path.exists(fp):
        os.remove(fp)
    attach(st, folder)                                           # освещение и дисторсия — при чтении кадров
    st._fixed.clear()
    cols = [r.get('col') for r in st.index if r.get('col') not in ('', None)]
    fb = int(np.prod(st.shape))                                  # байт на кадр
    st.cache_n = max(12, min((max(cols) + 1 if cols else 1) + 8,  # сосед по столбцу — через ряд: ряд целиком в памяти,
                             int(1.2e9 // max(fb, 1))))           # но не больше ~1.2 ГБ
    if st.fix is not None:
        st._cache.clear()
    n = len(st)
    meta = st.meta
    px = meta.get('pixel_size_um') or 1.0
    A = meta_A(meta, W, H)
    p = np.array([[r['x_p'], r['y_p']] for r in st.index], float)
    pred = -(p @ A.T)                                            # верхний левый угол кадра, px
    max_shift = max_shift_um / px
    if sigma is None:
        sigma = 1.0 if px < 0.5 else 0.0
    used = None
    if engine == 'ashlar':
        try:
            pos, info = _ashlar(st, pred, px, max_shift_um, sigma, verbose)
            used = 'ashlar'
        except Exception as e:
            if engine == 'ashlar':
                raise
            L.event('STI-002', f"ASHLAR not used ({type(e).__name__}: {e}); built-in engine.", 20)
    if used is None:
        pos, info = _builtin(st, pred, p, W, H, max_shift, sigma, alpha, verbose)
        used = 'builtin'
    out = os.path.join(folder, 'join')
    os.makedirs(out, exist_ok=True)
    P_before = load_positions(folder, st)                   # как кадры стояли до этой сшивки (для крестов)
    with open(os.path.join(out, 'positions_all.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['id', 'x_px', 'y_px', 'scale'])
        for i in range(n):
            wr.writerow([i, f"{pos[i, 0]:.2f}", f"{pos[i, 1]:.2f}", "1.0000"])
    info.update(engine=used, frames=n, frame_px=[W, H], pixel_size_um=px, time_s=round(time.monotonic() - t0, 1))
    with open(os.path.join(out, 'stitch.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1)
    r = Result("Stitched", Engine=used, Frames=n,
               **{"Seams accepted": f"{info.get('accepted', '-')} of {info.get('pairs', '-')}",
                  "Seam error, px": info.get('residual_px', '-'),
                  "Camera vs stage": info.get('camera', '-'),
                  "Backlash X, Y, um": info.get('backlash_um', '-'),
                  "Corrections": f"light {'yes' if getattr(st, 'fix_info', {}).get('flatfield') else 'no'}, "
                                 f"distortion k1 {getattr(st, 'fix_info', {}).get('k1', 0.0):.2e}",
                  "Time, s": info['time_s']})
    if flat:                                                # яркость кадров по перекрытиям (после сшивки)
        try:
            from .optics import frame_gains
            gn = frame_gains(st, pos, W, H)
            if gn is not None:
                np.save(gp, gn)
                st.gains = gn
                st._fixed.clear()
                if st.fix is None:
                    attach(st, folder)
                r['Corrections'] += f", frame brightness evened (±{100 * float(np.abs(np.log(gn)).max()):.0f} %)"
        except Exception as e:
            L.event('STI-004', f"Frame brightness not evened: {type(e).__name__}: {e}", 30)
    try:                                                    # кресты уже стоят: перенести их вместе с кадрами
        from .mask import Grid
        if os.path.exists(os.path.join(folder, 'grid.json')):
            g_ = Grid(folder)
            if g_.cr.marks:
                if g_.frames_px is None or len(g_.frames_px) != n:
                    g_.frames_px = P_before - P_before.min(0)
                nmv = g_.sync(pos - pos.min(0), W, H)
                if nmv:
                    g_.refined = None
                    g_.save(px)
                    r['Crosses'] = f"{nmv} moved with their frames (F in the viewer fits the grid again)"
    except Exception as e:                                  # сшивка важнее: кресты можно поставить заново
        L.event('STI-003', f"Crosses not moved: {type(e).__name__}: {e}", 30)
    if preview:
        r['Preview'] = make_preview(folder, pos)
    st.close()
    L.event('STI-001', f"{folder}: {dict(r)}")
    return r


def _builtin(st, pred, p, W, H, max_shift, sigma, alpha, verbose):
    n = len(pred)
    pad = int(max_shift)
    # 1) соседи
    pairs, area = [], []
    for a in range(n):
        d = pred[a + 1:] - pred[a]
        for b in np.flatnonzero((np.abs(d[:, 0]) < W - 16) & (np.abs(d[:, 1]) < H - 16)) + a + 1:
            pairs.append((a, int(b)))
            area.append((W - abs(d[b - a - 1, 0])) * (H - abs(d[b - a - 1, 1])))
    if pairs:                                                   # уголки по диагонали — малы и ненадёжны
        area = np.array(area)
        pairs = [q for q, ar in zip(pairs, area) if ar >= 0.3 * area.max()]
    if not pairs:
        return pred.copy(), dict(pairs=0, accepted=0, note="frames do not overlap: positions from the stage")
    # 2) сдвиги
    meas, err = [], []
    pr = Progress(len(pairs), "Stitch: seams") if verbose else None
    for a, b in pairs:
        dt, e = _pair(st, a, b, pred[b] - pred[a], W, H, pad, sigma, max_shift)
        meas.append(pred[b] - pred[a] if dt is None else dt); err.append(e)
        if pr:
            pr.update(1)
    if pr:
        pr.close()
    meas, err = np.array(meas), np.array(err)
    # 3) тест перестановок: та же геометрия, но вместо соседа — далёкий кадр (заведомо ложная пара)
    rng = np.random.default_rng(0)
    fake = []
    for _ in range(min(300, 3 * len(pairs))):
        a, b = pairs[rng.integers(len(pairs))]
        c = int(rng.integers(n))
        dc = np.abs(pred[c] - pred[a])
        if c in (a, b) or (dc[0] < 1.5 * W and dc[1] < 1.5 * H):
            continue
        fake.append(_pair(st, a, b, pred[b] - pred[a], W, H, pad, sigma, 1e9, other=c)[1])
    thr = float(np.quantile(fake, alpha)) if len(fake) >= 20 else 0.7
    ok = err < min(thr, 0.99)
    # 4) модель столик -> пиксели по принятым парам: d = A·Δp
    I = np.array([q[0] for q in pairs]); J = np.array([q[1] for q in pairs])
    dp = p[J] - p[I]
    Ast = np.linalg.lstsq(p - p.mean(0), pred - pred.mean(0), rcond=None)[0].T   # -A по столику
    model = Ast
    # люфт: с какой стороны столик подъехал к кадру (знак последнего движения по каждой оси)
    sg = np.zeros((n, 2))
    last = np.zeros(2)
    for k in range(1, n):
        dk = np.sign(p[k] - p[k - 1])
        last = np.where(dk != 0, dk, last)
        sg[k] = last
    sg[0] = sg[1] if n > 1 else 0
    dsg = sg[J] - sg[I]
    Bl = np.zeros((2, 2))                                       # px на смену направления по X и по Y
    if ok.sum() >= 3:                                           # d = M·Δp + Bl·Δs по принятым швам
        cols = [dp[ok]] + ([dsg[ok]] if np.abs(dsg[ok]).sum() > 0 else [])
        sol = np.linalg.lstsq(np.hstack(cols), meas[ok], rcond=None)[0]
        model = sol[:2].T
        if len(cols) > 1:
            Bl = sol[2:].T / 2.0                                # Δs = ±2 при смене направления
    base = p @ model.T + sg @ Bl.T
    base += (pred - base).mean(0)
    # 5) взвешенный МНК с подавлением выбросов (веса Коши по невязке): шов, не согласный с остальными,
    #    теряет вес; кадр без надёжных швов держится моделью столика
    w0 = np.where(err < 1.0, np.clip(1.0 - err, 0.05, 1.0), 0.0)
    w0[ok] = np.maximum(w0[ok], 0.5)
    w = w0.copy()
    pos = base.copy()
    for cs in (1e9, 30.0, 15.0, 8.0, 5.0, 3.0, 2.0, 2.0):         # масштаб Коши уменьшается постепенно
        k = len(pairs)
        M = np.zeros((k + n, n))
        M[np.arange(k), J] = w
        M[np.arange(k), I] = -w
        M[k + np.arange(n), np.arange(n)] = 0.01
        for c in range(2):
            pos[:, c] = np.linalg.lstsq(M, np.r_[w * meas[:, c], 0.01 * base[:, c]], rcond=None)[0]
        res = np.hypot(*(pos[J] - pos[I] - meas).T)
        w = w0 / (1.0 + (res / cs) ** 2)
    keep = (w0 > 0) & (res < 3.0)
    res_k = res[keep] if keep.any() else np.array([np.nan])
    R_ = np.linalg.solve(Ast, model)                              # поворот камеры относительно принятой ориентации
    ang = np.degrees(np.arctan2(R_[1, 0] - R_[0, 1], R_[0, 0] + R_[1, 1]))
    sc = np.sqrt(abs(np.linalg.det(model)))
    bl_um = np.hypot(*Bl.T) * 2 * (st.meta.get('pixel_size_um') or 1.0)
    return pos, dict(pairs=len(pairs), accepted=int(keep.sum()), threshold=round(thr, 3),
                     backlash_um=[round(float(v), 2) for v in bl_um],
                     residual_px=f"{np.nanmedian(res_k):.2f} median, {np.nanmax(res_k):.2f} max",
                     camera=f"turn {ang:+.2f} deg, {sc:.4f} px/step", model=model.tolist())


def _ashlar(st, pred, px, max_shift_um, sigma, verbose):
    """Настоящий ASHLAR (EdgeAligner) на наших кадрах: положения — из предсказания по столику."""
    from ashlar import reg

    class Meta(reg.Metadata):
        @property
        def _num_images(self):
            return len(pred)

        @property
        def num_channels(self):
            return 1

        @property
        def pixel_size(self):
            return px

        @property
        def pixel_dtype(self):
            return np.dtype(np.uint8)

        def tile_position(self, i):
            return np.array([pred[i, 1], pred[i, 0]])            # ASHLAR: (y, x), пиксели

        def tile_size(self, i):
            return np.array(st.shape[:2])

    class Rd(reg.Reader):
        def __init__(self):
            self.metadata = Meta()

        def read(self, series, c):
            a = st.frame(series)
            return a if a.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_BGR2GRAY)

    al = reg.EdgeAligner(Rd(), channel=0, max_shift=max_shift_um, filter_sigma=sigma, verbose=verbose)
    al.run()
    pos = np.asarray(al.positions)[:, ::-1].astype(float)          # (y, x) -> (x, y)
    pos += pred.min(0) - pos.min(0)
    return pos, dict(pairs=len(al.neighbors_graph.edges), accepted=len(al.spanning_tree.edges))


def make_preview(folder, pos=None, max_side=4000, levels=True):
    """Уменьшенная мозаика (JPG) для быстрого взгляда: кадры уменьшаются и кладутся по положениям.
    Заодно (levels=True) каждый кадр пишется уменьшенным в 2, 4, 8 раз в viewer/levels/ — просмотр больших
    сканов на слабом ПК берёт оттуда маленькие файлы, а не раскрывает целые кадры."""
    st = ScanData(folder)
    H, W = st.shape[:2]
    lvdir = os.path.join(folder, 'viewer', 'levels')
    if levels:
        os.makedirs(lvdir, exist_ok=True)
    if pos is None:
        pos = load_positions(folder, st)
    pos = pos - pos.min(0)
    size = pos.max(0) + (W, H)
    k = min(1.0, max_side / size.max())
    out = np.full((int(size[1] * k) + 2, int(size[0] * k) + 2, 3), 0, np.uint8)
    sw, sh = max(1, int(math.ceil(W * k)) + 1), max(1, int(math.ceil(H * k)) + 1)   # с нахлёстом: без щелей
    thumbs = [None] * len(st) if (levels and st.index[0].get('file')) else None   # JPG-кадры: превью для view
    for i, img in st.frames():
        if levels:
            lv_img = img
            for lv in (1, 2, 3):
                lv_img = cv2.resize(lv_img, (max(1, W >> lv), max(1, H >> lv)), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode('.jpg', lv_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                buf.tofile(os.path.join(lvdir, f"{i:05d}_{lv}.jpg"))
        if thumbs is not None:
            thumbs[i] = _thumb(img)
        t = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        t = t if t.ndim == 3 else cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        x, y = int(pos[i, 0] * k), int(pos[i, 1] * k)
        out[y:y + sh, x:x + sw] = t[:out.shape[0] - y, :out.shape[1] - x]
    if thumbs is not None:
        np.savez(os.path.join(folder, 'viewer', 'thumbs.npz'), **{f"t{i}": a for i, a in enumerate(thumbs)})
    path = os.path.join(folder, 'join', 'preview_all.jpg')
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    buf.tofile(path)
    st.close()
    return path


def load_positions(folder, st=None):
    """Положения кадров (px): после stitch — уточнённые, иначе — по столику."""
    own = st is None
    st = st or ScanData(folder)
    H, W = st.shape[:2]
    p = np.array([[r['x_p'], r['y_p']] for r in st.index], float)
    P = -(p @ meta_A(st.meta, W, H).T)
    jp = os.path.join(folder, 'join', 'positions_all.csv')
    if os.path.exists(jp):
        with open(jp, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                P[int(r['id'])] = (float(r['x_px']), float(r['y_px']))
    if own:
        st.close()
    return P
