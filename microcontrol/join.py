"""Сшивка (join) кадров карты в мозаику (вся карта или выбранная область).

1) начальные позиции — по координатам столика и масштабу кадра (камера ровная);
2) уточнение — фазовая корреляция перекрытий соседних кадров (читаются только полосы перекрытия);
3) глобальная подгонка — метод наименьших квадратов по всем парам (ошибки не копятся);
4) выравнивание освещения (flat-field) — убирает виньетирование и «клеточку»;
5) смешивание — плавные веса к краям (feather), без швов и без потери деталей.
Память ограничена max_mb: большая мозаика автоматически уменьшается, исходные кадры не трогаются.
"""
import csv, os, time
import numpy as np
import cv2
from . import log as L
from .core import imwrite
from .errors import StoreError
from .storage import ScanData, meta_A
from .util import Result, Progress, save_image


def _gray(a):
    return a if a.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_BGR2GRAY)


def _by_crosses(st, ids, p, A, W, H, min_score=0.5):
    """Поправки положения кадров по крестам: (n, 2) px или None для кадров без найденных крестов.
    Кресты стоят на ровной решётке O + i·a·x + j·b·y (meta['crosses']). В каждом кадре крест ищется
    образцом (cross_template.png) около предсказанного по столику места; поправка кадра — медиана
    отклонений найденных крестов. После поправки кресты всех кадров ложатся ровно на решётку."""
    cr = st.meta.get('crosses') or {}
    tp = cr.get('template') or {}
    if 'v1_um' not in cr or not tp.get('file'):
        return None, "no cross grid or no cross picture: mark crosses with P in m.manual() before the scan"
    tm = cv2.imdecode(np.fromfile(os.path.join(st.folder, tp['file']), np.uint8), cv2.IMREAD_COLOR)
    if tm is None:
        return None, "cross picture not readable"
    k = tp['pixel_size_um'] / st.meta['pixel_size_um']                 # образец -> масштаб сохранённых кадров
    tm = _gray(cv2.resize(tm, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)).astype(np.float32)
    th, tw = tm.shape
    d = st.meta['motor_step_um']
    O, v1, v2 = np.array(cr['origin_um']), np.array(cr['v1_um']), np.array(cr['v2_um'])
    a, b = np.linalg.norm(v1), np.linalg.norm(v2)
    i0, i1, j0, j1 = cr['extent']
    I, J = np.meshgrid(np.arange(i0 - 1, i1 + 2), np.arange(j0 - 1, j1 + 2))
    S = O + I.ravel()[:, None] * v1 + J.ravel()[:, None] * v2           # кресты, мкм (настоящие), с поворотом
    pitch_px = min(a, b) / st.meta['pixel_size_um']
    r = int(min(0.35 * pitch_px, 0.15 * min(W, H), 30.0 / st.meta['pixel_size_um']))   # поиск ±30 мкм
    c = np.array([W / 2, H / 2])
    corr, found = [None] * len(ids), 0
    for n_, i in enumerate(ids):
        u = c + (A @ (p[n_][:, None] - S.T / d)).T                      # где кресты в этом кадре, px
        ok = (u[:, 0] >= tw / 2 + r) & (u[:, 0] < W - tw / 2 - r) & (u[:, 1] >= th / 2 + r) & (u[:, 1] < H - th / 2 - r)
        if not ok.any():
            continue
        g = _gray(st.frame(i)).astype(np.float32)
        es = []
        for ux, uy in u[ok]:
            x0, y0 = int(round(ux - tw / 2 - r)), int(round(uy - th / 2 - r))
            win = g[y0:y0 + th + 2 * r, x0:x0 + tw + 2 * r]
            res = cv2.matchTemplate(win, tm, cv2.TM_CCOEFF_NORMED)
            _, mx, _, (bx, by) = cv2.minMaxLoc(res)
            if mx < min_score:
                continue
            dx = dy = 0.0                                                # доля пикселя: парабола по соседям
            if 0 < bx < res.shape[1] - 1:
                l_, c_, r_ = res[by, bx - 1], res[by, bx], res[by, bx + 1]
                dx = 0.5 * (l_ - r_) / (l_ - 2 * c_ + r_ + 1e-9)
            if 0 < by < res.shape[0] - 1:
                l_, c_, r_ = res[by - 1, bx], res[by, bx], res[by + 1, bx]
                dy = 0.5 * (l_ - r_) / (l_ - 2 * c_ + r_ + 1e-9)
            es.append((x0 + bx + dx + tw / 2 - ux, y0 + by + dy + th / 2 - uy))
        if es:
            corr[n_] = np.median(np.array(es), 0)
            found += len(es)
    return corr, f"{found} crosses in {sum(e is not None for e in corr)} of {len(ids)} frames"


def join(folder, ids=None, x=None, y=None, center=None, scale=None, flatfield=None, blend=True,
         max_mb=3000, min_response=0.08, name=None, crosses='auto', fmt='jpg', flip=''):
    """Склеить кадры скана в одну картинку. Область: ids=[...] | x=(x0,x1), y=(y0,y1) мкм | center=0.3 (центральные 30%).
    flip='h' / 'v' / 'hv' — исправить скан, снятый с неверным знаком оси (кадры в ряду в обратном порядке:
    9 8 7 вместо 7 8 9 — flip='h'). Новые сканы проверяют знаки сами (m.check_orientation()).
    crosses='auto' — если перед сканом отмечены кресты (P в m.manual()), кадры ставятся так, чтобы кресты
    легли ровно на решётку; False — только по перекрытиям кадров.
    Пишет в <скан>/join/: mosaic_<имя>.jpg (fmt='png' — без потерь), preview_<имя>.jpg, positions_<имя>.csv."""
    t_start = time.monotonic()
    st = ScanData(folder)
    if ids is None:
        ids = st.query(x=x, y=y, center=center) if (x or y or center) else np.arange(len(st))
    ids = np.asarray(ids, int)
    if len(ids) == 0:
        raise StoreError('E801', f"no frames in x={x} y={y} center={center}")
    tag = name or ('all' if len(ids) == len(st) else f"{len(ids)}tiles_{int(st.x[ids].mean())}_{int(st.y[ids].mean())}")
    
    H, W = st.shape[:2]
    C = 1 if len(st.shape) == 2 else st.shape[2]
    p = np.array([[st.index[i]['x_p'], st.index[i]['y_p']] for i in ids], float)
    A = meta_A(st.meta, W, H)
    if flip:
        A = np.diag([-1.0 if 'h' in flip else 1.0, -1.0 if 'v' in flip else 1.0]) @ A
    pred = -(p @ A.T)
    n = len(ids)

    # --- 2) пары соседей (векторно) и уточнение по полосам перекрытия ---
    pairs = []
    pr = Progress(n, "Join: matching", unit='frames')
    for a in range(n):
        pr.update(1)
        d = pred[a + 1:] - pred[a]
        js = np.flatnonzero((np.abs(d[:, 0]) < W - 16) & (np.abs(d[:, 1]) < H - 16)) + a + 1
        for b in js:
            dx, dy = (int(round(v)) for v in pred[b] - pred[a])
            xa, ya = max(0, dx), max(0, dy)
            w_, h_ = W - abs(dx), H - abs(dy)
            ta = _gray(st.frame(ids[a], (xa, ya, xa + w_, ya + h_))).astype(np.float32)
            tb = _gray(st.frame(ids[b], (xa - dx, ya - dy, xa - dx + w_, ya - dy + h_))).astype(np.float32)
            (ex, ey), resp = cv2.phaseCorrelate(ta, tb, cv2.createHanningWindow((w_, h_), cv2.CV_32F))
            ok = resp >= min_response and abs(ex) < 0.25 * w_ and abs(ey) < 0.25 * h_
            meas = np.array([dx, dy], float) - (ex, ey) if ok else pred[b] - pred[a]
            pairs.append((a, b, meas, resp if ok else 0.02))

    pr.close()
    # --- 2б) кресты: абсолютная привязка кадров к ровной решётке крестов ---
    corr, cross_note = (None, "off") if crosses is False else _by_crosses(st, ids, p, A, W, H)
    anchor_w = np.full(n, 0.01)
    anchor = pred.copy()
    if corr is not None:
        for k_, e in enumerate(corr):
            if e is not None:
                anchor[k_] = pred[k_] - e
                anchor_w[k_] = 5.0
    # --- 3) глобальная подгонка (МНК), отдельно по x и y ---
    pos = np.zeros((n, 2))
    if corr is not None and not pairs:
        pos = anchor.copy()
    elif pairs:
        I = np.array([q[0] for q in pairs]); J = np.array([q[1] for q in pairs])
        w = np.array([q[3] for q in pairs]); Mx = np.array([q[2] for q in pairs])
        k = len(pairs)
        M = np.zeros((k + n, n))
        M[np.arange(k), J] = w; M[np.arange(k), I] = -w
        M[k + np.arange(n), np.arange(n)] = anchor_w       # привязка: к крестам (сильная) или к столику (слабая)
        for c in range(2):
            rhs = np.r_[w * Mx[:, c], anchor_w * anchor[:, c]]
            pos[:, c] = np.linalg.lstsq(M, rhs, rcond=None)[0]
    else:
        pos = pred.copy()
    pos_abs = pos.copy()                                   # в системе координат столика (для окна)
    pos -= pos.min(0)
    good = sum(q[3] > 0.05 for q in pairs)

    # --- масштаб под память ---
    need = (pos[:, 0].max() + W) * (pos[:, 1].max() + H) * (C + 1) * 4 / 2 ** 20
    if scale is None:
        scale = min(1.0, (max_mb / need) ** 0.5)

    sw, sh = max(1, round(W * scale)), max(1, round(H * scale))
    P = np.round(pos * scale).astype(int)

    # --- 4) flat-field по выборке кадров ---
    if flatfield is None:
        flatfield = len(st) >= 12
    ff = None
    if flatfield:
        smp = np.linspace(0, len(st) - 1, min(40, len(st))).astype(int)
        ff = np.median(np.stack([st.frame(i).astype(np.float32) for i in smp]), 0)
        ff = cv2.GaussianBlur(ff, (0, 0), min(W, H) / 12)
        ff = np.maximum(ff / ff.mean(axis=(0, 1)), 0.2)
        if scale != 1:
            ff = cv2.resize(ff, (sw, sh), interpolation=cv2.INTER_AREA)
        ff = ff.reshape(sh, sw, C)

    # --- 5) смешивание (каждый кадр читается один раз, пачками) ---
    Hm, Wm = P[:, 1].max() + sh, P[:, 0].max() + sw
    acc = np.zeros((Hm, Wm, C), np.float32)
    wsum = np.zeros((Hm, Wm, 1), np.float32)
    rx = np.minimum(np.arange(sw) + 1, sw - np.arange(sw)).astype(np.float32)
    ry = np.minimum(np.arange(sh) + 1, sh - np.arange(sh)).astype(np.float32)
    wt = (np.outer(ry, rx) if blend else np.ones((sh, sw), np.float32))[..., None]
    where = {int(i): k for k, i in enumerate(ids)}
    pr = Progress(n, "Join: blending", unit='frames')
    for i, t in st.frames(ids):
        pr.update(1)
        x0, y0 = P[where[i]]
        t = t.astype(np.float32)
        if scale != 1:
            t = cv2.resize(t, (sw, sh), interpolation=cv2.INTER_AREA)
        t = t.reshape(sh, sw, C)
        if ff is not None:
            t = t / ff
        acc[y0:y0 + sh, x0:x0 + sw] += t * wt
        wsum[y0:y0 + sh, x0:x0 + sw] += wt
    pr.close()
    mosaic = np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)
    mosaic = mosaic[..., 0] if C == 1 else mosaic
    del acc, wsum

    out_dir = os.path.join(folder, 'join')
    os.makedirs(out_dir, exist_ok=True)
    if fmt == 'jpg' and max(mosaic.shape[:2]) > 65000:                 # предел формата JPG
        fmt = 'png'
    out = save_image(os.path.join(out_dir, f"mosaic_{tag}.{fmt if fmt == 'jpg' else 'png'}"), mosaic, None, "Join: saving")
    k = 3000 / max(mosaic.shape[:2])
    imwrite(os.path.join(out_dir, f"preview_{tag}.jpg"),
            cv2.resize(mosaic, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else mosaic)
    with open(os.path.join(out_dir, f"positions_{tag}.csv"), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f); wr.writerow(['id', 'x_px', 'y_px', 'scale'])
        for i, (px, py) in zip(ids, pos_abs):
            wr.writerow([int(i), f"{px:.2f}", f"{py:.2f}", f"{scale:.4f}"])
    st.close()
    dt = time.monotonic() - t_start
    L.event('JOI-001', f"{out} {mosaic.shape[1]}x{mosaic.shape[0]} px scale={scale:.3f} pairs={len(pairs)} refined={good} ({dt:.1f} s)")
    return Result("Joined", Frames=int(n), **{"Seams refined": f"{good} of {len(pairs)}",
                  "By crosses": cross_note,
                  "Mosaic, px": f"{mosaic.shape[1]} x {mosaic.shape[0]}", "Scale": round(scale, 3),
                  "Time, s": round(dt, 1)}, File=out)
