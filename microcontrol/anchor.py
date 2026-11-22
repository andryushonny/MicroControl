"""Сшивка по крестам: кресты литографии — линейка с точным шагом, к ней привязывается каждый кадр.

Обычная сшивка (stitch.py) ставит кадры по швам между соседями; ошибка шва мала, но вдоль мозаики
складывается, а дисторсия объектива сдвигает края кадров. Здесь всё решается одним МНК:

    неизвестные: положения кадров P_k, решётка (O, a: шаг и наклон; второй вектор — a, повёрнутый на 90°),
                 дисторсия θ = (k1, k2, p1, p2) — см. optics.py;
    крест (i, j), найденный в СЫРОМ кадре k в точке r:   P_k + r − Δθ(u) = O + i·a + j·rot90(a),
                 u = r − Δθ(u) — где он в ровном кадре;
    шов кадров a, b (сдвиг d, измерен при дисторсии θ0):  P_b − P_a + θ·G_ab = d + θ0·G_ab,
                 G_ab — разность смещений Δ по полосе перекрытия в кадрах a и b при единичных θ.

Все уравнения линейны по неизвестным (Δ — по θ); u уточняется 2-3 раза. Выбросы (крест не нашёлся,
грязь у креста, ложный шов) теряют вес (Коши) и отбрасываются. Кадр с крестом стоит по решётке точно,
кадры без крестов держатся швами между привязанными. Решётка остаётся строго квадратной — это и есть
опора: шаг креста на маске точен до долей нанометра на миллиметр.

Итог: join/positions_all.csv, join/distortion.json, сетка (grid.json) — по найденной решётке, отчёт:
доля крестов, которые после сшивки лежат на узле сетки точнее 0.5 мкм.
"""
import json, os, csv
import numpy as np
import cv2
from . import log as L
from .util import Result, Progress
from .stats import timed_fn

TOL_UM = 0.5                         # крест «на сетке», если от узла не дальше, мкм


def _detect(st, g, P, D0, px, verbose):
    """Кресты в сырых кадрах около мест, где их ставит сетка. -> (k, i, j, r) — r в px сырого кадра."""
    from .mask import snap_cross
    from .optics import dist_shift
    H, W = st.shape[:2]
    c = np.array([W / 2, H / 2])
    R = float(np.hypot(W / 2, H / 2))
    rr = max(8.0, 15.0 / px)                                # поиск ±15 мкм вокруг узла
    s = min(1.0, 4.0 * px)                                  # 50x, 100x: крест крупнее перебора плеч — уменьшить
    m = 6.0 / px                                            # узел не ближе 6 мкм к краю кадра (полкреста видно)
    arm = g.arm * s if g.arm else None
    arms = []
    out = []
    pr = Progress(len(st), "Crosses: finding", unit='frames') if verbose else None
    for k in range(len(st)):
        if pr:
            pr.update(1, f"crosses found: {len(out)}")
        lo, hi = P[k] + m, P[k] + (W, H) - m
        nodes = []
        for i, j in g.cells_range(lo, hi):
            q = g.cr.point(i, j)
            if np.all(q >= lo) and np.all(q <= hi):
                nodes.append((i, j, q - P[k]))
        if not nodes:
            continue
        img = st.frame(k)
        img = img if img.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2GRAY)
        for i, j, u in nodes:
            rp = u + dist_shift(u - c, D0, R)[0]             # где узел в сыром кадре
            x0, y0 = (np.maximum(0, np.round(rp - 2 * rr))).astype(int)
            x1, y1 = (np.minimum((W, H), np.round(rp + 2 * rr) + 1)).astype(int)
            win = img[y0:y1, x0:x1].astype(np.float32)
            if min(win.shape) < 16 or win.std() < 2:
                continue
            h_, w_ = win.shape
            if s < 1:
                win = cv2.resize(win, (max(16, int(round(w_ * s))), max(16, int(round(h_ * s)))),
                                 interpolation=cv2.INTER_AREA)
            sx, sy = win.shape[1] / w_, win.shape[0] / h_
            mid = (np.array(win.shape[::-1], float) - 1) / 2
            o = (rp - (x0, y0) + 0.5) * (sx, sy) - 0.5 - mid  # узел относительно середины окна
            info = {}
            d, resp = snap_cross(win, (rr * s + float(np.hypot(*o))), arm=arm, info=info)
            if resp < 0.1 or not np.any(d) or np.hypot(*(d - o)) > rr * s:
                continue
            q = (mid + d + 0.5) / (sx, sy) - 0.5 + (x0, y0)
            out.append((k, i, j, q[0], q[1]))
            if arm is None and info.get('arm'):
                arms.append(info['arm'])
                if len(arms) >= 8:                          # плечо найдено — дальше без перебора
                    arm = float(np.median(arms))
    if pr:
        pr.close()
    return np.array(out, float).reshape(-1, 5)


def _solve(rows, N, w, x0=None, dense_max=3000):
    """Взвешенный МНК по строкам (idx, val, rhs): плотно (N малое) или CGLS с нормировкой столбцов."""
    idx, val, rhs = rows
    wv = val * w[:, None]
    wr = rhs * w
    if N <= dense_max:
        AtA = np.zeros((N, N))
        np.add.at(AtA, (idx[:, :, None], idx[:, None, :]), wv[:, :, None] * wv[:, None, :])
        Atb = np.bincount(idx.ravel(), (wv * wr[:, None]).ravel(), minlength=N)
        return np.linalg.solve(AtA + 1e-12 * np.eye(N), Atb)
    cn = np.sqrt(np.bincount(idx.ravel(), (wv ** 2).ravel(), minlength=N))
    cn[cn == 0] = 1.0
    sv = wv / cn[idx]
    Ax = lambda z: (sv * z[idx]).sum(1)
    ATy = lambda y: np.bincount(idx.ravel(), (sv * y[:, None]).ravel(), minlength=N)
    z = np.zeros(N) if x0 is None else x0 * cn
    r = wr - Ax(z)
    s_ = ATy(r)
    p_ = s_.copy()
    g0 = g = s_ @ s_
    for _ in range(4000):
        q = Ax(p_)
        al = g / max(q @ q, 1e-300)
        z += al * p_
        r -= al * q
        s_ = ATy(r)
        gn = s_ @ s_
        if gn < 1e-24 * max(g0, 1e-300):
            break
        p_ = s_ + (gn / g) * p_
        g = gn
    return z / cn


@timed_fn('Stitch: crosses')
def anchor(folder, save=True, fit_dist=True, verbose=True, preview=True):
    """Сшить скан заново по крестам (см. описание модуля). Нужны: mc.stitch(folder) и сетка крестов
    (mc.view: C — 2 креста и больше). Возвращает Result; r.theta — дисторсия, r.share — доля крестов на сетке
    (0..1), r.info — всё."""
    from .mask import Grid
    from .storage import ScanData
    from .stitch import load_positions, make_preview
    from .optics import load_dist, dist_vec, dist_basis, dist_corner, DIST_KEYS, _dist_path
    g = Grid(folder)
    if not g.ready():
        raise ValueError("Mark the crosses first: mc.view(folder), key C (2 crosses or more).")
    st = ScanData(folder)
    st.fix = None                                           # кресты — в сырых кадрах
    H, W = st.shape[:2]
    n = len(st)
    px = st.meta.get('pixel_size_um') or 1.0
    c = np.array([W / 2, H / 2])
    R = float(np.hypot(W / 2, H / 2))
    D0 = load_dist(folder, st.meta)
    th0 = dist_vec(D0)
    Pabs = load_positions(folder, st)
    P0 = Pabs.min(0)
    P = Pabs - P0                                           # координаты сетки (мозаика от левого верхнего кадра)
    obs = _detect(st, g, P, D0, px, verbose)
    st.close()
    if len(obs) < 3:
        raise ValueError(f"Only {len(obs)} crosses found in the frames: nothing to hold the frames on.")
    K = obs[:, 0].astype(int)
    IJ = obs[:, 1:3]
    Rr = obs[:, 3:5]
    sp = os.path.join(folder, 'join', 'seams.npz')
    if os.path.exists(sp):
        z = np.load(sp)
        I, J, meas, ws = z['I'].astype(int), z['J'].astype(int), z['meas'].astype(float), z['w'].astype(float)
        th_s = z['theta0'] if 'theta0' in z.files else th0
        ok = ws > 0
        I, J, meas, ws = I[ok], J[ok], meas[ok], ws[ok]
    else:
        I = J = np.zeros(0, int)
        meas, ws, th_s = np.zeros((0, 2)), np.zeros(0), th0
    # G_ab: разность смещений по полосе перекрытия (5x5 точек) при единичных θ
    G = np.zeros((len(I), 2, 4))
    t5 = np.linspace(0.1, 0.9, 5)
    for e, (a, b) in enumerate(zip(I, J)):
        d = P[b] - P[a]
        xa = np.maximum(0, d[0]) + t5 * (W - abs(d[0]))
        ya = np.maximum(0, d[1]) + t5 * (H - abs(d[1]))
        ua = np.array(np.meshgrid(xa, ya)).reshape(2, -1).T
        G[e] = (dist_basis(ua - c, R) - dist_basis(ua - d - c, R)).mean(0)
    many = len(obs) >= 20 and (len(np.unique(K)) >= 3 or len(I) >= 3)
    fit_th = fit_dist and many
    N = 2 * n + 4 + 4
    iO, iA, iT = 2 * n, 2 * n + 2, 2 * n + 4

    def model_rows(U, th_fix):
        """Строки: кресты (x, y), швы (x, y), слабые привязки положений и θ."""
        B = dist_basis(U - c, R)                            # (m, 2, 4)
        i_, j_ = IJ[:, 0], IJ[:, 1]
        m = len(obs)
        idx = np.zeros((2 * m, 8), int)
        val = np.zeros((2 * m, 8))
        rhs = np.zeros(2 * m)
        idx[0::2, :4] = np.c_[2 * K, np.full(m, iO), np.full(m, iA), np.full(m, iA + 1)]
        val[0::2, :4] = np.c_[np.ones(m), -np.ones(m), -i_, j_]
        idx[1::2, :4] = np.c_[2 * K + 1, np.full(m, iO + 1), np.full(m, iA + 1), np.full(m, iA)]
        val[1::2, :4] = np.c_[np.ones(m), -np.ones(m), -i_, -j_]
        idx[:, 4:] = iT + np.arange(4)
        val[0::2, 4:] = -B[:, 0, :]
        val[1::2, 4:] = -B[:, 1, :]
        rhs[0::2], rhs[1::2] = -Rr[:, 0], -Rr[:, 1]
        e = len(I)
        sidx = np.zeros((2 * e, 8), int)
        sval = np.zeros((2 * e, 8))
        sidx[0::2, :2] = np.c_[2 * J, 2 * I]
        sidx[1::2, :2] = np.c_[2 * J + 1, 2 * I + 1]
        sval[:, :2] = [1.0, -1.0]
        sidx[:, 2:6] = iT + np.arange(4)
        sval[0::2, 2:6] = G[:, 0, :]
        sval[1::2, 2:6] = G[:, 1, :]
        srhs = meas.reshape(-1) + (G @ th_s).reshape(-1)
        pidx = np.zeros((2 * n + 4, 8), int)
        pval = np.zeros((2 * n + 4, 8))
        pidx[:2 * n, 0] = np.arange(2 * n)
        pval[:2 * n, 0] = 1.0
        prhs = np.r_[P.reshape(-1), np.zeros(4)]
        pidx[2 * n:, 0] = iT + np.arange(4)
        pval[2 * n:, 0] = R                                 # θ·R — смещение в углу, px
        prhs[2 * n:] = R * (th_s if th_fix is None else th_fix)
        return (np.r_[idx, sidx, pidx], np.r_[val, sval, pval], np.r_[rhs, srhs, prhs]), 2 * m, 2 * e

    # до: кресты по старым положениям и дисторсии, лучшая квадратная решётка
    from .crosses import fit_square
    U0 = Rr - (dist_basis(Rr - c, R) @ th0)
    Q0 = P[K] + U0
    Ob, ab = fit_square(IJ, Q0)
    e0 = np.hypot(*(Q0 - (Ob + IJ[:, :1] * ab + IJ[:, 1:] * np.array([-ab[1], ab[0]]))).T)
    wc = np.ones(len(obs))
    U = U0.copy()
    x = None
    th = th0.copy()
    for it, cs in enumerate((1e9, 4.0, 2.0, 1.0, 1.0, 0.7, 0.7)):
        rows, mc_, ms_ = model_rows(U, None if fit_th else th0)
        w = np.r_[np.repeat(wc, 2), np.repeat(ws, 2) if len(I) else np.zeros(0),
                  np.full(2 * n, 1e-3), np.full(4, 0.3 if fit_th else 1e4)]
        if x is None:                                       # начальное: старые положения, решётка — по ним
            x = np.r_[P.reshape(-1), Ob, ab, th0]
        x = _solve(rows, N, w, x0=x)
        th = x[iT:]
        U = Rr - (dist_basis(U - c, R) @ th)                # u = r − Δθ(u)
        Pn, On, an = x[:2 * n].reshape(n, 2), x[iO:iO + 2], x[iA:iA + 2]
        node = On + IJ[:, :1] * an + IJ[:, 1:] * np.array([-an[1], an[0]])
        res = np.hypot(*(Pn[K] + U - node).T)
        sd = max(1.48 * float(np.median(res[wc > 0.05])) if (wc > 0.05).any() else 1.0, 0.15)
        wc = 1.0 / (1.0 + (res / max(cs, 2.5 * sd)) ** 2)
        wc[res > max(3.0, 6 * sd)] = 0.0                    # не тот крест / грязь
    Pn, On, an = x[:2 * n].reshape(n, 2), x[iO:iO + 2], x[iA:iA + 2]
    U = Rr - (dist_basis(U - c, R) @ th)
    node = On + IJ[:, :1] * an + IJ[:, 1:] * np.array([-an[1], an[0]])
    e1 = np.hypot(*(Pn[K] + U - node).T)
    used = wc > 0
    seam_res = np.hypot(*(Pn[J] - Pn[I] + G @ th - meas - G @ th_s).T) if len(I) else np.zeros(0)
    D = {k_: float(v) for k_, v in zip(DIST_KEYS, th)}
    corner = dist_corner(D, W, H)
    share = float((e1[used] * px <= TOL_UM).mean()) if used.any() else 0.0
    share0 = float((e0 * px <= TOL_UM).mean())
    pitch_px = float(np.hypot(*an))
    info = dict(crosses=int(len(obs)), used=int(used.sum()), frames_with_crosses=int(len(np.unique(K))),
                frames=n, within_um=TOL_UM, share=round(share, 4), share_before=round(share0, 4),
                p95_um=round(float(np.quantile(e1[used], 0.95)) * px, 3) if used.any() else None,
                p95_um_before=round(float(np.quantile(e0, 0.95)) * px, 3),
                rms_px=round(float(np.sqrt((e1[used] ** 2).mean())), 3) if used.any() else None,
                seam_px=round(float(np.median(seam_res)), 3) if len(seam_res) else None,
                pitch_px=round(pitch_px, 3), distortion=D, corner_px=round(corner, 2), fitted=bool(fit_th))
    if save:
        out = os.path.join(folder, 'join')
        Pabs_n = Pn + P0
        with open(os.path.join(out, 'positions_all.csv'), 'w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(['id', 'x_px', 'y_px', 'scale'])
            for k_ in range(n):
                wr.writerow([k_, f"{Pabs_n[k_, 0]:.3f}", f"{Pabs_n[k_, 1]:.3f}", "1.0000"])
        if fit_th:
            dd = dict(D, corner_px=corner, crosses=int(used.sum()), apply=bool(corner >= 0.2), frame_px=[W, H],
                      source="crosses + seams (anchor)")
            dp = _dist_path(st.meta)
            for p_ in (os.path.join(out, 'distortion.json'),) + ((dp,) if dp and dd['apply'] else ()):
                os.makedirs(os.path.dirname(p_), exist_ok=True)
                with open(p_, 'w', encoding='utf-8') as f:
                    json.dump(dd, f, indent=1)
        sj = os.path.join(out, 'stitch.json')
        try:
            si = json.load(open(sj, encoding='utf-8'))
        except (OSError, ValueError):
            si = {}
        si['crosses'] = info
        with open(sj, 'w', encoding='utf-8') as f:
            json.dump(si, f, indent=1)
        sh = Pn.min(0)                                      # сетка: в координатах новой мозаики
        g.cr.O, g.cr.v1, g.cr.v2 = On - sh, an.copy(), np.array([-an[1], an[0]])
        g.cr.marks = [g.cr.point(*ij) for ij in g.cr.ij]
        g.frames_px = Pn - sh
        g.refined = dict(crosses=int(used.sum()), rms_px=info['rms_px'], anchored=True, share=info['share'])
        g.save(px)
        if preview:
            make_preview(folder, Pabs_n, verbose=False)
    L.event('STI-005', f"{folder}: crosses {info}")
    r = Result("Stitched on the crosses",
               **{"Crosses": f"{info['used']} used of {info['crosses']} found, in {info['frames_with_crosses']} of {n} frames",
                  f"On the grid within {TOL_UM} um": f"{100 * share0:.1f} % -> {100 * share:.1f} %",
                  "95 % of crosses within, um": f"{info['p95_um_before']} -> {info['p95_um']}",
                  "Seam error, px": info['seam_px'] if info['seam_px'] is not None else '-',
                  "Distortion at the edge, px": f"{dist_corner(D0, W, H):.2f} -> {corner:.2f}"
                  + ("" if fit_th else " (kept: too few crosses)"),
                  "Grid pitch, um": round(pitch_px * px, 3)})
    r.theta, r.share, r.info = th, share, info
    return r


def fit_crosses(folder, verbose=True):
    """Второй шаг сшивки: кадры — на кресты (после mc.stitch и разметки крестов в mc.view, C).
    Решётка, положения кадров и дисторсия — вместе (anchor); если дисторсия заметно другая, чем была,
    швы меряются заново по выпрямленным кадрам и кресты — ещё раз. Затем превью."""
    from .storage import ScanData
    from .stitch import stitch, make_preview, load_positions
    from .optics import load_dist, dist_corner
    if not os.path.exists(os.path.join(folder, 'join', 'positions_all.csv')):
        raise ValueError("Stitch first (S): the frames need their seams before the crosses.")
    st = ScanData(folder)
    H, W = st.shape[:2]
    c0 = dist_corner(load_dist(folder, st.meta), W, H)
    st.close()
    r = anchor(folder, verbose=verbose, preview=False)
    if r.info['fitted'] and abs(r.info['corner_px'] - c0) > 0.5:
        stitch(folder, preview=False, verbose=verbose)     # швы — по выпрямленным кадрам
        r = anchor(folder, verbose=verbose, preview=False)
    r['Preview'] = make_preview(folder, load_positions(folder), verbose=verbose)
    return r
