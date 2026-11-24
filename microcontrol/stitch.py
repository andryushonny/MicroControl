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
from .util import Result, Progress, save_image
from .stats import timed_fn


def _gray(a):
    a = (a if a.ndim == 2 else a[..., 1]).astype(np.float32)   # зелёный канал: самый резкий, без цветной каймы
    return a.astype(np.float32)


def _whiten(a, sigma):
    """Как в ASHLAR: подавить плавный фон (лапласиан гауссиана) — корреляция по деталям, не по яркости."""
    return -cv2.Laplacian(cv2.GaussianBlur(a, (0, 0), sigma), cv2.CV_32F) if sigma > 0 else a - a.mean()


def _pcm_peaks(A_, B_, k=5, c=(0.0, 0.0), rad=None, pw=0.25):
    """Корреляция с несколькими пиками (идея Preibisch et al., 2009: у шума и повторяющегося узора самый
    высокий пик не всегда верный — проверяются k лучших). Спектр делится на |C|^pw, а не на |C| (чистая
    фазовая корреляция): после обеления высокие частоты — шум и рисунок сенсора, привязанный к кадру, и при
    pw = 1 их пик встаёт на предсказание столика. c, rad — где искать (сдвиг по столику и его ошибка).
    -> [(sx, sy)] с долей пикселя, B(x) = A(x - s)."""
    h, w = A_.shape
    FA, FB = cv2.dft(A_, flags=cv2.DFT_COMPLEX_OUTPUT), cv2.dft(B_, flags=cv2.DFT_COMPLEX_OUTPUT)
    C = cv2.mulSpectrums(FB, FA, 0, conjB=True)
    mag = cv2.magnitude(C[..., 0], C[..., 1])
    mag = cv2.pow(mag, pw) + 1e-9 * float(mag.max() ** pw + 1e-30) if pw else 1.0
    C[..., 0] /= mag
    C[..., 1] /= mag
    r = cv2.idft(C, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    r = np.fft.fftshift(r)
    oy_, ox_ = 0, 0
    if rad is not None:                                         # только окно вокруг ожидаемого сдвига
        cx_, cy_ = int(round(c[0] + w // 2)), int(round(c[1] + h // 2))
        R_ = int(np.ceil(rad)) + 3
        oy_, ox_ = max(0, cy_ - R_), max(0, cx_ - R_)
        r = r[oy_:min(h, cy_ + R_ + 1), ox_:min(w, cx_ + R_ + 1)]
        if r.size == 0:
            return []
    r_ = cv2.GaussianBlur(r, (3, 3), 0)
    dil = cv2.dilate(r_, np.ones((5, 5), np.uint8))
    cand = np.argwhere((r_ >= dil) & (r_ > 0))
    if not len(cand):
        return []
    v_ = r_[cand[:, 0], cand[:, 1]]
    o_ = np.argsort(-v_)[:k]
    cand = cand[o_[v_[o_] >= 0.3 * v_[o_[0]]]]                    # слабые пики — не кандидаты
    out = []
    hh, ww_ = r.shape
    for y, x in cand:                                           # доля пикселя — центр масс 3 x 3
        y0, y1, x0, x1 = max(0, y - 1), min(hh, y + 2), max(0, x - 1), min(ww_, x + 2)
        wv = np.clip(r[y0:y1, x0:x1], 0, None)
        if wv.sum() <= 0:
            fy, fx = float(y), float(x)
        else:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            fy, fx = float((wv * yy).sum() / wv.sum()), float((wv * xx).sum() / wv.sum())
        out.append((fx + ox_ - w // 2, fy + oy_ - h // 2))
    return out


def _ncc_err(a, b, ox, oy):
    """1 - NCC общей части: точка a-окна (i) <-> b-окна (i + o)."""
    M = np.float32([[1, 0, ox], [0, 1, oy]])                    # bb(i) = b(i + o)
    bb = cv2.warpAffine(b, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderValue=np.nan)
    ok = ~np.isnan(bb)
    if ok.sum() < 256:
        return 2.0
    va, vb = a[ok] - a[ok].mean(), bb[ok] - bb[ok].mean()
    return 1.0 - float((va * vb).sum() / (np.sqrt((va ** 2).sum() * (vb ** 2).sum()) + 1e-9))


def _lk(a, b, ox, oy, it=4):
    """Доли пикселя без смещения к целому: Лукас — Канаде по общей части. a(i) <-> b(i + o);
    bb(i) = b(i + o), шаг δ = (GᵀG)⁻¹ Gᵀ (a - bb), G = ∇bb (яркость и контраст выровнены). -> (ox, oy) или None."""
    o = np.array([ox, oy], float)
    for _ in range(it):
        M = np.float32([[1, 0, o[0]], [0, 1, o[1]]])
        bb = cv2.warpAffine(b, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                            borderValue=np.nan)
        ok = ~np.isnan(bb)
        ok[:2], ok[-2:], ok[:, :2], ok[:, -2:] = False, False, False, False
        ok &= np.roll(ok, 1, 0) & np.roll(ok, -1, 0) & np.roll(ok, 1, 1) & np.roll(ok, -1, 1)
        if ok.sum() < 256:
            return None
        bb0 = np.where(np.isnan(bb), 0, bb).astype(np.float32)
        gx = cv2.Sobel(bb0, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(bb0, cv2.CV_32F, 0, 1, ksize=3) / 8
        av, bv, gxv, gyv = a[ok], bb0[ok], gx[ok], gy[ok]
        sa, sb = av.std() + 1e-9, bv.std() + 1e-9
        r = (av - av.mean()) / sa - (bv - bv.mean()) / sb
        gxv, gyv = gxv / sb, gyv / sb
        G = np.array([[gxv @ gxv, gxv @ gyv], [gxv @ gyv, gyv @ gyv]], float)
        if np.linalg.cond(G) > 1e4:                             # одно направление без деталей
            return None
        d = np.linalg.solve(G, [gxv @ r, gyv @ r])              # bb(i + d) ≈ a(i)
        if np.hypot(*d) > 2.0:
            return None
        o += d
        if np.hypot(*d) < 0.01:
            break
    return o


def _solve_graph(I, J, meas, w, base, lam=0.01, x0=None, iters=400, rel=0.0):
    """Положения кадров по швам: min Σ w²·|x_J - x_I - meas|² + rel²·Σ|x_J - x_I - (base_J - base_I)|²
    + lam²·|x - base|² — сопряжённые градиенты по разреженному графу (без плотных матриц: тысячи кадров
    за доли секунды). rel — столик как «шов» каждой пары: держит шаг между соседями, а не место кадра
    (ошибка столика плавная: соседи сдвинуты вместе)."""
    if rel > 0:
        I, J = np.r_[I, I], np.r_[J, J]
        meas = np.r_[meas, base[J[len(J) // 2:]] - base[I[len(I) // 2:]]]
        w = np.r_[w, np.full(len(I) // 2, rel)]
    n = len(base)
    ww, l2 = w ** 2, lam ** 2

    def Lop(x):
        d = (x[J] - x[I]) * ww[:, None]
        y = l2 * x
        np.add.at(y, J, d)
        np.add.at(y, I, -d)
        return y
    rhs = l2 * base
    m_ = meas * ww[:, None]
    np.add.at(rhs, J, m_)
    np.add.at(rhs, I, -m_)
    diag = np.full(n, l2)
    np.add.at(diag, J, ww)
    np.add.at(diag, I, ww)
    x = (base if x0 is None else x0).astype(float).copy()
    r = rhs - Lop(x)
    z = r / diag[:, None]
    pdir = z.copy()
    rz = float((r * z).sum())
    nb = float(np.sqrt((rhs ** 2).sum())) + 1e-12
    for _ in range(iters):
        if np.sqrt((r ** 2).sum()) < 1e-7 * nb:
            break
        Ap = Lop(pdir)
        al = rz / max(float((pdir * Ap).sum()), 1e-30)
        x += al * pdir
        r -= al * Ap
        z = r / diag[:, None]
        rz2 = float((r * z).sum())
        pdir = z + (rz2 / max(rz, 1e-30)) * pdir
        rz = rz2
    return x


def _pair(st, ia, ib, d, W, H, pad, sigma, max_shift, other=None, peaks=5, fine=False):
    """Сдвиг кадра ib относительно ia (px, левый верхний угол b минус a) по полосам перекрытия.
    d — предсказание по столику; полосы берутся с запасом pad в каждом кадре отдельно (у краёв —
    сколько есть), дополняются до одного размера. other — подставной кадр вместо ib (тест перестановок).
    fine=True — предсказание уже точное (второй раунд): только уточнение Лукаса — Канаде.
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
    if fine:
        o = _lk(a, b, ax0 - bx0 - d[0], ay0 - by0 - d[1], it=6)
        if o is None:
            return None, 2.0
        dt = np.array([ax0 - bx0 - o[0], ay0 - by0 - o[1]])
        if np.hypot(*(dt - d)) > max_shift:
            return None, 2.0
        return dt, _ncc_err(a, b, o[0], o[1])
    # грубо: полосы уменьшены (до ~40 тыс. пикселей), окно Ханна, дополнение нулями вдвое — без «заворота»
    f = max(1, int(np.ceil(np.sqrt(max(a.size, b.size) / 3e4))))
    ac, bc = (a, b) if f == 1 else (cv2.resize(a, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA),
                                    cv2.resize(b, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA))
    h, w = (cv2.getOptimalDFTSize(2 * max(ac.shape[0], bc.shape[0])),  # размер, удобный для БПФ (2^a 3^b 5^c)
            cv2.getOptimalDFTSize(2 * max(ac.shape[1], bc.shape[1])))
    A_, B_ = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    A_[:ac.shape[0], :ac.shape[1]] = (ac - ac.mean()) * cv2.createHanningWindow(ac.shape[::-1], cv2.CV_32F)
    B_[:bc.shape[0], :bc.shape[1]] = (bc - bc.mean()) * cv2.createHanningWindow(bc.shape[::-1], cv2.CV_32F)
    best = (None, 2.0)
    for sx, sy in _pcm_peaks(A_, B_, peaks, ((ax0 - bx0 - dx) / f, (ay0 - by0 - dy) / f), max_shift / f + 2):                    # B_(x) = A_(x - s): несколько пиков —
        dt = np.array([ax0 - bx0 - f * sx, ay0 - by0 - f * sy])  # каждый проверяется совпадением картинки
        if np.hypot(*(dt - (dx, dy))) > max_shift + f:
            continue
        e_ = _ncc_err(a, b, ax0 - bx0 - dt[0], ay0 - by0 - dt[1])
        if e_ < best[1]:
            best = (dt, e_)
    if best[0] is None:
        return best
    dt = best[0]
    o = _lk(a, b, ax0 - bx0 - dt[0], ay0 - by0 - dt[1])
    if o is not None:
        dt2 = np.array([ax0 - bx0 - o[0], ay0 - by0 - o[1]])
        if np.hypot(*(dt2 - dt)) < f + 1.0:
            e2 = _ncc_err(a, b, o[0], o[1])
            if e2 <= best[1] + 0.01:
                best = (dt2, e2)
    return best


def seams(folder):
    """Швы последней сшивки: dict(I, J, meas, w, w0, base) или None (сшивка другим движком / старая)."""
    sp = os.path.join(folder, 'join', 'seams.npz')
    if not os.path.exists(sp):
        return None
    z = np.load(sp)
    if 'base' not in z.files:
        return None
    return {k: z[k] for k in z.files}


def seams_edit(folder, ovr=None):
    """Швы, включённые / выключенные руками (J: щелчок по шву): {(I, J): True | False}. ovr=None — прочитать,
    иначе — записать."""
    fp = os.path.join(folder, 'join', 'seams_edit.json')
    if ovr is None:
        try:
            return {(int(a), int(b)): bool(v) for a, b, v in json.load(open(fp, encoding='utf-8'))}
        except Exception:
            return {}
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(sorted([int(a), int(b), bool(v)] for (a, b), v in ovr.items()), f)
    return dict(ovr)


def resolve(z, ovr=None):
    """Положения кадров заново по швам z (seams()) с правкой ovr (seams_edit) — за доли секунды (сопряжённые
    градиенты). Шов, выключенный программой и включённый руками, — с исходным весом. -> (pos, res, w)."""
    I, J, meas = z['I'].astype(int), z['J'].astype(int), z['meas'].astype(float)
    w = z['wq' if 'wq' in z else 'w'].astype(float).copy()
    for k in range(len(I)):
        v = (ovr or {}).get((int(I[k]), int(J[k])))
        if v is False:
            w[k] = 0.0
        elif v is True and w[k] <= 0:
            w[k] = max(float(z['w0'][k]), 0.2)
    pos = _solve_graph(I, J, meas, w, z['base'].astype(float), 0.01, rel=float(z['lam']) if 'lam' in z else 0.0)
    return pos, np.hypot(*(pos[J] - pos[I] - meas).T), w


def _pair_full(st, ia, ib, d, W, H, sigma, min_ov=0.15, peaks=8):
    """Как в MIST: сдвиг ib относительно ia по кадрам целиком, без опоры на столик (столик ошибся сильнее
    перекрытия: люфт на большом объективе, сбой). d — предсказание (только чтобы из равных выбрать ближний).
    -> (сдвиг, 1 - NCC) или (None, 2.0)."""
    a = _whiten(_gray(st.frame(ia)), sigma)
    b = _whiten(_gray(st.frame(ib)), sigma)
    f = max(1, int(np.ceil(np.sqrt(a.size / 3e4))))
    ac, bc = (a, b) if f == 1 else (cv2.resize(a, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA),
                                    cv2.resize(b, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA))
    h, w = cv2.getOptimalDFTSize(2 * ac.shape[0]), cv2.getOptimalDFTSize(2 * ac.shape[1])
    A_, B_ = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    A_[:ac.shape[0], :ac.shape[1]] = ac - ac.mean()
    B_[:bc.shape[0], :bc.shape[1]] = bc - bc.mean()
    best = (None, 2.0, 0.0)
    for sx, sy in _pcm_peaks(A_, B_, peaks):                    # b(u) = a(u + dt): dt = -s
        dt = -f * np.array([sx, sy])
        ov = (W - abs(dt[0])) * (H - abs(dt[1])) / float(W * H)
        if abs(dt[0]) >= W or abs(dt[1]) >= H or ov < min_ov:
            continue
        o = _lk(a, b, -dt[0], -dt[1], it=6)                     # a(i) <-> b(i + o), o = -dt
        if o is None:
            continue
        dt = -o
        e_ = _ncc_err(a, b, o[0], o[1])
        if e_ < best[1] - 0.02 or (e_ < best[1] + 0.02 and best[0] is not None
                                   and np.hypot(*(dt - d)) < np.hypot(*(best[0] - d))):
            best = (dt, e_, ov)
    return best[0], best[1]


class _Reduced:
    """Кадры скана, уменьшенные в r раз (quick): JPEG раскрывается сразу уменьшенным (в разы быстрее)."""
    FLAGS = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}

    def __init__(self, st, r, verbose=False):
        from concurrent.futures import ThreadPoolExecutor
        self.index, self.meta, self.r = st.index, st.meta, r
        H, W = st.shape[:2]
        self.W, self.H = W // r, H // r
        self.shape = (self.H, self.W) + tuple(st.shape[2:])
        pr = Progress(len(st), "Quick stitch: reading", unit='frames') if verbose else None

        def rd(i):
            row = st.index[i]
            if row.get('file') and r in self.FLAGS and row['file'].lower().endswith(('.jpg', '.jpeg')):
                a = cv2.imdecode(np.fromfile(os.path.join(st.folder, 'frames', row['file']), np.uint8), self.FLAGS[r])
            else:
                with st.lock:
                    a = st.frame(i) if st.fix is None else st._jpg(row['file']) if row.get('file') else st.frame(i)
                a = cv2.resize(a, (self.W, self.H), interpolation=cv2.INTER_AREA) if r > 1 else a
            if a.shape[1] != self.W or a.shape[0] != self.H:
                a = cv2.resize(a, (self.W, self.H), interpolation=cv2.INTER_AREA)
            return a
        self.img = [None] * len(st)
        with ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2)))) as ex:
            for i, a in enumerate(ex.map(rd, range(len(st)))):
                self.img[i] = a
                if pr:
                    pr.update(1, f"frame {i + 1} of {len(st)}")
        if pr:
            pr.close()

    def frame(self, i, crop=None):
        a = self.img[i]
        if crop is None:
            return a
        x0, y0, x1, y1 = crop
        return a[y0:y1, x0:x1]


def _quick(st, pred, p, W, H, max_shift_um, px, verbose, A):
    """Quick: тот же движок, но на кадрах, уменьшенных до ~320 px по ширине, без перебора ложных пар сверх
    нужного; положения — обратно в полный размер."""
    r = max([k for k in (1, 2, 4, 8) if W // k >= 300] or [1])
    red = _Reduced(st, r, verbose=verbose)
    ms = min(max_shift_um / (px * r), 0.5 * min(red.W, red.H))
    pos, info = _builtin(red, pred / r, p, red.W, red.H, ms, 0.0, 0.01, verbose, A=A / r)
    sm = info.get('_seams')
    if sm is not None:                                       # швы — в пикселях полного кадра (кресты, F)
        sm['meas'] = sm['meas'] * r
        sm['base'] = sm['base'] * r
    info.update(reduced=r)
    return pos * r, info


@_per_sample
@timed_fn('Stitch (all of it)')
def stitch(folder, engine='builtin', max_shift_um=40.0, sigma=None, alpha=0.01, preview=True, verbose=True,
           flat=True, mode=None):
    """Сшить скан: найти положения кадров. mode — 'quick' (быстро, на уменьшенных кадрах: обзор; править и
    отправлять в экспорт нельзя) или 'quality' (точно: только сдвиги кадров, все поправки; для рамки — ровный
    прямоугольник join/quality.png).
    max_shift_um — насколько столик может ошибиться между соседними кадрами. Без mode — quality (после съёмки сама
    идёт quick)."""
    t0 = time.monotonic()
    st = ScanData(folder)
    mode = mode or 'quality'
    if mode == 'quick':
        flat = False
    H, W = st.shape[:2]
    from .optics import flatfield, attach
    os.makedirs(os.path.join(folder, 'join'), exist_ok=True)
    fp = os.path.join(folder, 'join', 'flatfield.npy')
    gp = os.path.join(folder, 'join', 'gains.npy')
    if os.path.exists(gp):                                       # яркость кадров считается заново
        os.remove(gp)
    st.gains = None
    dkp = os.path.join(folder, 'join', 'dark.npy')
    for p_ in (fp, dkp):
        if os.path.exists(p_):
            os.remove(p_)
    if flat:                                                     # поле освещения: снятое кнопкой Flat field, иначе —
        from .optics import measured_flat                        # по сырым кадрам скана
        F, Dk = measured_flat(st.meta, W, H)
        if F is None:
            F = flatfield(st, verbose=verbose)
        elif Dk is not None:
            np.save(dkp, Dk)
        if F is not None:
            np.save(fp, F)
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
    max_shift = min(max_shift_um / px, 0.5 * min(W, H))       # больше — уже не шов, а совпадение шума
    if sigma is None:
        sigma = 1.3 / px if px < 0.6 else 0.0               # детали ~1 мкм: шум пикселей не мешает (полный размер кадра)
    used = None
    if engine == 'ashlar':
        try:
            pos, info = _ashlar(st, pred, px, max_shift_um, sigma, verbose)
            used = 'ashlar'
        except Exception as e:
            if engine == 'ashlar':
                raise
            L.event('STI-002', f"ASHLAR not used ({type(e).__name__}: {e}); built-in engine.", 20)
    if used is None and mode == 'quick':
        pos, info = _quick(st, pred, p, W, H, max_shift_um, px, verbose, A)
        used = 'quick'
    if used is None:
        pos, info = _builtin(st, pred, p, W, H, max_shift, sigma, alpha, verbose, A=A)
        used = 'builtin'
    out = os.path.join(folder, 'join')
    os.makedirs(out, exist_ok=True)
    P_before = load_positions(folder, st)                   # как кадры стояли до этой сшивки (для крестов)
    with open(os.path.join(out, 'positions_all.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['id', 'x_px', 'y_px', 'scale'])
        for i in range(n):
            wr.writerow([i, f"{pos[i, 0]:.2f}", f"{pos[i, 1]:.2f}", "1.0000"])
    sm = info.pop('_seams', None)
    if os.path.exists(os.path.join(out, 'seams_edit.json')):   # правки швов прежней сшивки — к ней
        os.remove(os.path.join(out, 'seams_edit.json'))
    sp = os.path.join(out, 'seams.npz')                     # швы — для сшивки по крестам (anchor.py)
    if sm is not None:
        from .optics import load_dist, dist_vec
        np.savez(sp, theta0=dist_vec(load_dist(folder, meta)), **sm)
    elif os.path.exists(sp):
        os.remove(sp)
    info.update(engine=used, mode=mode, frames=n, frame_px=[W, H], pixel_size_um=px, time_s=round(time.monotonic() - t0, 1))
    with open(os.path.join(out, 'stitch.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1)
    r = Result("Stitched (quick)" if mode == 'quick' else "Stitched", Engine=used, Frames=n,
               **{"Seams accepted": f"{info.get('accepted', '-')} of {info.get('pairs', '-')}",
                  "Seam error, px": info.get('residual_px', '-'),
                  "Camera vs stage": info.get('camera', '-'),
                  "Backlash X, Y, um": info.get('backlash_um', '-'),
                  "Corrections": f"light {'yes' if getattr(st, 'fix_info', {}).get('flatfield') else 'no'}, "
                                 f"distortion {getattr(st, 'fix_info', {}).get('corner_px', 0.0):.1f} px at the edge",
                  "Time, s": info['time_s']})
    if flat:                                                # яркость кадров по перекрытиям (после сшивки)
        try:
            from .optics import frame_gains
            gn = frame_gains(st, pos, W, H, verbose=verbose)
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
        r['Preview'] = make_preview(folder, pos, verbose=verbose, fast=mode == 'quick')
        r['Time, s'] = round(time.monotonic() - t0, 1)
    st.close()
    if mode == 'quality' and meta.get('quality_rect_um'):     # рамка: ровный прямоугольник
        from . import quality as Q
        q = Q.render(folder, verbose=verbose)
        if q:
            r['Quality'] = (f"{q['size_px'][0]} x {q['size_px'][1]} px, {q['pixel_um']:.5f} um/px"
                            + ("" if q['pixel_source'] == 'crosses' else " (pixel not calibrated)")
                            + ("" if q['ok'] else "; " + "; ".join(q['problems'])))
    for f_ in ('quality.png', 'quality.json'):               # другой вид — прежний снимок рамки не годится
        if mode != 'quality' and os.path.exists(os.path.join(folder, 'join', f_)):
            os.remove(os.path.join(folder, 'join', f_))
    L.event('STI-001', f"{folder}: {dict(r)}")
    return r


@timed_fn('Stitch: seams')
def _builtin(st, pred, p, W, H, max_shift, sigma, alpha, verbose, A=None):
    n = len(pred)
    pad = int(max_shift)
    # 1) соседи
    pairs, area = [], []
    cell = {}                                                   # сетка W x H: соседи — только из 3 x 3 клеток
    key = np.floor(pred / (W, H)).astype(int)
    for i, (kx, ky) in enumerate(key):
        cell.setdefault((kx, ky), []).append(i)
    for a in range(n):
        kx, ky = key[a]
        near = [b for ddx in (-1, 0, 1) for ddy in (-1, 0, 1) for b in cell.get((kx + ddx, ky + ddy), ()) if b > a]
        for b in sorted(near):
            d = pred[b] - pred[a]
            if abs(d[0]) < W - 16 and abs(d[1]) < H - 16:
                pairs.append((a, int(b)))
                area.append((W - abs(d[0])) * (H - abs(d[1])))
    if pairs:                                                   # уголки по диагонали — малы и ненадёжны
        area = np.array(area)
        pairs = [q for q, ar in zip(pairs, area) if ar >= 0.3 * area.max()]
    if not pairs:
        return pred.copy(), dict(pairs=0, accepted=0, note="frames do not overlap: positions from the stage")
    # 2) сдвиги
    meas, err = [], []
    nf = min(300, 3 * len(pairs))
    pr = Progress(len(pairs) + nf + 1, "Stitch: seams", unit='comparisons') if verbose else None
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock_ = threading.Lock()
    fr0_ = st.frame

    def frame_locked(*a_, **k_):                                # чтение кадров — по одному (кэш, HDF5), счёт — параллельно
        with lock_:
            return fr0_(*a_, **k_)
    st.frame = frame_locked
    try:
        nw_ = max(1, min(8, (os.cpu_count() or 2) - 1))
        ex_ = ThreadPoolExecutor(max_workers=nw_)
        try:
            futs_ = [ex_.submit(_pair, st, a, b, pred[b] - pred[a], W, H, pad, sigma, max_shift) for a, b in pairs]
            for k_, ((a, b), fu_) in enumerate(zip(pairs, futs_)):
                dt, e = fu_.result()
                meas.append(pred[b] - pred[a] if dt is None else dt); err.append(e)
                if pr:
                    pr.update(1, f"seam {k_ + 1} of {len(pairs)}: frames {a + 1} + {b + 1}")
        finally:
            ex_.shutdown(wait=True, cancel_futures=True)       # Stop: оставшиеся швы не считаются
    finally:
        st.frame = fr0_
    meas, err = np.array(meas), np.array(err)
    # 3) тест перестановок: та же геометрия, но вместо соседа — далёкий кадр (заведомо ложная пара)
    rng = np.random.default_rng(0)
    fake = []
    for f_ in range(nf):
        if pr:
            pr.update(1, f"false-match test {f_ + 1} of {nf}")
        a, b = pairs[rng.integers(len(pairs))]
        c = int(rng.integers(n))
        dc = np.abs(pred[c] - pred[a])
        if c in (a, b) or (dc[0] < 1.5 * W and dc[1] < 1.5 * H):
            continue
        fake.append(_pair(st, a, b, pred[b] - pred[a], W, H, pad, sigma, 1e9, other=c)[1])
    if pr:
        pr.update(1, "solving frame positions")
    thr = float(np.quantile(fake, alpha)) if len(fake) >= 20 else 0.7
    I = np.array([q[0] for q in pairs]); J = np.array([q[1] for q in pairs])
    # 3б) не нашёлся по полосам у предсказания — по кадрам целиком (столик ошибся больше перекрытия)
    failed = np.flatnonzero(err >= min(thr, 0.99))
    full = 0
    if len(failed):
        if pr:
            pr.extend(len(failed))
        for i_, k in enumerate(sorted(failed, key=lambda k_: pairs[k_])):
            a, b = pairs[k]
            if pr:
                pr.update(1, f"not found near the stage position: whole frames, {i_ + 1} of {len(failed)}")
            dt, e = _pair_full(st, a, b, pred[b] - pred[a], W, H, sigma)
            if dt is not None and e < err[k] - 0.05:
                meas[k], err[k] = dt, e
                full += 1

    def solve_all(meas, err):
        ok = err < min(thr, 0.99)
        # 4) модель столик -> пиксели по принятым парам: d = A·Δp
        dp = p[J] - p[I]
        Ast = -np.asarray(A, float) if A is not None else \
            np.linalg.lstsq(p - p.mean(0), pred - pred.mean(0), rcond=None)[0].T   # -A по столику (один ряд — тоже)
        model = Ast
        # люфт: с какой стороны столик подъехал к кадру (знак последнего движения по каждой оси)
        sg = np.zeros((n, 2))
        last = np.zeros(2)
        for k in range(1, n):
            dk = np.sign(p[k] - p[k - 1])
            last = np.where(dk != 0, dk, last)
            sg[k] = last
        sg[0] = sg[1] if n > 1 else 0
        try:                                                    # записанное сканом направление подъезда — точнее
            d_ = np.array([[float(r_['dir_x']), float(r_['dir_y'])] for r_ in st.index])
            if d_.shape == sg.shape and np.isfinite(d_).all() and np.abs(d_).max() > 0:
                sg = np.sign(d_)
        except (KeyError, TypeError, ValueError):
            pass
        dsg = sg[J] - sg[I]
        Bl = np.zeros((2, 2))                                       # px на смену направления по X и по Y
        okm = err < max(min(thr, 0.99), 0.3)                         # модель — по всем правдоподобным швам: выбросы
        if okm.sum() >= 3 and np.linalg.matrix_rank(dp[okm]) == 2:  # уберёт МНК ниже (d = M·Δp + Bl·Δs; один ряд — нет)
            ok_ = ok
            ok = okm
            X_, axes = dp[ok], []                                 # люфт — по каждой оси, если он отличим от шага
            for j_ in (0, 1):
                c_ = dsg[ok][:, j_:j_ + 1]
                if np.abs(c_).sum() > 0 and len(X_) >= X_.shape[1] + 3 and \
                        np.linalg.matrix_rank(np.hstack([X_, c_])) == X_.shape[1] + 1:
                    X_ = np.hstack([X_, c_])
                    axes.append(j_)
            inl = np.ones(len(X_), bool)
            for _ in range(4):                                      # ложные швы (совпадение шума) — прочь
                sol = np.linalg.lstsq(X_[inl], meas[ok][inl], rcond=None)[0]
                rs_ = np.hypot(*(X_ @ sol - meas[ok]).T)
                new = rs_ < max(3.0, 3.0 * 1.48 * float(np.median(rs_[inl])))
                if new.sum() < 3 or (new == inl).all():
                    break
                inl = new
            model = sol[:2].T
            for i_, j_ in enumerate(axes):                          # шов: β·Δs (Δs = ±2), кадр: β·s (s = ±1)
                Bl[:, j_] = sol[2 + i_]
            ok = ok_
        base = p @ model.T + sg @ Bl.T
        base += (pred - base).mean(0)
        # 5) взвешенный МНК с подавлением выбросов (веса Коши по невязке): шов, не согласный с остальными,
        #    теряет вес; кадр без надёжных швов держится моделью столика
        # веса — по точности: σ шва растёт с 1 - NCC (0.01 -> 0.2 px, 0.45 -> 4.6 px); столик держит кадр с σ_st —
        # его разброс относительно модели по самим швам (как «повторяемость» в MIST)
        w0 = np.where(err < 1.0, 1.0 / (0.1 + 10.0 * err), 0.0)
        w0[~ok] = np.minimum(w0[~ok], 0.2)
        rm_ = np.hypot(*(meas - (base[J] - base[I])).T)[ok & (err < 0.1)]
        sig_st = float(np.clip(1.48 * np.median(rm_), 0.3, max(max_shift, 0.3))) if len(rm_) >= 5 \
            else max(max_shift, 0.3)
        lam = 1.0 / sig_st                                          # σ шага столика между соседями
        w = w0.copy()
        pos = base.copy()
        for cs in (1e9, 30.0, 15.0, 8.0, 5.0, 3.0, 2.0, 2.0):         # масштаб Коши уменьшается постепенно
            pos = _solve_graph(I, J, meas, w, base, 0.01, x0=pos, rel=lam)
            res = np.hypot(*(pos[J] - pos[I] - meas).T)
            w = w0 / (1.0 + (res / cs) ** 2)
        # как в BigStitcher: худший из ещё сильных швов — прочь, если он сильно хуже остальных; и снова
        dropped = 0
        for _ in range(max(1, len(pairs) // 5)):
            strong_ = np.flatnonzero(w > 0.25 * w0.clip(1e-9))
            if len(strong_) < 3:
                break
            rs_ = res[strong_]
            k_ = int(np.argmax(rs_))
            if rs_[k_] <= max(2.5, 3.0 * float(np.mean(rs_))):
                break
            w0[strong_[k_]] = 0.0
            w = w0 / (1.0 + (res / 2.0) ** 2)
            pos = _solve_graph(I, J, meas, w, base, 0.01, x0=pos, rel=lam)
            res = np.hypot(*(pos[J] - pos[I] - meas).T)
            dropped += 1
        return pos, res, w, w0, dropped, model, Bl, base, ok, Ast, lam

    pos, res, w, w0, dropped, model, Bl, base, ok, Ast, lam = solve_all(meas, err)
    # 6) модель столика (с люфтом) точна до пикселей: шов, далёкий от неё, — чаще всего соседний период узора
    #    (кресты решётки на размытом кадре) или совпадение шума. Такой шов меряется заново в узком окне вокруг
    #    модели; не нашёлся и там — прочь
    rm = np.hypot(*(meas - (base[J] - base[I])).T)
    gm = (w0 > 0) & (err < 1.0)
    sig_m = 1.48 * float(np.median(rm[gm & (err < 0.1)])) if (gm & (err < 0.1)).sum() >= 5 else max_shift
    thr_m = float(min(max_shift, max(4.0, 5.0 * sig_m)))
    far = np.flatnonzero(rm > thr_m)
    refound = 0
    banned = np.zeros(len(pairs), bool)                         # не сошлись с моделью: второй раунд их не трогает
    tent = []                                                   # найдены по кадрам целиком: проверка новой моделью
    if len(far) and thr_m < max_shift:
        if pr:
            pr.extend(len(far))
        for i_, k in enumerate(sorted(far, key=lambda k_: pairs[k_])):
            a, b = pairs[k]
            if pr:
                pr.update(1, f"far from the stage model: measured again near it, {i_ + 1} of {len(far)}")
            db = base[b] - base[a]
            dt, e = _pair(st, a, b, db, W, H, int(thr_m) + 4, sigma, thr_m)
            if dt is not None and np.hypot(*(dt - db)) <= thr_m:
                meas[k], err[k] = dt, e
                refound += 1
                continue
            dt, e = _pair_full(st, a, b, db, W, H, sigma)          # может, модель не знает люфта по этой оси
            if dt is not None and e < min(thr, 0.99):
                meas[k], err[k] = dt, e
                tent.append(k)
            else:
                err[k] = 2.0
                banned[k] = True
        pos, res, w, w0, dropped, model, Bl, base, ok, Ast, lam = solve_all(meas, err)
        if tent:                                                # с новой моделью: сошлись — остаются, нет — прочь
            tent = np.array(tent)
            bad_ = tent[np.hypot(*(meas[tent] - (base[J[tent]] - base[I[tent]])).T) > thr_m]
            refound += len(tent) - len(bad_)
            if len(bad_):
                err[bad_] = 2.0
                banned[bad_] = True
                pos, res, w, w0, dropped, model, Bl, base, ok, Ast, lam = solve_all(meas, err)
    # 7) второй раунд (как в BigStitcher): где столик ошибся сильно (люфт) или шов не сошёлся с остальными —
    #    шов меряется заново, предсказание — уже по найденным положениям, окно поиска узкое
    dpred = np.hypot(*((pos[J] - pos[I]) - (pred[J] - pred[I])).T)
    redo = np.flatnonzero(((dpred > 3.0) | (res > 1.0) | ((w0 == 0) & (err < 1.5))) & ~banned)
    redone = 0
    if len(redo):
        if pr:
            pr.extend(len(redo))
        ms2 = min(max_shift, max(10.0, 3.0 * float(np.median(np.abs(res[res < 3.0]))) if (res < 3.0).any() else 10.0))
        for i_, k in enumerate(sorted(redo, key=lambda k_: pairs[k_])):   # по порядку кадров: кэш
            a, b = pairs[k]
            if pr:
                pr.update(1, f"second round: seam {i_ + 1} of {len(redo)} from the found positions")
            dt, e = _pair(st, a, b, pos[b] - pos[a], W, H, int(ms2) + 4, sigma, ms2, fine=True)
            if dt is None:
                dt, e = _pair(st, a, b, pos[b] - pos[a], W, H, int(ms2) + 4, sigma, ms2)
            if dt is not None and e <= err[k] + 0.02 and \
                    np.hypot(*(dt - (base[b] - base[a]))) <= thr_m:    # и не дальше от модели, чем окно
                meas[k], err[k] = dt, e
                redone += 1
        if redone:
            pos, res, w, w0, dropped, model, Bl, base, ok, Ast, lam = solve_all(meas, err)
    keep = (w0 > 0) & (res < 3.0)
    if pr:
        pr.stage(f"{int(keep.sum())} of {len(pairs)} seams accepted")
        pr.close()
    res_k = res[keep] if keep.any() else np.array([np.nan])
    R_ = np.linalg.solve(Ast, model)                              # поворот камеры относительно принятой ориентации
    ang = np.degrees(np.arctan2(R_[1, 0] - R_[0, 1], R_[0, 0] + R_[1, 1]))
    sc = np.sqrt(abs(np.linalg.det(model)))
    bl_um = np.hypot(*Bl) * 2 * (st.meta.get('pixel_size_um') or 1.0)
    return pos, dict(_seams=dict(I=I, J=J, meas=meas, w=np.where(keep, np.clip(1.0 - err, 0.05, 1.0) / (1.0 + (res / 2.0) ** 2), 0.0),   # для крестов (anchor): 0..1
                                    wq=np.where(keep, w, 0.0), w0=w0, err=err, base=base, lam=lam),
                     pairs=len(pairs), accepted=int(keep.sum()), threshold=round(thr, 3), dropped=dropped, second_round=redone, whole_frames=full,
                     model_window_px=round(thr_m, 1), far_from_model=int(len(far)), refound_near_model=refound,
                     backlash_um=[round(float(v), 2) for v in bl_um],
                     residual_px=f"{np.nanmedian(res_k):.2f} median, {np.nanmax(res_k):.2f} max", stage_sigma_px=round(1.0 / lam, 2),
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


def _write_levels(img, i, W, H, lvdir):
    """Кадр, уменьшенный в 2, 4, 8 раз, — в viewer/levels/ (для быстрого просмотра)."""
    lv_img = img
    for lv in (1, 2, 3):
        lv_img = cv2.resize(lv_img, (max(1, W >> lv), max(1, H >> lv)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', lv_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        buf.tofile(os.path.join(lvdir, f"{i:05d}_{lv}.jpg"))


@timed_fn('Stitch: preview and small copies')
def make_preview(folder, pos=None, max_side=4000, levels=True, verbose=False, fast=False):
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
    n = len(st)
    pr = Progress(n + 2, "Stitch: saving", unit='files') if verbose else None   # кадры + thumbs + preview_all.jpg
    from concurrent.futures import ThreadPoolExecutor
    pool, jobs = ThreadPoolExecutor(max_workers=3), []
    def frames_():
        if not fast or not st.index[0].get('file'):
            yield from st.frames()
            return
        from concurrent.futures import ThreadPoolExecutor as _TP

        def rd(i):                                          # quick: кадр раскрывается сразу вдвое меньше
            f_ = st.index[i]['file']
            if f_.lower().endswith(('.jpg', '.jpeg')):
                a = cv2.imdecode(np.fromfile(os.path.join(folder, 'frames', f_), np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
            else:
                a = cv2.resize(st._jpg(f_), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            return cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
        with _TP(max_workers=max(1, min(8, os.cpu_count() or 2))) as ex_:
            for i, a in enumerate(ex_.map(rd, range(n))):
                yield i, a
    for i, img in frames_():
        if pr:
            pr.update(1, f"frame {i + 1} of {n}: preview and 3 small copies")
        if levels:                                          # уровни пишутся в фоне (кодирование JPG — в потоках)
            while len(jobs) > 6:
                jobs.pop(0).result()
            jobs.append(pool.submit(_write_levels, img, i, W, H, lvdir))
        if thumbs is not None:
            thumbs[i] = _thumb(img)
        t = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        t = t if t.ndim == 3 else cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        x, y = int(pos[i, 0] * k), int(pos[i, 1] * k)
        out[y:y + sh, x:x + sw] = t[:out.shape[0] - y, :out.shape[1] - x]
    if pr and jobs:
        pr.stage("writing the last small copies")
    for j in jobs:
        j.result()
    pool.shutdown()
    if pr:
        pr.stage("viewer/thumbs.npz")
    if thumbs is not None:
        np.savez(os.path.join(folder, 'viewer', 'thumbs.npz'), **{f"t{i}": a for i, a in enumerate(thumbs)})
    if pr:
        pr.stage(f"join/preview_all.jpg, {out.shape[1]} x {out.shape[0]} px", 1)
    path = os.path.join(folder, 'join', 'preview_all.jpg')
    save_image(path, out, [cv2.IMWRITE_JPEG_QUALITY, 90], pr=pr)
    if pr:
        pr.stage("saved", 1)
        pr.close()
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
