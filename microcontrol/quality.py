"""Quality: точный снимок небольшого участка (кристалла) для чертежа литографии.

Отличие от quick (обзора) — геометрия строгая:
  * кадры только сдвигаются (никаких растяжений и поворотов кадра): расстояния внутри снимка — как на образце;
    дисторсия объектива (если откалибрована) и освещение поправлены при чтении кадров;
  * масштаб — пиксель объектива, откалиброванный по крестам маски (F в просмотре: кадры на кресты); без калибровки —
    по таблице, с предупреждением в отчёте;
  * съёмка: перекрытие QUALITY_OVERLAP, подъезд всегда с одной стороны (люфт не случаен), кадры PNG без потерь,
    запас вокруг рамки;
  * после съёмки — проверка и пересъёмка (до 2 кругов): размытые кадры — заново с автофокусом; кадр с объектом, не
    привязанный ни одним швом, — доснимаются кадры между ним и соседями; непокрытые места рамки — кадры туда;
  * результат — ровный прямоугольник рамки: join/quality.png (без потерь) и join/quality.json (мкм на пиксель,
    где он на столике, точность швов, что переснято).
Положение снимка на столике (и на чертеже) — по столику (модель столика из этой съёмки).
"""
import json, math, os
import numpy as np
import cv2
from . import log as L
from .util import Result

SEAM_TOL_UM = 0.1                # шов quality-снимка: расхождение не больше (мкм)


def rect_um(m, reg):
    """Прямоугольник выделения (настоящие мкм столика): x0, y0, x1, y1."""
    from .layers import geom
    return tuple(float(v) for v in geom(m, reg).bounds)


def margin_um(m):
    """Запас вокруг рамки: шаг мотора + половина люфта и разброс столика (по последней сшитой съёмке объектива)."""
    ms = float(m.motor_step)
    extra = 0.0
    try:
        from .project import latest_scans
        from .snap import geometry
        fs = [f for f in latest_scans(m.sample.folder)
              if os.path.exists(os.path.join(f, 'join', 'positions_all.csv'))]
        if fs:
            g = geometry(max(fs, key=lambda f: os.path.getmtime(os.path.join(f, 'join', 'positions_all.csv'))))
            extra = 0.5 * max(g.get('backlash_um') or [0.0]) + g.get('resid', 0.0) * g['ms'] / max(1e-9, float(np.abs(g['Mo']).max()))
    except Exception:
        pass
    return ms + extra


def plan_area(m, reg, overlap):
    """Кадры quality-съёмки: рамка + запас, перекрытие overlap, центры — на шагах мотора. -> Area (рабочие мкм)."""
    from .scan import Area
    ms = float(m.motor_step)
    x0, y0, x1, y1 = rect_um(m, reg)
    mg = margin_um(m)
    fw, fh = reg.get('fw') or m.frame_size[0], reg.get('fh') or m.frame_size[1]
    out, n = [], []
    for lo, hi, f in ((x0 - mg, x1 + mg, fw), (y0 - mg, y1 + mg, fh)):
        step = max(1, math.floor(f * (1 - overlap) / ms)) * ms
        if hi - lo <= f - 2 * ms:                             # кадр накрывает рамку с запасом: один, в середине
            c = round((lo + hi) / 2 / ms) * ms
            out.append((c, c)); n.append(1)
            continue
        k = max(2, math.ceil((hi - lo - f) / step - 1e-9) + 1)
        span = (k - 1) * step
        c0 = math.floor(((lo + hi) / 2 - span / 2) / ms + 1e-9) * ms
        out.append((c0, c0 + span)); n.append(k)
    a = Area(out[0][0] - m._off['x'], out[1][0] - m._off['y'], out[0][1] - m._off['x'], out[1][1] - m._off['y'])
    a.n = tuple(n)
    return a


class settings:
    """Съёмка quality: подъезд с одной стороны, кадры PNG без потерь (на время съёмки)."""
    def __init__(self, m):
        self.m = m

    def __enter__(self):
        c = self.m._cfg
        self.old = {k: c.get(k) for k in ('one_side', 'frame_format')}
        c.update(one_side=True, frame_format='png')
        return self

    def __exit__(self, *a):
        self.m._cfg.update(self.old)


# ---------------------------------------------------------------- проверка и снимок
def _content(st, ids):
    """Есть ли в кадре объект (контраст деталей): std лапласиана уменьшенного кадра."""
    v = {}
    for i in ids:
        g = st.frame(int(i))
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY) if g.ndim == 3 else g
        g = cv2.resize(g, (max(8, g.shape[1] // 4), max(8, g.shape[0] // 4)), interpolation=cv2.INTER_AREA)
        v[int(i)] = float(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F, ksize=3).std())
    return v


def render(folder, verbose=False):
    """Ровный прямоугольник рамки из сшитых кадров: join/quality.png + quality.json. -> dict (отчёт)."""
    from .storage import ScanData
    from .snap import stage_to_mosaic, mosaic_to_stage, geometry
    from .stitch import seams
    from .mask import Frames
    st = ScanData(folder)
    meta = st.meta
    R = meta.get('quality_rect_um')
    if not R:
        st.close()
        return None
    x0, y0, x1, y1 = R
    n = len(st)
    replaced = [i for i, r in enumerate(st.index) if int(r.get('blurry') or 0) == 2]
    blurry = [i for i, r in enumerate(st.index) if int(r.get('blurry') or 0) == 1]
    st.close()
    px = float(meta.get('pixel_size_um') or 1.0)
    g = geometry(folder)
    C = stage_to_mosaic(folder, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    xs, ys = np.sort(C[:, 0]), np.sort(C[:, 1])
    bx0, bx1, by0, by1 = xs[1], xs[2], ys[1], ys[2]           # вписанный в рамку прямоугольник мозаики
    w, h = int(math.floor(bx1 - bx0)), int(math.floor(by1 - by0))
    fr = Frames(folder, skip=replaced)
    M = np.array([[1.0, 0.0, bx0], [0.0, 1.0, by0]])
    img, cov = fr.warp(M, (w, h), 0, blend='multiband')
    fr.close()
    # непокрытые места рамки -> точки столика (туда доснять)
    gaps = []
    if not cov.all():
        k = max(1, int(max(w, h) / 400))
        small = cv2.resize((~cov).astype(np.uint8), (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_NEAREST)
        nlab, lab, stats, cents = cv2.connectedComponentsWithStats(small, 8)
        for j in range(1, nlab):
            if stats[j, cv2.CC_STAT_AREA] >= 1:
                gaps.append(mosaic_to_stage(folder, [(bx0 + cents[j][0] * k, by0 + cents[j][1] * k)])[0].tolist())
    # швы: кадры, не привязанные ни одним принятым швом (их место — только по столику)
    z = seams(folder)
    loose, stage_only = [], []
    res_px = None
    if z is not None and n > 1:
        acc = np.zeros(n, int)
        wq = z['wq' if 'wq' in z else 'w']
        for a, b, wv in zip(z['I'], z['J'], wq):
            if wv > 0:
                acc[int(a)] += 1; acc[int(b)] += 1
        stage_only = [i for i in range(n) if acc[i] == 0 and i not in replaced]
        if stage_only:
            st = ScanData(folder)
            try:
                from .optics import attach
                attach(st, folder)
                cv_ = _content(st, range(n))
            finally:
                st.close()
            med = float(np.median(list(cv_.values())))
            loose = [i for i in stage_only if cv_[i] > 1.5 * med]   # с объектом — не годится на одном столике
        pos = g['pos']
        r_ = np.hypot(*(pos[z['J'].astype(int)] - pos[z['I'].astype(int)] - z['meas']).T)[wq > 0]
        res_px = (float(np.median(r_)), float(r_.max())) if len(r_) else None
    A = np.array([mosaic_to_stage(folder, [(bx0, by0)])[0], mosaic_to_stage(folder, [(bx0 + 1, by0)])[0],
                  mosaic_to_stage(folder, [(bx0, by0 + 1)])[0]])
    aff = np.c_[A[1] - A[0], A[2] - A[0], A[0]]                # пиксель (u, v, 1) снимка -> мкм столика
    su = float(np.hypot(*aff[:, 0]))                           # мкм столика на пиксель (по столику)
    ang = float(np.degrees(np.arctan2(aff[1, 0], aff[0, 0])))
    src = meta.get('pixel_source') or 'table'
    rep = dict(rect_um=[round(v, 3) for v in R], size_px=[w, h], pixel_um=px, pixel_source=src,
               size_um=[round(w * px, 3), round(h * px, 3)], pixel_to_stage_um=aff.round(6).tolist(),
               stage_um_per_px=round(su, 6), stage_vs_pixel=round(su / px - 1, 5), turn_deg=round(ang, 4),
               coverage=round(float(cov.mean()), 5), gaps_um=gaps, frames=n, replaced=replaced, blurry=blurry,
               stage_only=stage_only, loose=loose,
               seam_px=None if res_px is None else [round(res_px[0], 3), round(res_px[1], 3)],
               seam_um=None if res_px is None else [round(res_px[0] * px, 4), round(res_px[1] * px, 4)],
               reshot=int(meta.get('reshot', 0)), rounds=int(meta.get('quality_rounds', 0)))
    problems = []
    if gaps:
        problems.append(f"{len(gaps)} uncovered place(s) in the box")
    if loose:
        problems.append(f"frame(s) {', '.join(str(i + 1) for i in loose)} with the object tied by no seam")
    if blurry:
        problems.append(f"blurry frame(s) {', '.join(str(i + 1) for i in blurry)}")
    tol_px = max(1.0, SEAM_TOL_UM / px)                        # шов: не хуже 0.1 мкм (и не строже 1 px)
    if res_px is not None and res_px[1] > tol_px:
        problems.append(f"a seam off by {res_px[1]:.2f} px = {res_px[1] * px:.3f} um")
    if src != 'crosses':
        problems.append("pixel size not calibrated by the crosses for this objective (F in the view of a scan "
                        "with crosses)")
    rep['problems'] = problems
    rep['ok'] = not [p for p in problems if not p.startswith("pixel size")]
    out = os.path.join(folder, 'join')
    ok_, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    buf.tofile(os.path.join(out, 'quality.png'))
    with open(os.path.join(out, 'quality.json'), 'w', encoding='utf-8') as f:
        json.dump(rep, f, indent=1)
    L.event('QLT-001', f"{folder}: {w} x {h} px, {px:.5f} um/px ({src}), coverage {rep['coverage']}, "
                       f"seams {rep['seam_px']} px, problems: {problems or 'none'}.")
    return rep


def info(folder):
    """Отчёт quality-снимка (quality.json) или None."""
    p = os.path.join(folder, 'join', 'quality.json')
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_quality(folder):
    try:
        with open(os.path.join(folder, 'meta.json'), encoding='utf-8') as f:
            return json.load(f).get('stitch_mode') == 'quality'
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------- пересъёмка (у прибора)
def _targets(folder, rep):
    """Что доснять: [(вид, шаги x, y, z или None)]."""
    from .storage import ScanData
    from .stitch import seams
    st = ScanData(folder)
    p = np.array([[r['x_p'], r['y_p'], r['z_p']] for r in st.index], float)
    ms = float(st.meta.get('motor_step_um') or 1.0)
    st.close()
    out = []
    for i in rep.get('blurry', []):
        out.append(('blurry', int(p[i, 0]), int(p[i, 1]), None, i))
    z = seams(folder)
    if rep.get('loose') and z is not None:
        for i in rep['loose']:
            nb = set(int(b) for a, b in zip(z['I'], z['J']) if int(a) == i) | \
                set(int(a) for a, b in zip(z['I'], z['J']) if int(b) == i)
            for j in sorted(nb):
                q = (p[i] + p[j]) / 2
                out.append(('bridge', int(round(q[0])), int(round(q[1])), float(q[2]), i))
    for gx, gy in rep.get('gaps_um', []):
        k = int(np.argmin(np.hypot(p[:, 0] * ms - gx, p[:, 1] * ms - gy)))
        out.append(('gap', int(round(gx / ms)), int(round(gy / ms)), float(p[k, 2]), None))
    seen, uniq = set(), []
    for t in out:
        key = (t[0] == 'blurry', t[1], t[2])
        if key not in seen:
            seen.add(key); uniq.append(t)
    return uniq


def reshoot(m, folder, rep, verbose=True):
    """Доснять кадры по отчёту (у прибора). -> сколько кадров снято."""
    from .storage import ScanData
    from . import focus as F
    tg = _targets(folder, rep)
    if not tg:
        return 0
    st = ScanData.append(folder)
    repl = []
    try:
        with settings(m):
            for kind, xs, ys, zs, src in tg:
                t = {'x': xs, 'y': ys}
                if zs is not None:
                    t['z'] = int(round(zs))
                m._check(t)
                m._go(t)
                if kind == 'blurry':
                    m.focus()
                img = m.image()
                s = F.sharpness(img, m.focus_method)
                p = m.position
                st.add(img, row='', col='', x_um=p.x, y_um=p.y, z_um=p.z, x_p=m._p['x'], y_p=m._p['y'],
                       z_p=m._p['z'], sharpness=round(s, 1), refocused=int(kind == 'blurry'), blurry=0, t_s=0,
                       dir_x=m._dir.get('x', 0), dir_y=m._dir.get('y', 0))
                if kind == 'blurry' and src is not None:
                    repl.append(src)
                L.event('QLT-002', f"Re-shot ({kind}) at steps {xs}, {ys}" + (f" for frame {src + 1}" if src is not None else "") + ".")
    finally:
        st.close_append(replaced=repl, reshot=len(tg))
    return len(tg)


def finish(m, folder, rounds=2, verbose=True):
    """После quality-съёмки: сшить, проверить, доснять (до rounds кругов), ровный прямоугольник. -> Result."""
    from .stitch import stitch
    rep, total = None, 0
    for k in range(rounds + 1):
        stitch(folder, verbose=verbose)
        rep = info(folder) or {}
        if k == rounds or rep.get('ok'):
            break
        n = reshoot(m, folder, rep, verbose=verbose)
        if not n:
            break
        total += n
        _meta_add(folder, quality_rounds=k + 1)
    r = Result("Quality picture" + ("" if rep.get('ok') else " (check the problems)"),
               **{"Box, um": f"{rep['size_um'][0]:.2f} x {rep['size_um'][1]:.2f}",
                  "Picture, px": f"{rep['size_px'][0]} x {rep['size_px'][1]}",
                  "Pixel, um": f"{rep['pixel_um']:.5f} ({'calibrated by the crosses' if rep['pixel_source'] == 'crosses' else 'table value, not calibrated'})",
                  "Seams, px": "-" if rep.get('seam_px') is None else f"{rep['seam_px'][0]:.2f} median, {rep['seam_px'][1]:.2f} max",
                  "Re-shot frames": total, "Covered": f"{100 * rep['coverage']:.2f} %",
                  "File": os.path.join(folder, 'join', 'quality.png')}) if rep.get('size_px') else \
        Result("Quality picture not made", Folder=folder)
    if rep.get('problems'):
        r['Problems'] = "; ".join(rep['problems'])
    print(r)
    return r


def _meta_add(folder, **kw):
    p = os.path.join(folder, 'meta.json')
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    d.update(kw)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=1, default=str)
