"""Чертёж образца для AutoCAD: выбранные участки (их последние съёмки) одним файлом — картинки на своих местах,
контуры, подписи, кресты решётки и номера квадратов, пустой слой BONDS для бондов.

Координаты — мкм по решётке образца (lattice.py): начало в кресте 1,1 литографии, X — к кресту 2,1, Y — к 1,2
(на чертеже вверх, как в AutoCAD), крест X,Y — в ((X−1)·P, −(Y−1)·P). Нужна решётка с номерами (C и N в просмотре).
Клетки выделений (вложенные сетки) — каждая целиком на слое своего выделения, с адресом «12,7 > 2,1».
Картинка участка — его сшитая мозаика (join/preview_all.jpg) с прозрачным фоном, поставлена строго по геометрии
съёмки (snap): каждый её пиксель — в своей точке. Картинки в DXF / DWG — внешние файлы: папку переносить целиком.

    path = mc.export_sample(m)                  # все снятые участки, отмеченные для экспорта
    path = mc.export_sample(m, [0, 2, 5])       # эти (индексы выделений)
"""
import os
import json
import time
import numpy as np
import cv2


def _grid_uv(g):
    """Сетка скана -> решётка образца (с номерами): uv = Pm·(i, j) + c0, крест X,Y — в uv (X−1, Y−1).
    -> (Pm, c0) или None, если номера квадратов не заданы однозначно."""
    Pm, t, note = g.numbering()
    if not g.labels or 'fixed' not in note:
        return None
    Pm = np.asarray(Pm, float)
    return Pm, np.asarray(t, float) - 0.5 - Pm @ np.array([0.5, 0.5])


def local_picture(folder, g, uv0, uv1, pitch, max_px=8000, nodes=None):
    """Картинка участка строго в координатах решётки: каждый пиксель чертежа -> точка мозаики по 4 крестам
    своей клетки (билинейно; кресты измерены на этой съёмке, ненайденные — по сетке). Ошибка сшивки,
    остаток дисторсии и наклон не попадают в чертёж: внутри клетки всё привязано к её крестам.
    uv0, uv1 — углы прямоугольника в uv (крест X,Y — в (X−1, Y−1)). -> (BGRA, мкм на пиксель, отчёт)."""
    from .mask import Frames, measure_crosses
    gu = _grid_uv(g)
    if gu is None:
        return None
    Pm, c0 = gu
    Pi = np.linalg.inv(Pm)
    fr = Frames(folder)
    try:
        px = fr.st.meta.get('pixel_size_um') or 1.0
        w_um, h_um = (uv1 - uv0) * pitch
        res = max(px, max(w_um, h_um) / max_px)               # мкм на пиксель картинки
        W, H = int(np.ceil(w_um / res)), int(np.ceil(h_um / res))
        # узлы сетки этой съёмки вокруг участка
        cs = np.array([uv0, (uv1[0], uv0[1]), uv1, (uv0[0], uv1[1])])
        gc = (cs - c0) @ Pi.T
        i_lo, j_lo = np.floor(gc.min(0)).astype(int) - 1
        i_hi, j_hi = np.ceil(gc.max(0)).astype(int) + 1
        allN = [(i, j) for i in range(i_lo, i_hi + 1) for j in range(j_lo, j_hi + 1)]
        meas = measure_crosses(folder, g, allN, fr)
        Nx = np.zeros((i_hi - i_lo + 1, j_hi - j_lo + 1, 2))
        dev = []
        for i, j in allN:
            q = g.cr.point(i, j)
            if (i, j) in meas:
                dev.append(np.hypot(*(meas[(i, j)] - q)))
                q = meas[(i, j)]
            Nx[i - i_lo, j - j_lo] = q
        # пиксель картинки (u — вправо, v — вниз) -> uv -> (i, j) сетки -> мозаика
        u = (np.arange(W) + 0.5) * res / pitch + uv0[0]
        img = np.full((H, W, 3), 255, np.uint8)
        cov = np.zeros((H, W), bool)
        Nx = Nx.astype(np.float32)
        for r0 in range(0, H, 512):                          # полосами: память не растёт с размером
            v = uv0[1] + (np.arange(r0, min(H, r0 + 512)) + 0.5) * res / pitch   # строки вниз: uv_y растёт
            UU, VV = np.meshgrid(u, v)
            gi = Pi[0, 0] * (UU - c0[0]) + Pi[0, 1] * (VV - c0[1]) - i_lo
            gj = Pi[1, 0] * (UU - c0[0]) + Pi[1, 1] * (VV - c0[1]) - j_lo
            i0 = np.clip(np.floor(gi).astype(np.int32), 0, Nx.shape[0] - 2)
            j0 = np.clip(np.floor(gj).astype(np.int32), 0, Nx.shape[1] - 2)
            s_, t_ = (gi - i0).astype(np.float32)[..., None], (gj - j0).astype(np.float32)[..., None]
            Q = ((1 - s_) * (1 - t_) * Nx[i0, j0] + s_ * (1 - t_) * Nx[i0 + 1, j0] +
                 (1 - s_) * t_ * Nx[i0, j0 + 1] + s_ * t_ * Nx[i0 + 1, j0 + 1])
            im_, cv_ = fr.remap(np.ascontiguousarray(Q[..., 0]), np.ascontiguousarray(Q[..., 1]), 255)
            img[r0:r0 + len(v)], cov[r0:r0 + len(v)] = im_, cv_
        rep = dict(crosses_measured=len(meas), crosses_around=len(allN),
                   off_grid_um=round(float(np.sqrt(np.mean(np.square(dev)))) * px, 3) if dev else None,
                   max_off_grid_um=round(float(np.max(dev)) * px, 3) if dev else None, um_per_px=round(res, 4))
    finally:
        fr.close()
    return np.dstack([img, cov.astype(np.uint8) * 255]), res, rep


def _overview_png(path, pics, outlines, max_side=12000, res=None, max_px=200e6):
    """Картинки (pics: объектив, BGR, 2x3 пиксель -> мкм чертежа, маска) — в одну PNG по порядку pics (первая —
    снизу, последняя — сверху: как слои списка); res — мкм на пиксель (None — как у самой точной картинки); не
    больше max_px пикселей и max_side по стороне (тогда крупнее, и это пишется в отчёт); контуры — тонко.
    -> (путь, мкм на пиксель, уменьшено ли)."""
    corners = []
    for o, img, A, a in pics:
        h, w = img.shape[:2]
        corners.append(np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float) @ A.T)
    allc = np.vstack(corners + [q for q, _ in outlines])
    x0, y0 = allc.min(0)
    x1, y1 = allc.max(0)
    want = float(res) if res else min(float(np.hypot(*A[:, 0])) for _, _, A, _ in pics)
    res = max(want, max(x1 - x0, y1 - y0) / max_side, ((x1 - x0) * (y1 - y0) / max_px) ** 0.5)
    W, H = int(np.ceil((x1 - x0) / res)) + 1, int(np.ceil((y1 - y0) / res)) + 1
    out = np.full((H, W, 3), 255, np.uint8)
    for o, img, A, a in pics:
        img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        Ai = np.linalg.inv(np.r_[A, [[0, 0, 1]]])                  # мкм чертежа -> пиксель картинки
        Mo = Ai @ np.array([[res, 0, x0], [0, -res, y1], [0, 0, 1]])   # пиксель PNG -> пиксель картинки
        mk = np.ones(img.shape[:2], np.uint8) * 255 if a is None else a
        Mf = np.linalg.inv(Mo)                                     # пиксель картинки -> пиксель PNG: где она ляжет
        h_, w_ = img.shape[:2]
        cq = np.array([[0, 0, 1], [w_, 0, 1], [0, h_, 1], [w_, h_, 1]], float) @ Mf[:2].T
        bx0, by0 = np.maximum(np.floor(cq.min(0)).astype(int) - 2, 0)
        bx1, by1 = np.minimum(np.ceil(cq.max(0)).astype(int) + 2, (W, H))
        if bx1 <= bx0 or by1 <= by0:
            continue
        from .util import warp_clean                               # только свой прямоугольник; без чёрной каймы
        step = max(64, int(2e7 / max(1, bx1 - bx0)))              # полосами: память не на весь (до 200 Мпикс) снимок
        for ty0 in range(by0, by1, step):
            ty1 = min(by1, ty0 + step)
            Mb = Mo @ np.array([[1, 0, bx0], [0, 1, ty0], [0, 0, 1]], float)
            wimg, wm = warp_clean(img, Mb[:2], (bx1 - bx0, ty1 - ty0), alpha=mk, inverse=True)
            sub = out[ty0:ty1, bx0:bx1]
            sub[wm] = wimg[wm]
    for q, n in outlines:
        pq = np.c_[(q[:, 0] - x0) / res, (y1 - q[:, 1]) / res]
        cv2.polylines(out, [np.int32(np.round(pq))], True, (0, 0, 255), max(1, W // 3000), cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    buf.tofile(path)
    return path, res, res > want * 1.001


def _preview_pic(folder, D):
    """Обзор съёмки (preview_all.jpg) для mixed.png: (картинка, 2x3 пиксель -> мкм чертежа, маска) или None."""
    from .snap import mosaic_to_stage, geometry
    jp = os.path.join(folder, 'join', 'preview_all.jpg')
    img = cv2.imread(jp) if os.path.exists(jp) else None
    if img is None:
        return None
    g = geometry(folder)
    size = g['pos'].max(0) + (g['W'], g['H'])
    kk = min(1.0, 4000 / size.max())
    h_, w_ = img.shape[:2]
    C3 = D(mosaic_to_stage(folder, np.array([[0, 0], [w_, 0], [0, h_]], float) / kk))
    return img, np.c_[(C3[1] - C3[0]) / w_, (C3[2] - C3[0]) / h_, C3[0]], (img.max(2) > 6).astype(np.uint8) * 255


def best_scan(m, r):
    """Съёмка выделения для чертежа: основная (Scans -> Make main), иначе самая свежая quality, иначе последняя."""
    from .project import Selection
    from . import quality as Q
    sel = Selection(m.sample, r)
    if r.get('main'):
        return sel.latest
    q = [s_ for s_ in sel.scans if Q.is_quality(s_.folder)]
    return q[-1] if q else sel.latest


def export_message(path, n):
    """Строка для окна: что вышло из export_sample (path — его результат), n — сколько выделений."""
    d_ = os.path.dirname(str(path))
    s_ = f"Exported {n} selection{'s' if n != 1 else ''} to {d_}"
    if str(path).endswith('drawing.json'):
        try:
            why_ = json.load(open(path, encoding='utf-8')).get('no_drawing') or []
        except (OSError, ValueError):
            why_ = []
        return s_ + ": pictures and mixed.png, no drawing (" + "; ".join(why_) + ")"
    return s_ + ("" if str(path).endswith('.dwg') else "  (DWG: install ODA File Converter)")


def export_sample(m, ks=None, out=None, dwg=True, verbose=True, nested=False, png=True, mixed_um=None, drawing=None):
    """Экспорт образца в одну папку <образец>/export/ (прежнее там заменяется): drawing.dxf (и .dwg), снимки
    выделений S03_50x.png, mixed.png — все снимки одной картинкой с наложением в порядке слоёв списка (верх списка —
    сверху), drawing.json — отчёт. ks — индексы m._regions (по умолчанию — все с галочкой «в экспорт»): снятое —
    картинка (в чертёж — только quality) и контур, не снятое и draft — контур (и клетки). mixed_um — мкм на
    пиксель mixed.png (None — как у самого точного снимка). Решётка с номерами — чертёж от креста 1,1; без неё —
    в координатах столика (мкм, Y вверх). Чертёж (DXF / DWG) — только когда кресты размечены (решётка с номерами) и
    все отмеченные выделения сшиты quality: кресты — маяки, по которым снимки ложатся без ошибки для литографии;
    иначе — снимки и mixed.png без чертежа (причина — в drawing.json, no_drawing). drawing=True — чертёж всё равно
    (в координатах столика, если крестов нет), False — без чертежа.
    Возвращает путь к DWG / DXF, а без чертежа — к drawing.json."""
    import ezdxf
    from .project import Selection
    from .samples import obj, num, title
    from .layers import rings
    from .snap import mosaic_to_stage, geometry
    from .util import Progress
    from .mask import dxf_to_dwg
    from .samples import order as _order
    if ks is None:
        ks = [k for k, r in enumerate(m._regions) if r.get('kind') != 'group' and r.get('export', False)]
    ks = [k for k in ks if m._regions[k].get('kind') != 'group']
    if not ks:
        raise ValueError("Nothing to export: tick the selections to export (the box on their cards)")
    from . import quality as Qx
    nqx_ = [title(m, k) for k in ks if m._regions[k].get('kind') == 'draft' or
            not (lambda sc_: sc_ is not None and Qx.is_quality(sc_.folder))(best_scan(m, m._regions[k]))]
    if nqx_ and drawing is None:                                # не quality — не экспорт: картинки разной точности
        raise ValueError("Not exported: not quality-stitched - " + ", ".join(nqx_) +    # вводят в заблуждение
                         " (Stitch / Shift+J for each, or untick them)")
    out = out or os.path.join(m.sample.folder, 'export')
    os.makedirs(out, exist_ok=True)
    for f_ in os.listdir(out):                                   # одна папка: прежний экспорт заменяется
        p_ = os.path.join(out, f_)
        if os.path.isfile(p_) and f_.lower().endswith(('.png', '.dxf', '.dwg', '.json', '.bak')):
            try:
                os.remove(p_)
            except OSError:
                pass
    from . import lattice as LT
    from .layers import lattice_of
    L = lattice_of(m)
    P = float(getattr(m, 'mask_pitch', 0) or 100.0)          # шаг крестов маски: все размеры чертежа — от него
    stage_frame = L is None or not L.get('numbered') or L.get('stage')
    if stage_frame and verbose:
        print("  [!] No numbered cross lattice: the drawing is in stage coordinates (um). For the lithography frame "
              "mark the crosses (C) and number two squares (N) in the viewer")
    from .mask import Grid as _G
    try:
        lab_ = list(_G(L['src']).labels.values())
    except Exception:
        lab_ = []
    if any(not (1 <= X <= 58 and 1 <= Y <= 58) for X, Y in lab_) and verbose:
        print("  [!] Some square numbers are outside 1..58: check the numbering (N in the viewer)")
    from . import quality as Q0
    no_draw_ = []                                               # почему без чертежа
    if stage_frame:
        no_draw_.append("the crosses are not marked (C and N in the view)")
    nq_ = [title(m, k) for k in ks if m._regions[k].get('kind') == 'draft' or
           not (lambda sc_: sc_ is not None and Q0.is_quality(sc_.folder))(best_scan(m, m._regions[k]))]
    if nq_:
        no_draw_.append("not quality-stitched: " + ", ".join(nq_))
    make_dxf = (not no_draw_) if drawing is None else bool(drawing)
    D = (lambda S: np.asarray(S, float) * (1.0, -1.0)) if stage_frame else \
        (lambda S: LT.to_uv(L, S) * (P, -P))                    # столик -> чертёж: крест 1,1 в (0, 0), Y вверх

    doc = ezdxf.new('R2010', setup=True)
    doc.units = 13
    doc.header['$INSUNITS'] = 13                                  # микрометры
    for name, color in (('CROSSES', 2), ('NUMBERS', 8), ('LABELS', 7), ('ADDRESSES', 8), ('BONDS', 6)):
        doc.layers.add(name, color=color)
    msp = doc.modelspace()
    pr = Progress(len(ks) + 2, "Export: drawing", unit='steps') if verbose else None
    rank_ = {k: i for i, k in enumerate(_order(m))}
    order = sorted(ks, key=lambda k: -rank_.get(k, 0))           # низ списка — первым (снизу), верх — последним
    placed = []
    local_rep = {}
    pics_ = []                                                  # (объектив, картинка, пиксель -> чертёж, прозрачность)
    mix_ = []                                                   # для mixed.png: всё снятое, по порядку слоёв

    def _addp(t_):
        pics_.append(t_)
        mix_.append(t_)
    skipped_ = []
    for i, k in enumerate(order):
        r = m._regions[k]
        lay = f"S{num(m, k):02d}_draft" if r.get('kind') == 'draft' else f"S{num(m, k):02d}_{obj(m, r)}x"
        doc.layers.add(lay, color=(i % 6) + 1)
        sc = best_scan(m, r) if r.get('kind') != 'draft' else None
        jp = os.path.join(sc.folder, 'join', 'preview_all.jpg') if sc else None
        if pr:
            pr.update(1, f"{title(m, k)}: picture and outline")
        from . import quality as Q
        qi = Q.info(sc.folder) if sc else None
        qscan_ = sc.folder if (sc and Q.is_quality(sc.folder)) else None
        if sc and not Q.is_quality(sc.folder):                  # quick (обзор) — не для чертежа: только контур
            mi_ = _preview_pic(sc.folder, D)                    # но в mixed.png — да
            if mi_ is not None:
                mix_.append((obj(m, r) or 0,) + mi_)
            jp = None
            skipped_.append(f"{title(m, k)}: {Q.stitched_as(sc.folder) or 'not stitched'} - outline only (the picture "
                            "goes to the drawing after the quality stitch: Stitch)")
            sc = None
        if qi is not None and not qscan_:
            qi = None
        if sc is not None and qi is not None:                   # quality: ровный снимок рамки, строго по столику
            qp = os.path.join(sc.folder, 'join', 'quality.png')
            img = cv2.imread(qp)
            if img is not None:
                h_, w_ = img.shape[:2]
                aff = np.array(qi['pixel_to_stage_um'], float)
                px2 = lambda u, v: D((aff @ np.array([u, v, 1.0]))[None])[0]
                C0, Cu, Cv = px2(0, h_), px2(1, h_), px2(0, h_ - 1)      # левый нижний угол; вправо; вверх
                png = f"{lay}.png"
                cv2.imwrite(os.path.join(out, png), img)
                idef = doc.add_image_def(filename=png, size_in_pixel=(w_, h_))
                im = msp.add_image(idef, insert=tuple(C0), size_in_units=(1, 1), dxfattribs={'layer': lay})
                im.dxf.u_pixel = (float(Cu[0] - C0[0]), float(Cu[1] - C0[1]), 0.0)
                im.dxf.v_pixel = (float(Cv[0] - C0[0]), float(Cv[1] - C0[1]), 0.0)
                im.dxf.insert = (float(C0[0]), float(C0[1]), 0.0)
                im.dxf.flags = 1 | 2 | 8
                p00, p10, p01 = px2(0, 0), px2(1, 0), px2(0, 1)         # пиксель (u, v) -> мкм чертежа
                _addp((obj(m, r) or 0, img, np.c_[p10 - p00, p01 - p00, p00], ((img.max(2) if img.ndim == 3 else img) > 6).astype(np.uint8) * 255))   # чёрное (не снято) — прозрачно
                local_rep[lay] = dict(quality=True, pixel_um=qi['pixel_um'], pixel_source=qi['pixel_source'],
                                      seam_um=qi.get('seam_um'), size_um=qi['size_um'])
                if qi['pixel_source'] != 'crosses':
                    skipped_.append(f"{title(m, k)}: pixel size not calibrated by the crosses (distances by the "
                                    "table value of the objective)")
            sc, jp = None, None
        gs_ = None
        if sc and not stage_frame:
            try:
                gs_ = _G(sc.folder)
                gs_ = gs_ if gs_.ready() and _grid_uv(gs_) is not None else None
            except Exception:
                gs_ = None
        if gs_ is not None:                                     # картинка — по крестам этой съёмки, клетка за клеткой
            qq = np.vstack([D(rg) for rg in rings(m, r)]) / (P, -P)   # uv участка
            uv0, uv1 = qq.min(0) - 0.05, qq.max(0) + 0.05
            uv0, uv1 = np.array([uv0[0], uv0[1]]), np.array([uv1[0], uv1[1]])
            res_ = local_picture(sc.folder, gs_, uv0, uv1, P)
            if res_ is not None:
                img4, rs_, rep_ = res_
                png = f"{lay}.png"
                cv2.imwrite(os.path.join(out, png), img4)
                h_, w_ = img4.shape[:2]
                idef = doc.add_image_def(filename=png, size_in_pixel=(w_, h_))
                x0_, y0_ = uv0[0] * P, -uv0[1] * P - h_ * rs_          # левый нижний угол, мкм чертежа
                im = msp.add_image(idef, insert=(x0_, y0_), size_in_units=(w_ * rs_, h_ * rs_), dxfattribs={'layer': lay})
                im.dxf.u_pixel = (float(rs_), 0.0, 0.0)
                im.dxf.v_pixel = (0.0, float(rs_), 0.0)
                im.dxf.flags = 1 | 2 | 4 | 8
                try:                                                    # строго по контуру участка
                    ring0 = D(max(rings(m, r), key=len))
                    im.set_boundary_path([(float((x - x0_) / rs_), float(h_ - (y - y0_) / rs_)) for x, y in ring0])
                except Exception:
                    pass
                local_rep[lay] = rep_
                _addp((obj(m, r) or 0, img4[..., :3] if img4.ndim == 3 else img4,
                              np.array([[rs_, 0.0, x0_], [0.0, -rs_, y0_ + h_ * rs_]]),
                              img4[..., 3] if img4.ndim == 3 and img4.shape[2] == 4 else None))
                jp = None                                               # старый способ не нужен
        if jp and os.path.exists(jp):
            g = geometry(sc.folder)
            img = cv2.imread(jp)
            size = g['pos'].max(0) + (g['W'], g['H'])
            kk = min(1.0, 4000 / size.max())                       # масштаб превью (как в stitch.make_preview)
            a = (img.max(2) > 6).astype(np.uint8) * 255            # вне кадров — прозрачно
            png = f"{lay}.png"
            cv2.imwrite(os.path.join(out, png), np.dstack([img, a]))
            h_, w_ = img.shape[:2]
            C3 = D(mosaic_to_stage(sc.folder, np.array([[0, 0], [w_, 0], [0, h_]], float) / kk))
            ux, vy = (C3[1] - C3[0]) / w_, (C3[2] - C3[0]) / h_         # чертёж на пиксель: вправо, вниз
            _addp((obj(m, r) or 0, img, np.c_[ux, vy, C3[0]], a))
            idef = doc.add_image_def(filename=png, size_in_pixel=(w_, h_))
            im = msp.add_image(idef, insert=tuple(C3[2]), size_in_units=(1, 1), dxfattribs={'layer': lay})
            im.dxf.u_pixel = (float(ux[0]), float(ux[1]), 0.0)     # точно по геометрии: поворот, масштаб
            im.dxf.v_pixel = (float(-vy[0]), float(-vy[1]), 0.0)   # вверх по картинке
            im.dxf.insert = (float(C3[2][0]), float(C3[2][1]), 0.0)  # левый нижний угол
            im.dxf.flags = 1 | 2 | 4 | 8                            # видно, прозрачный фон, обрезка по контуру
            try:                                                    # картинка — строго по контуру участка
                from .snap import stage_to_mosaic
                ring0 = max(rings(m, r), key=len)
                q_ = stage_to_mosaic(sc.folder, ring0) * kk
                im.set_boundary_path([(float(x), float(y)) for x, y in q_])
            except Exception:
                pass
        for ring in rings(m, r):
            q = D(ring)
            msp.add_lwpolyline([tuple(p) for p in q], close=True, dxfattribs={'layer': lay})
        if r.get('cells') and not stage_frame:                 # клетки — каждая целиком (общие грани — на каждом слое)
            N = LT.level(r)
            many = len(r['cells']) > 400
            from .layers import geom as _geom
            from shapely.geometry import Polygon as _Pg
            rg_ = _geom(m, r).buffer(1e-3)
            for a, b in r['cells']:
                cs_ = LT.cell_stage(L, a, b, N)
                if not rg_.contains(_Pg(cs_)):                  # клетка на краю — по краю участка (пирамидка)
                    cut_ = _Pg(cs_).intersection(_geom(m, r))
                    cut_ = max(getattr(cut_, 'geoms', [cut_]), key=lambda p_: p_.area)
                    if cut_.is_empty or cut_.geom_type != 'Polygon':
                        continue
                    cs_ = np.array(cut_.exterior.coords)[:-1]
                q = D(cs_)
                msp.add_lwpolyline([tuple(p) for p in q], close=True, dxfattribs={'layer': lay})
                if not many:
                    msp.add_text(LT.address(a, b, r.get('factors') or []), height=0.08 * P / N,
                                 dxfattribs={'layer': 'ADDRESSES', 'insert': (q[0][0] + 0.05 * P / N, q[0][1] - 0.15 * P / N)})
        q = np.vstack([D(rg) for rg in rings(m, r)])
        x0, y1 = q[:, 0].min(), q[:, 1].max()
        hgt = max(1.0, min(0.12 * P, 0.03 * max(np.ptp(q[:, 0]), np.ptp(q[:, 1]))))
        msp.add_text(f"{title(m, k)}  " + (f"{obj(m, r)}x" if r.get('kind') != 'draft' else "draft"), height=hgt, dxfattribs={'layer': 'LABELS', 'insert': (x0, y1 + hgt * 0.4)})
        qd_ = (local_rep.get(lay) or {}).get('quality')
        placed.append(dict(selection=num(m, k), objective=obj(m, r), layer=lay, scan=qscan_ or (sc.folder if sc else None),
                           placed=("quality picture, by the stage" if qd_ else
                                   "by the crosses of its squares" if lay in local_rep else
                                   "by the stage" if sc else "outline only (not a quality scan)"),
                           crosses=local_rep.get(lay)))
        if qscan_ and (obj(m, r) or 0) < 50 and verbose:
            print(f"  [!] {title(m, k)} is {obj(m, r)}x: for lithography outline the flakes at 50x / 100x "
                  "(at 10x the optics blur edges by ~1 um)")
    if pr:                                                       # кресты литографии с их номерами
        pr.update(1, "crosses and their numbers")
    allq = np.vstack([D(rg) for k in ks for rg in rings(m, m._regions[k])])
    (x0, y0), (x1, y1) = allq.min(0), allq.max(0)
    arm = 0.1 * P
    for X in ([] if stage_frame else range(max(1, int(np.floor(x0 / P)) + 1), int(np.ceil(x1 / P)) + 2)):
        for Y in range(max(1, int(np.floor(-y1 / P)) + 1), int(np.ceil(-y0 / P)) + 2):
            cx, cy = (X - 1) * P, -(Y - 1) * P
            msp.add_line((cx - arm, cy), (cx + arm, cy), dxfattribs={'layer': 'CROSSES'})
            msp.add_line((cx, cy - arm), (cx, cy + arm), dxfattribs={'layer': 'CROSSES'})
            msp.add_text(f"{X},{Y}", height=0.06 * P, dxfattribs={'layer': 'NUMBERS', 'insert': (cx + 0.04 * P, cy + 0.04 * P)})
    path = os.path.join(out, 'drawing.dxf')
    if make_dxf:
        doc.saveas(path)
    mixed_ = None
    if png and mix_:                                             # mixed.png: все снимки с наложением по слоям
        try:
            _p, res_, cut_ = _overview_png(os.path.join(out, 'mixed.png'), mix_,
                                           [(D(rg), num(m, k)) for k in ks for rg in rings(m, m._regions[k])],
                                           max_side=60000, res=mixed_um)
            mixed_ = dict(file='mixed.png', um_per_px=round(float(res_), 5), coarser_than_asked=bool(cut_))
            if cut_:
                skipped_.append(f"mixed.png: {res_:.3f} um/px (finer would be over 200 Mpx)")
        except Exception as e:
            skipped_.append(f"mixed.png not made: {e}")
    json.dump(dict(frame=("stage coordinates, um, Y up" if stage_frame else
                          "lattice: cross 1,1 at (0, 0), X to cross 2,1, Y up (to cross 1,2 is -Y)"), pitch_um=P,
                   lattice=L, areas=placed, pictures=local_rep, mixed=mixed_, notes=skipped_,
                   drawing=make_dxf, no_drawing=[] if make_dxf else no_draw_),
              open(os.path.join(out, 'drawing.json'), 'w'), indent=1, default=float)
    if verbose:
        for s_ in skipped_:
            print("  [!] " + s_)
        if not make_dxf:
            print("  [!] No drawing: " + "; ".join(no_draw_) + " - pictures and mixed.png only")
    if not make_dxf:
        if pr:
            pr.update(1, "no drawing")
            pr.close()
        return os.path.join(out, 'drawing.json')
    dw = dxf_to_dwg(path) if dwg else None
    if pr:
        pr.update(1, "DWG" if dw else "DXF (DWG: ODA File Converter not found)")
        pr.close()
    return dw or path
