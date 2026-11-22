"""Чертёж образца для AutoCAD: выбранные участки (их последние съёмки) одним файлом — картинки на своих местах,
контуры, подписи, кресты решётки и номера квадратов, пустой слой BONDS для бондов.

Координаты — мкм, по решётке крестов (если её отметили в обзорной съёмке): начало в кресте (0, 0) решётки,
ось X вдоль её строк, Y вверх; без решётки — мкм столика (Y вверх).
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


def _lattice(m, ks):
    """Решётка крестов: из самой крупной (меньшее увеличение) съёмки среди ks, где она отмечена.
    -> (O, u1, u2, сетка, папка) в мкм столика или None."""
    from .project import Selection
    from .samples import obj
    from .mask import Grid
    from .snap import mosaic_to_stage
    for k in sorted(ks, key=lambda k_: obj(m, m._regions[k_])):
        sc = Selection(m.sample, m._regions[k]).latest
        if not sc or not os.path.exists(os.path.join(sc.folder, 'grid.json')):
            continue
        g = Grid(sc.folder)
        if not g.ready():
            continue
        P = mosaic_to_stage(sc.folder, [g.cr.point(0, 0), g.cr.point(1, 0), g.cr.point(0, 1)])
        e1, e2 = P[1] - P[0], P[2] - P[0]
        u1 = e1 / np.hypot(*e1)
        u2 = np.array([-u1[1], u1[0]])
        if np.dot(u2, e2) < 0:
            u2 = -u2
        return P[0], u1, u2, g, sc.folder
    return None


def export_sample(m, ks=None, out=None, dwg=True, verbose=True):
    """Один чертёж образца из участков ks (индексы m._regions; по умолчанию — снятые и не снятые с экспорта).
    Возвращает путь к DXF (и DWG рядом, если есть ODA File Converter)."""
    import ezdxf
    from .project import Selection
    from .samples import obj, num, title
    from .layers import rings
    from .snap import mosaic_to_stage, geometry
    from .util import Progress
    from .mask import dxf_to_dwg
    if ks is None:
        ks = [k for k, r in enumerate(m._regions) if r.get('kind') not in ('group', 'mask') and r.get('export', True)
              and Selection(m.sample, r).latest]
    ks = [k for k in ks if m._regions[k].get('kind') not in ('group',)]
    if not ks:
        raise ValueError("Nothing to export: no scanned area is chosen")
    out = out or os.path.join(m.sample.folder, 'export', time.strftime('%Y-%m-%d_%H%M%S'))
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    lat = _lattice(m, ks)
    if lat is not None:
        O, u1, u2, grid, lat_folder = lat
    else:
        O, u1, u2, grid, lat_folder = np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]), None, None
    D = lambda S: np.c_[(np.atleast_2d(S) - O) @ u1, -((np.atleast_2d(S) - O) @ u2)]   # столик -> чертёж (Y вверх)

    doc = ezdxf.new('R2010', setup=True)
    doc.units = 13
    doc.header['$INSUNITS'] = 13                                  # микрометры
    for name, color in (('CROSSES', 2), ('NUMBERS', 8), ('LABELS', 7), ('BONDS', 6)):
        doc.layers.add(name, color=color)
    msp = doc.modelspace()
    pr = Progress(len(ks) + 2, "Export: drawing", unit='steps') if verbose else None
    order = sorted(ks, key=lambda k: obj(m, m._regions[k]))     # мелкое увеличение снизу, крупное сверху
    placed = []
    for i, k in enumerate(order):
        r = m._regions[k]
        lay = f"S{num(m, k):02d}_{obj(m, r)}x"
        doc.layers.add(lay, color=(i % 6) + 1)
        sc = Selection(m.sample, r).latest
        jp = os.path.join(sc.folder, 'join', 'preview_all.jpg') if sc else None
        if pr:
            pr.update(1, f"{title(m, k)}: picture and outline")
        if jp and os.path.exists(jp):
            g = geometry(sc.folder)
            img = cv2.imread(jp)
            size = g['pos'].max(0) + (g['W'], g['H'])
            kk = min(1.0, 4000 / size.max())                       # масштаб превью (как в stitch.make_preview)
            a = (img.max(2) > 6).astype(np.uint8) * 255            # вне кадров — прозрачно
            png = os.path.join('images', f"{lay}.png")
            cv2.imwrite(os.path.join(out, png), np.dstack([img, a]))
            h_, w_ = img.shape[:2]
            P = D(mosaic_to_stage(sc.folder, np.array([[0, 0], [w_, 0], [0, h_]], float) / kk))
            ux, vy = (P[1] - P[0]) / w_, (P[2] - P[0]) / h_         # чертёж на пиксель: вправо, вниз
            idef = doc.add_image_def(filename=png, size_in_pixel=(w_, h_))
            im = msp.add_image(idef, insert=tuple(P[2]), size_in_units=(1, 1), dxfattribs={'layer': lay})
            im.dxf.u_pixel = (float(ux[0]), float(ux[1]), 0.0)     # точно по геометрии: поворот, масштаб
            im.dxf.v_pixel = (float(-vy[0]), float(-vy[1]), 0.0)   # вверх по картинке
            im.dxf.insert = (float(P[2][0]), float(P[2][1]), 0.0)  # левый нижний угол
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
        q = np.vstack([D(rg) for rg in rings(m, r)])
        x0, y1 = q[:, 0].min(), q[:, 1].max()
        hgt = max(2.0, 0.04 * max(np.ptp(q[:, 0]), np.ptp(q[:, 1])))
        msp.add_text(f"{title(m, k)}  {obj(m, r)}x", height=hgt, dxfattribs={'layer': 'LABELS', 'insert': (x0, y1 + hgt * 0.4)})
        placed.append(dict(selection=num(m, k), objective=obj(m, r), layer=lay, scan=sc.folder if sc else None))
    if grid is not None:                                          # кресты и номера квадратов
        if pr:
            pr.update(1, "crosses and square numbers")
        allq = np.vstack([D(rg) for k in ks for rg in rings(m, m._regions[k])])
        (x0, y0), (x1, y1) = allq.min(0), allq.max(0)
        pitch = np.hypot(*(mosaic_to_stage(lat_folder, [grid.cr.point(1, 0)]) -
                           mosaic_to_stage(lat_folder, [grid.cr.point(0, 0)]))[0])
        L = 0.1 * pitch
        i0, i1 = int(np.floor(x0 / pitch)) - 1, int(np.ceil(x1 / pitch)) + 1
        j0, j1 = int(np.floor(-y1 / pitch)) - 1, int(np.ceil(-y0 / pitch)) + 1
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                c = D(mosaic_to_stage(lat_folder, [grid.cr.point(i, j)]))[0]
                if not (x0 - pitch <= c[0] <= x1 + pitch and y0 - pitch <= c[1] <= y1 + pitch):
                    continue
                msp.add_line((c[0] - L, c[1]), (c[0] + L, c[1]), dxfattribs={'layer': 'CROSSES'})
                msp.add_line((c[0], c[1] - L), (c[0], c[1] + L), dxfattribs={'layer': 'CROSSES'})
                X, Y = grid.labels.get((i, j), grid.name(i, j))
                msp.add_text(f"{X},{Y}", height=0.06 * pitch,
                             dxfattribs={'layer': 'NUMBERS', 'insert': (c[0] + 0.05 * pitch, c[1] - 0.12 * pitch)})
    path = os.path.join(out, 'drawing.dxf')
    doc.saveas(path)
    json.dump(dict(frame='lattice' if grid is not None else 'stage', origin_um=O.tolist(), u1=u1.tolist(),
                   u2=u2.tolist(), areas=placed), open(os.path.join(out, 'drawing.json'), 'w'), indent=1)
    dw = dxf_to_dwg(path) if dwg else None
    if pr:
        pr.update(1, "DWG" if dw else "DXF (DWG: ODA File Converter not found)")
        pr.close()
    return dw or path
