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
        ks = [k for k, r in enumerate(m._regions) if r.get('kind') not in ('group', 'draft') and r.get('export', True)
              and Selection(m.sample, r).latest]
    ks = [k for k in ks if m._regions[k].get('kind') not in ('group',)]
    if not ks:
        raise ValueError("Nothing to export: no scanned area is chosen")
    out = out or os.path.join(m.sample.folder, 'export', time.strftime('%Y-%m-%d_%H%M%S'))
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    from . import lattice as LT
    from .layers import lattice_of
    L = lattice_of(m)
    if L is None or not L.get('numbered'):
        raise ValueError("The drawing needs the lattice with real numbers: in the viewer mark the crosses (C) and "
                         "number two squares in different rows and columns (N)")
    P = L['pitch']
    from .mask import Grid as _G
    try:
        lab_ = list(_G(L['src']).labels.values())
    except Exception:
        lab_ = []
    if any(not (1 <= X <= 58 and 1 <= Y <= 58) for X, Y in lab_) and verbose:
        print("  [!] Some square numbers are outside 1..58: check the numbering (N in the viewer)")
    D = lambda S: LT.to_uv(L, S) * (P, -P)                      # столик -> чертёж: крест 1,1 в (0, 0), Y вверх

    doc = ezdxf.new('R2010', setup=True)
    doc.units = 13
    doc.header['$INSUNITS'] = 13                                  # микрометры
    for name, color in (('CROSSES', 2), ('NUMBERS', 8), ('LABELS', 7), ('ADDRESSES', 8), ('BONDS', 6)):
        doc.layers.add(name, color=color)
    msp = doc.modelspace()
    pr = Progress(len(ks) + 2, "Export: drawing", unit='steps') if verbose else None
    order = sorted(ks, key=lambda k: obj(m, m._regions[k]))     # мелкое увеличение снизу, крупное сверху
    placed = []
    for i, k in enumerate(order):
        r = m._regions[k]
        lay = f"S{num(m, k):02d}_{obj(m, r)}x" + ("_draft" if r.get('kind') == 'draft' else "")
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
            C3 = D(mosaic_to_stage(sc.folder, np.array([[0, 0], [w_, 0], [0, h_]], float) / kk))
            ux, vy = (C3[1] - C3[0]) / w_, (C3[2] - C3[0]) / h_         # чертёж на пиксель: вправо, вниз
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
        if r.get('cells'):                                     # клетки — каждая целиком (общие грани — на каждом слое)
            N = LT.level(r)
            many = len(r['cells']) > 400
            for a, b in r['cells']:
                q = D(LT.cell_stage(L, a, b, N))
                msp.add_lwpolyline([tuple(p) for p in q], close=True, dxfattribs={'layer': lay})
                if not many:
                    msp.add_text(LT.address(a, b, r.get('factors') or []), height=0.08 * P / N,
                                 dxfattribs={'layer': 'ADDRESSES', 'insert': (q[0][0] + 0.05 * P / N, q[0][1] - 0.15 * P / N)})
        q = np.vstack([D(rg) for rg in rings(m, r)])
        x0, y1 = q[:, 0].min(), q[:, 1].max()
        hgt = max(2.0, 0.04 * max(np.ptp(q[:, 0]), np.ptp(q[:, 1])))
        msp.add_text(f"{title(m, k)}  {obj(m, r)}x", height=hgt, dxfattribs={'layer': 'LABELS', 'insert': (x0, y1 + hgt * 0.4)})
        placed.append(dict(selection=num(m, k), objective=obj(m, r), layer=lay, scan=sc.folder if sc else None))
    if pr:                                                       # кресты литографии с их номерами
        pr.update(1, "crosses and their numbers")
    allq = np.vstack([D(rg) for k in ks for rg in rings(m, m._regions[k])])
    (x0, y0), (x1, y1) = allq.min(0), allq.max(0)
    arm = 0.1 * P
    for X in range(max(1, int(np.floor(x0 / P)) + 1), int(np.ceil(x1 / P)) + 2):
        for Y in range(max(1, int(np.floor(-y1 / P)) + 1), int(np.ceil(-y0 / P)) + 2):
            cx, cy = (X - 1) * P, -(Y - 1) * P
            msp.add_line((cx - arm, cy), (cx + arm, cy), dxfattribs={'layer': 'CROSSES'})
            msp.add_line((cx, cy - arm), (cx, cy + arm), dxfattribs={'layer': 'CROSSES'})
            msp.add_text(f"{X},{Y}", height=0.06 * P, dxfattribs={'layer': 'NUMBERS', 'insert': (cx + 0.04 * P, cy + 0.04 * P)})
    path = os.path.join(out, 'drawing.dxf')
    doc.saveas(path)
    json.dump(dict(frame="lattice: cross 1,1 at (0, 0), X to cross 2,1, Y up (to cross 1,2 is -Y)", pitch_um=P,
                   lattice=L, areas=placed), open(os.path.join(out, 'drawing.json'), 'w'), indent=1, default=float)
    dw = dxf_to_dwg(path) if dwg else None
    if pr:
        pr.update(1, "DWG" if dw else "DXF (DWG: ODA File Converter not found)")
        pr.close()
    return dw or path
