"""Selection — слой (как в фотошопе): контур из прямоугольников в настоящих мкм столика, дерево, операции над
множествами. Одна сущность на всё: большая карта 10x, участки 50x внутри неё, участки 100x внутри тех.

    Контур   reg['shape'] — многоугольники [{'ext': [[x, y], ...], 'holes': [[[x, y], ...], ...]}, ...];
             без него — прямоугольник углов A, B (центры крайних кадров) плюс полкадра.
    Дерево   reg['parent'] — uid выделения, в котором оно лежит (None — верхний уровень); группа
             (reg['kind'] == 'group') — папка без своего контура и съёмки: контур — объединение детей.
    Кисть    выделение, рождённое кистью по квадратам решётки в просмотре: reg['src'] — папка съёмки (от образца),
             reg['cells'] — квадраты. Пока его не меняли операциями, оно следует за кистью (sync_brush).

Операции: union (слить в первое), subtract (из первого вычесть остальные), intersect, group, ungroup, move.
"""
import os
import json
import uuid
import numpy as np
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
from shapely import affinity

EPS = 0.01                                                   # мкм: точность контура


# ---------------------------------------------------------------- контур
def from_json(sh):
    polys = [Polygon(p['ext'], p.get('holes') or []) for p in sh or []]
    polys = [p.buffer(0) for p in polys if p.area > 0]
    return unary_union(polys) if polys else Polygon()


def to_json(g):
    g = g.buffer(0)
    ps = [g] if isinstance(g, Polygon) else list(getattr(g, 'geoms', []))
    out = []
    for p in ps:
        if p.is_empty or not isinstance(p, Polygon):
            continue
        r = lambda ring: [[round(x, 3), round(y, 3)] for x, y in list(ring.coords)[:-1]]
        out.append(dict(ext=r(p.exterior), holes=[r(h) for h in p.interiors]))
    return out


def is_group(reg):
    return reg.get('kind') == 'group'


def frame(reg, fw=None, fh=None):
    return reg.get('fw', fw or 1.0), reg.get('fh', fh or 1.0)


def geom(m, reg, _depth=0):
    """Контур выделения (shapely), настоящие мкм."""
    if is_group(reg):
        if _depth > 20:
            return Polygon()
        kids = [r for r in m._regions if r.get('parent') == reg.get('uid')]
        return unary_union([geom(m, r, _depth + 1) for r in kids]) if kids else Polygon()
    if reg.get('shape'):
        return from_json(reg['shape'])
    if 'B' not in reg:
        return Polygon()
    fw, fh = frame(reg, *m.frame_size)
    (ax, ay), (bx, by) = reg['A'], reg['B']
    return box(min(ax, bx) - fw / 2, min(ay, by) - fh / 2, max(ax, bx) + fw / 2, max(ay, by) + fh / 2)


def set_geom(m, reg, g):
    """Записать контур; углы A, B (центры крайних кадров) — по охватывающему прямоугольнику."""
    g = g.buffer(0).simplify(EPS)                            # без лишних точек на прямых сторонах
    reg['shape'] = to_json(g)
    fw, fh = frame(reg, *m.frame_size)
    x0, y0, x1, y1 = g.bounds
    ax, bx = (x0 + fw / 2, x1 - fw / 2) if x1 - x0 > fw else ((x0 + x1) / 2,) * 2
    ay, by = (y0 + fh / 2, y1 - fh / 2) if y1 - y0 > fh else ((y0 + y1) / 2,) * 2
    reg['A'], reg['B'] = (ax, ay), (bx, by)


def rings(m, reg):
    """Контур для рисования: [кольцо Nx2] (наружные и дыры)."""
    g = geom(m, reg)
    ps = [g] if isinstance(g, Polygon) else list(getattr(g, 'geoms', []))
    out = []
    for p in ps:
        if isinstance(p, Polygon) and not p.is_empty:
            out.append(np.array(p.exterior.coords)[:-1])
            out += [np.array(h.coords)[:-1] for h in p.interiors]
    return out


def bounds(m, reg):
    g = geom(m, reg)
    return g.bounds if not g.is_empty else (0, 0, 0, 0)


def contains(m, reg, x, y):
    from shapely.geometry import Point
    return geom(m, reg).buffer(EPS).contains(Point(x, y))


def touches_frame(m, reg, x, y, fw, fh):
    """Кадр с центром (x, y) задевает контур (не только краем)?"""
    return geom(m, reg).intersection(box(x - fw / 2, y - fh / 2, x + fw / 2, y + fh / 2)).area > 1e-3 * fw * fh


def refresh_groups(m):
    """Углы групп (A, B) — по охватывающему прямоугольнику детей (для мест, где нужен прямоугольник)."""
    for gr in m._regions:
        if is_group(gr):
            g = geom(m, gr)
            x0, y0, x1, y1 = g.bounds if not g.is_empty else (0, 0, 0, 0)
            gr.update(A=(x0, y0), B=(x1, y1), fw=0.0, fh=0.0)


# ---------------------------------------------------------------- дерево
def idx(m, uid):
    return next((i for i, r in enumerate(m._regions) if r.get('uid') == uid), None)


def new_num(m):
    m._area_seq = getattr(m, '_area_seq', 0) + 1
    return m._area_seq


def geometric_parent(m, reg, skip=None):
    """Выделение меньшего увеличения, внутри которого лежит центр reg (ближайший слой, при равных — меньшее)."""
    from .samples import obj
    g = geom(m, reg)
    if g.is_empty:
        return None
    c = g.representative_point()
    best = None
    for i, o in enumerate(m._regions):
        if o is reg or i == skip or is_group(o) or obj(m, o) >= obj(m, reg):
            continue
        go = geom(m, o)
        if go.buffer(EPS).contains(c):
            key = (-obj(m, o), go.area)
            if best is None or key < best[0]:
                best = (key, i)
    return best[1] if best else None


def detach(m, reg):
    """Выделение кисти меняют операциями — дальше оно живёт само: его квадраты кисть больше не трогает."""
    if reg.get('src'):
        _ignore_cells(m, reg['src'], reg.get('cells') or [])
    reg.pop('src', None)
    reg.pop('cells', None)


# ---------------------------------------------------------------- операции
def _pick(m, ks):
    ks = [k for k in dict.fromkeys(ks) if 0 <= k < len(m._regions)]
    if not ks:
        raise ValueError("Choose selections first (Ctrl+click their cards)")
    return ks


def _drop(m, ks):
    """Удалить выделения ks (их дети переходят к их родителю)."""
    for k in sorted(ks, reverse=True):
        r = m._regions[k]
        for c in m._regions:
            if c.get('parent') == r.get('uid'):
                c['parent'] = r.get('parent')
        m._regions.pop(k)


def union(m, ks):
    """Слить выделения в первое (контур — объединение; дети и точки фокуса — к нему)."""
    ks = _pick(m, ks)
    if len(ks) < 2:
        raise ValueError("Merge needs two or more selections")
    t = m._regions[ks[0]]
    if is_group(t):
        raise ValueError("Groups are not merged: ungroup first")
    g = unary_union([geom(m, m._regions[k]) for k in ks])
    for k in ks:
        detach(m, m._regions[k])
    for k in ks[1:]:
        r = m._regions[k]
        t['pts'] = (t.get('pts') or []) + (r.get('pts') or [])
        for c in m._regions:
            if c.get('parent') == r.get('uid'):
                c['parent'] = t['uid']
    set_geom(m, t, g)
    nums = [m._regions[k]['num'] for k in ks]
    _drop(m, ks[1:])
    m._sel = idx(m, t['uid'])
    return f"Selections {', '.join(map(str, nums))} merged into Selection {t['num']}"


def subtract(m, ks):
    """Из первого выделения вычесть остальные (они остаются)."""
    ks = _pick(m, ks)
    if len(ks) < 2:
        raise ValueError("Subtract needs two or more selections: the first minus the others")
    t = m._regions[ks[0]]
    if is_group(t):
        raise ValueError("Nothing to subtract from a group: choose a selection first")
    g = geom(m, t).difference(unary_union([geom(m, m._regions[k]) for k in ks[1:]]))
    if g.area < 1e-3:
        raise ValueError("Nothing would be left: not subtracted")
    detach(m, t)
    set_geom(m, t, g)
    return f"Selection {t['num']} minus {', '.join(str(m._regions[k]['num']) for k in ks[1:])}"


def intersect(m, ks):
    """Первое выделение = пересечение всех (остальные остаются)."""
    ks = _pick(m, ks)
    if len(ks) < 2:
        raise ValueError("Intersect needs two or more selections")
    t = m._regions[ks[0]]
    g = geom(m, t)
    for k in ks[1:]:
        g = g.intersection(geom(m, m._regions[k]))
    if g.area < 1e-3:
        raise ValueError("They do not overlap: nothing changed")
    detach(m, t)
    set_geom(m, t, g)
    return f"Selection {t['num']} = its overlap with {', '.join(str(m._regions[k]['num']) for k in ks[1:])}"


def group(m, ks):
    """Сложить выделения в группу (папку). Группа встаёт туда, где был первый."""
    ks = _pick(m, ks)
    first = m._regions[ks[0]]
    from .samples import obj
    gr = dict(kind='group', uid=uuid.uuid4().hex, num=new_num(m), parent=first.get('parent'),
              obj=min(obj(m, m._regions[k]) for k in ks), pts=[], thumbs=[])
    if first.get('rank') is not None:
        gr['rank'] = first['rank']
    for k in ks:
        m._regions[k]['parent'] = gr['uid']
    m._regions.append(gr)
    refresh_groups(m)
    m._sel = len(m._regions) - 1
    return f"Group {gr['num']}: {len(ks)} selection{'s' if len(ks) > 1 else ''}"


def ungroup(m, ks):
    """Разгруппировать: дети группы переходят на её место, группа исчезает."""
    ks = [k for k in _pick(m, ks) if is_group(m._regions[k])]
    if not ks:
        raise ValueError("Choose a group to ungroup")
    nums = [m._regions[k]['num'] for k in ks]
    _drop(m, ks)
    m._sel = None
    return f"Group {', '.join(map(str, nums))} ungrouped"


def move(m, k, dx, dy):
    """Сдвинуть контур выделения (мкм)."""
    r = m._regions[k]
    if is_group(r):
        for c in [i for i, c in enumerate(m._regions) if c.get('parent') == r['uid']]:
            move(m, c, dx, dy)
        return
    g = affinity.translate(geom(m, r), dx, dy)
    detach(m, r)
    set_geom(m, r, g)
    r['pts'] = [dict(p, x=p['x'] + dx, y=p['y'] + dy) for p in r.get('pts') or []]


# ---------------------------------------------------------------- кисть по квадратам -> выделения
def _figures(cells):
    """Квадраты -> замкнутые фигуры (соседи по стороне)."""
    left, out = set(map(tuple, cells)), []
    while left:
        st, comp = [left.pop()], []
        while st:
            c = st.pop()
            comp.append(c)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if n in left:
                    left.remove(n)
                    st.append(n)
        out.append(sorted(comp))
    return sorted(out, key=lambda f: (-len(f), f[0]))


def _grid_path(m, src):
    return os.path.join(m.sample.folder, src, 'grid.json')


def _ignore_cells(m, src, cells):
    p = _grid_path(m, src)
    try:
        d = json.load(open(p, encoding='utf-8'))
    except (OSError, ValueError):
        return
    ign = {tuple(c) for c in d.get('brush_detached', [])} | {tuple(c) for c in cells}
    d['brush_detached'] = sorted([list(c) for c in ign])
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1)


def next_objective(o):
    from .config import OBJECTIVES
    obs = sorted(OBJECTIVES)
    up = [x for x in obs if x > o]
    return up[0] if up else o


def sync_brush(m):
    """Квадраты кисти последних съёмок -> выделения: каждая замкнутая фигура — своё выделение, ребёнок того,
    чью съёмку разметили, на следующий объектив. Возвращает сообщение или None (ничего не изменилось)."""
    from .project import Selection
    from .selections import marked_squares
    from .samples import obj
    if getattr(m, '_sample', None) is None:
        return None
    changed = []
    for par in [r for r in m._regions if not is_group(r)]:        # и выделения кисти: снятые — тоже родители
        sc = Selection(m.sample, par).latest
        if not sc:
            continue
        mk = marked_squares(sc.folder)
        src = os.path.relpath(sc.folder, m.sample.folder).replace('\\', '/')
        try:
            ign = {tuple(c) for c in json.load(open(os.path.join(sc.folder, 'grid.json'), encoding='utf-8'))
                   .get('brush_detached', [])}
        except (OSError, ValueError):
            ign = set()
        quads = dict(zip(mk['cells'], mk['um'])) if mk else {}
        figs = _figures([c for c in quads if c not in ign])
        mine = [r for r in m._regions if r.get('src') == src]
        used = set()
        for f in figs:
            fs = set(f)
            best = max(((len(fs & {tuple(c) for c in r.get('cells') or []}), i) for i, r in enumerate(mine)
                        if i not in used), default=(0, None))
            g = unary_union([Polygon(quads[c]) for c in f]).buffer(0)
            if best[0] > 0:
                r = mine[best[1]]
                used.add(best[1])
                if sorted(map(tuple, r.get('cells') or [])) != f:
                    r['cells'] = [list(c) for c in f]
                    set_geom(m, r, g)
                    changed.append(f"Selection {r['num']} follows the brush")
            else:
                o = next_objective(obj(m, par))
                fw0, fh0 = m.frame_size
                k_ = m.objective / o
                r = dict(uid=uuid.uuid4().hex, num=new_num(m), obj=o, fw=fw0 * k_, fh=fh0 * k_, pts=[], thumbs=[],
                         parent=par['uid'], src=src, cells=[list(c) for c in f])
                set_geom(m, r, g)
                m._regions.append(r)
                changed.append(f"Selection {r['num']} ({o}x) from the brush")
        for i, r in enumerate(mine):
            if i not in used:
                m._regions.remove(r)
                changed.append(f"Selection {r['num']} gone with its squares")
    return "; ".join(changed) if changed else None


def delete_brush_cells(m, reg):
    """Выделение кисти удаляют: его квадраты снимаются с выбора в съёмке (снятые — в 'selected_removed')."""
    if not reg.get('src'):
        return
    p = _grid_path(m, reg['src'])
    try:
        d = json.load(open(p, encoding='utf-8'))
    except (OSError, ValueError):
        return
    gone = {tuple(c) for c in reg.get('cells') or []}
    d['selected_removed'] = d.get('selected_removed', []) + [list(c) for c in sorted(gone)]
    d['selected'] = [c for c in d.get('selected', []) if tuple(c) not in gone]
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1)
