"""Selection — слой (как в фотошопе): контур из прямоугольников в настоящих мкм столика, дерево, операции над
множествами. Одна сущность на всё: большая карта 10x, участки 50x внутри неё, участки 100x внутри тех.

    Контур   reg['shape'] — многоугольники [{'ext': [[x, y], ...], 'holes': [[[x, y], ...], ...]}, ...];
             без него — прямоугольник углов A, B (центры крайних кадров) плюс полкадра.
    Дерево   reg['parent'] — uid выделения, в котором оно лежит (None — верхний уровень); группа
             (reg['kind'] == 'group') — папка без своего контура и съёмки: контур — объединение детей.
    Клетки   выделение — набор клеток вложенной сетки (lattice.py): reg['cells'] — клетки уровня
             lattice.level(reg), reg['factors'] — цепочка подсеток, reg['sub'] — подсетка для его детей (2…5).
    Draft    черновик (reg['kind'] == 'draft'): выделение без объектива, кистью в просмотре; не снимается.
             Пока его не меняли операциями (reg['src'] — откуда кисть), оно следует за кистью (sync_brush).

Операции: union (слить в первое), subtract (из первого вычесть остальные), intersect, group, ungroup, move.
"""
import os
import json
import time
import uuid
import numpy as np
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
from shapely import affinity

EPS = 0.01                                                   # мкм: точность контура


# ---------------------------------------------------------------- контур
_GCACHE = {}                                                 # id(список контура) -> (список, геометрия)


def from_json(sh):
    """Контур из sample.json -> shapely. Кэш по самому списку: set_geom всегда кладёт новый список, поэтому
    тот же объект — та же геометрия (окно рисует десятки раз в секунду, разбор каждый раз — дорого)."""
    c = _GCACHE.get(id(sh))
    if c is not None and c[0] is sh:
        return c[1]
    g = _from_json(sh)
    if len(_GCACHE) > 4000:
        _GCACHE.clear()
    _GCACHE[id(sh)] = (sh, g)
    return g


def _from_json(sh):
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


def is_mask(reg):
    """Draft — черновик с кисти: показывает, где искать; не снимается, пока не выбран объектив."""
    return reg.get('kind') == 'draft'


is_draft = is_mask


def to_area(m, k, o, dry=False):
    """Черновик — в обычный участок на объективе o (чтобы снять его). dry — без записи кисти (проверка на копии)."""
    r = m._regions[k]
    k_ = m.objective / o
    if not dry:
        detach(m, r)
    r.pop('kind', None)
    g = geom(m, r)
    r.update(obj=o, fw=m.frame_size[0] * k_, fh=m.frame_size[1] * k_)
    set_geom(m, r, g)


def draft_to_selection(m, k, o):
    """Черновик k -> выделение под объектив o (снимается и идёт в экспорт); обратного хода нет.
    Пирамидка проверяется до изменений: объектив больше, чем у снятого сверху, меньше, чем у выделений внутри,
    внутри родителя, без наезда на выделение того же объектива. -> (получилось ли, сообщение)."""
    r = m._regions[k]
    if not is_mask(r):
        return False, f"Selection {r['num']} is already a selection"
    bad = _try(m, lambda t_: (to_area(t_, k, o, dry=True), k)[1])
    if bad:
        return False, bad + (": put a larger objective (M) first" if "larger objective" in bad else "")
    to_area(m, k, o)
    return True, f"Draft {r['num']} is now Selection {r['num']} ({o}x): it will be scanned and goes into the export"


def next_objective_of(m, k, objectives):
    """Не снятое выделение k — на следующий объектив из списка, при котором пирамидка цела. -> (вышло, сообщение)."""
    from .samples import obj
    r = m._regions[k]
    cur = obj(m, r)
    i0 = objectives.index(cur) if cur in objectives else -1

    def set_o(mm, kk, o_):
        r2 = mm._regions[kk]
        g2 = geom(mm, r2)
        r2.update(obj=o_, fw=m.frame_size[0] * m.objective / o_, fh=m.frame_size[1] * m.objective / o_)
        if r2.get('shape'):                                     # углы A, B — по кадру нового объектива
            set_geom(mm, r2, g2)
        return kk
    for j in range(1, len(objectives) + 1):
        o_ = objectives[(i0 + j) % len(objectives)]
        if o_ == cur:
            break
        if not _try(m, lambda t_: set_o(t_, k, o_)):
            set_o(m, k, o_)
            return True, f"Selection {r['num']}: {o_}x (click the objective again for the next one)"
    return False, (f"Selection {r['num']}: no other objective fits here (it must be larger than the area it lies in, "
                   "smaller than the areas inside it and not overlap the same objective)")


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
    """Где лежит reg: самое маленькое выделение меньшего увеличения или маска, содержащая его середину."""
    from .samples import obj
    g = geom(m, reg)
    if g.is_empty:
        return None
    c = g.representative_point()
    best = None
    for i, o in enumerate(m._regions):
        if o is reg or i == skip or is_group(o) or (not is_mask(o) and obj(m, o) >= obj(m, reg)):
            continue
        if is_mask(o) and is_mask(reg):
            continue
        go = geom(m, o)
        if go.buffer(EPS).contains(c) and (best is None or go.area < best[0]):
            best = (go.area, i)
    return best[1] if best else None


def clip_to_parent(m, reg, pk=None):
    """Пирамидка: участок внутри большего не вылезает за его край — обрезается по нему (клетки — только целые
    внутри). pk — индекс родителя (None — по месту). Возвращает (родитель или None, обрезан ли).
    ValueError, если внутри родителя ничего не остаётся."""
    from . import lattice as LT
    if pk is None:
        pk = geometric_parent(m, reg)
    if pk is None:                                           # середина снаружи: тот, с кем больше общего
        from .samples import obj
        g0 = geom(m, reg)
        best = max(((geom(m, o).intersection(g0).area, i) for i, o in enumerate(m._regions)
                    if o is not reg and not is_group(o) and not is_mask(o) and obj(m, o) < obj(m, reg)),
                   default=(0, None))
        pk = best[1] if best[0] > 1e-6 else None
    if pk is None or is_group(m._regions[pk]):
        return pk, False
    gp = geom(m, m._regions[pk])
    g = geom(m, reg)
    if gp.is_empty or gp.buffer(EPS).contains(g):
        return pk, False
    L = lattice_of(m)
    if reg.get('cells') and L:
        N = LT.level(reg)
        tol = gp.buffer(0.02 * L['pitch'] / N)
        keep = [c for c in reg['cells'] if tol.contains(LT.cells_geom(L, [tuple(c)], N))]
        if not keep:
            raise ValueError("Nothing of it lies inside the larger area: draw inside it")
        reg['cells'] = [list(c) for c in keep]
        set_geom(m, reg, LT.cells_geom(L, [tuple(c) for c in keep], N))
    else:
        g2 = g.intersection(gp)
        if g2.area < 1e-3 * max(g.area, 1e-9):
            raise ValueError("Nothing of it lies inside the larger area: draw inside it")
        set_geom(m, reg, g2)
    return pk, True


def detach(m, reg):
    """Выделение кисти меняют операциями — дальше оно живёт само: его квадраты кисть больше не трогает."""
    if reg.get('src'):
        remove_brush_cells(m, reg)
    reg.pop('src', None)


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
    from .samples import obj
    objs = {obj(m, m._regions[k]) for k in ks if not is_mask(m._regions[k])}
    if len(objs) > 1:
        raise ValueError("Merge selections of one objective (" + ", ".join(f"{o}x" for o in sorted(objs)) +
                         " are different layers): drafts may join any")
    g = unary_union([geom(m, m._regions[k]) for k in ks])
    bad = _try(m, lambda t_: _merged_in(t_, ks, g))              # проверка на копии — до всех изменений
    if bad:
        raise ValueError(bad + ": not merged")
    cells_op(m, [m._regions[k] for k in ks], lambda a, *o: a.union(*o))
    for k in ks:
        detach(m, m._regions[k])
    for k in ks[1:]:
        r = m._regions[k]
        t['pts'] = (t.get('pts') or []) + (r.get('pts') or [])
        for c in m._regions:
            if c.get('parent') == r.get('uid'):
                c['parent'] = t['uid']
    real = [m._regions[k] for k in ks if not is_mask(m._regions[k])]
    if is_mask(t) and real:                                  # черновик + настоящий участок = настоящий участок
        t.pop('kind', None)
        t.update(obj=real[0].get('obj', m.objective), fw=real[0].get('fw'), fh=real[0].get('fh'))
    _geom_or_cells(m, t, g)
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
    cells_op(m, [m._regions[k] for k in ks], lambda a, *o: a.difference(*o))
    detach(m, t)
    _geom_or_cells(m, t, g)
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
    cells_op(m, [m._regions[k] for k in ks], lambda a, *o: a.intersection(*o))
    detach(m, t)
    _geom_or_cells(m, t, g)
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


def move(m, k, dx, dy, _top=True):
    """Сдвинуть контур выделения (мкм) — вместе со всем, что внутри него (пирамидка не рассыпается).
    Нельзя вылезти за родителя и наехать на выделение того же объектива: тогда ValueError, ничего не сдвинуто."""
    import copy as _cp
    if _top:                                                     # сначала проверить на копии: ничего не трогая
        class _Try:
            pass
        t_ = _Try()
        t_._regions, t_.frame_size, t_.objective = _cp.deepcopy(m._regions), m.frame_size, m.objective
        t_._sample = None

        def shift(i):
            rr = t_._regions[i]
            for j in [j for j, c in enumerate(t_._regions) if c.get('parent') == rr['uid']]:
                shift(j)
            if not is_group(rr):
                rr['shape'] = to_json(affinity.translate(geom(t_, rr), dx, dy))
        shift(k)
        bad = conflict(t_, t_._regions[k])
        if bad:
            raise ValueError(bad)
    r = m._regions[k]
    for c in [i for i, c in enumerate(m._regions) if c.get('parent') == r['uid']]:
        move(m, c, dx, dy, _top=False)
    if not is_group(r):
        g = affinity.translate(geom(m, r), dx, dy)
        detach(m, r)
        r.pop('cells', None)                                    # сдвинутое — уже не по сетке
        r.pop('factors', None)
        set_geom(m, r, g)
        r['pts'] = [dict(p, x=p['x'] + dx, y=p['y'] + dy) for p in r.get('pts') or []]


def _try(m, fn):
    """Сделать fn(копия) на копии выделений и вернуть conflict() того, что fn вернул (индекс), — без изменений."""
    import copy as _cp

    class _Try:
        pass
    t_ = _Try()
    t_._regions, t_.frame_size, t_.objective, t_._sample = _cp.deepcopy(m._regions), m.frame_size, m.objective, None
    k = fn(t_)
    return conflict(t_, t_._regions[k]) if k is not None else None


def _merged_in(t_, ks, g):
    tg = t_._regions[ks[0]]
    real = [t_._regions[k] for k in ks if not is_mask(t_._regions[k])]
    if is_mask(tg) and real:
        tg.pop('kind', None)
        tg.update(obj=real[0].get('obj'), fw=real[0].get('fw'), fh=real[0].get('fh'))
    tg['shape'] = to_json(g)
    for k in sorted(ks[1:], reverse=True):
        for c in t_._regions:
            if c.get('parent') == t_._regions[k].get('uid'):
                c['parent'] = tg['uid']
        t_._regions.pop(k)
    return t_._regions.index(tg)


def conflict(m, reg):
    """Какое правило нарушает выделение (сообщение) или None. Правила пирамидки:
    1) лежит внутри родителя; 2) объектив больше, чем у снятого выделения сверху, и меньше, чем у выделений
    внутри; 3) не наезжает на другое выделение того же объектива (черновики и группы не в счёт)."""
    from .samples import obj
    if is_group(reg):
        return None
    g = geom(m, reg)
    name = f"{'Draft' if is_mask(reg) else 'Selection'} {reg.get('num')}"
    pk = idx(m, reg.get('parent')) if reg.get('parent') else None
    if pk is not None and not is_group(m._regions[pk]):
        par = m._regions[pk]
        if not geom(m, par).buffer(max(EPS, 0.002 * max(1.0, (g.area ** 0.5)))).contains(g):
            return f"{name} would go beyond {'Draft' if is_mask(par) else 'Selection'} {par['num']} it lies in"
    if not is_mask(reg):
        up = m._regions[pk] if pk is not None else None
        while up is not None and (is_mask(up) or is_group(up)):
            q = idx(m, up.get('parent')) if up.get('parent') else None
            up = m._regions[q] if q is not None else None
        if up is not None and obj(m, up) >= obj(m, reg):
            return f"{name} ({obj(m, reg)}x) lies in Selection {up['num']} ({obj(m, up)}x): it needs a larger objective"
        todo, seen = [reg.get('uid')], set()                    # выделения внутри — только крупнее
        while todo:
            u = todo.pop()
            for c in m._regions:
                if c.get('parent') == u and c.get('uid') not in seen:
                    seen.add(c.get('uid'))
                    if is_mask(c) or is_group(c):
                        todo.append(c.get('uid'))
                    elif obj(m, c) <= obj(m, reg):
                        return (f"{name} ({obj(m, reg)}x) holds Selection {c['num']} ({obj(m, c)}x): "
                                "the areas inside need a larger objective")
        for c in m._regions:
            if c is not reg and not is_mask(c) and not is_group(c) and obj(m, c) == obj(m, reg) and \
                    geom(m, c).intersection(g).area > 1e-3 * max(g.area, 1e-9):
                return f"[E704] {name} would overlap Selection {c['num']} of the same objective ({obj(m, reg)}x)"
        if not g.is_empty:                                      # не внутри выделения крупнее себя
            pt = g.representative_point()
            for c in m._regions:
                if c is not reg and not is_mask(c) and not is_group(c) and obj(m, c) > obj(m, reg) and \
                        geom(m, c).buffer(EPS).contains(pt) and g.area < geom(m, c).area:
                    return (f"{name} ({obj(m, reg)}x) would lie inside Selection {c['num']} ({obj(m, c)}x): "
                            "a smaller area needs a larger objective")
    return None


# ---------------------------------------------------------------- клетки
def cells_op(m, regs, fn):
    """Операция над клетками: если у всех выделений клетки одного уровня — результат клетками (первому)."""
    t = regs[0]
    lv = [tuple(r.get('factors') or []) for r in regs]
    if all(r.get('cells') for r in regs) and len(set(lv)) == 1:
        sets = [set(map(tuple, r['cells'])) for r in regs]
        t['_cells_new'] = sorted(fn(sets[0], *sets[1:]))
    else:
        t['_cells_new'] = None


def _geom_or_cells(m, t, g):
    from . import lattice as LT
    new = t.pop('_cells_new', None)
    L = lattice_of(m)
    if new and L:
        t['cells'] = [list(c) for c in new]
        set_geom(m, t, LT.cells_geom(L, new, LT.level(t)))
    else:
        t.pop('cells', None)
        set_geom(m, t, g)


def lattice_of(m):
    """Решётка образца (sample.json 'lattice') или None."""
    sm = getattr(m, '_sample', None)
    return sm.data.get('lattice') if sm is not None else None


def next_objective(o):
    from .config import OBJECTIVES
    obs = sorted(OBJECTIVES)
    up = [x for x in obs if x > o]
    return up[0] if up else o


def _latest_grids(m):
    """[(выделение, папка съёмки, grid.json)]: у каждого выделения — та его съёмка, где сетку правили последней
    (обычно последняя; открыли в просмотре старую и выбрали там — она). Только съёмки, где сетка есть."""
    from .project import Selection
    out = []
    for r in m._regions:
        if is_group(r) or is_mask(r):
            continue
        try:
            scs = Selection(m.sample, r).scans
        except Exception:
            scs = []
        best = None
        for sc in scs:                                           # старые -> новые: при равенстве — новее
            gp = os.path.join(sc.folder, 'grid.json')
            if not os.path.exists(gp):
                continue
            try:
                t_ = float(json.load(open(gp, encoding='utf-8')).get('brush_t') or 0)
            except (OSError, ValueError):
                continue
            if best is None or t_ >= best[0]:
                best = (t_, sc.folder, gp)
        if best is not None:
            try:
                out.append((r, best[1], json.load(open(best[2], encoding='utf-8'))))
            except (OSError, ValueError):
                pass
    return out


def update_lattice(m):
    """Решётка образца — по размеченной съёмке (подписи номеров предпочтительнее). True, если изменилась."""
    from . import lattice as LT
    from .mask import Grid
    best = None
    for r, f, d in _latest_grids(m):
        try:
            g = Grid(f)
        except Exception:
            continue
        if not g.ready() or getattr(g, 'seeded', False) or d.get('seeded'):
            continue
        L = LT.from_grid(f, g)
        if L and (best is None or (L['numbered'], -obj_of(m, r)) > (best[0]['numbered'], -best[1])):
            best = (L, obj_of(m, r))
    if best is None:
        return False
    old = m.sample.data.get('lattice')
    if old and old.get('numbered') and not best[0]['numbered']:
        return False
    if old is None or any(not np.allclose(old[k], best[0][k], atol=1e-3) for k in ('O', 'e1', 'e2')) \
            or old.get('numbered') != best[0]['numbered']:
        m.sample.data['lattice'] = best[0]
        m.sample.save()
        return True
    return False


def obj_of(m, r):
    from .samples import obj
    return obj(m, r)


def sync_brush(m):
    """Кисть в просмотре -> черновики (Draft): каждая замкнутая фигура клеток — свой черновик.
    Уровень 1 (квадраты литографии, grid.json 'selected') — дети снятого выделения; подсетка выделения
    (grid.json 'refine': {uid: {'n': 2…5, 'cells': [[a, b], ...]}}) — его дети на уровне N·n (строго внутри него).
    Заодно — решётка образца по разметке. Возвращает сообщение или None."""
    from . import lattice as LT
    from .samples import obj
    from .snap import mosaic_to_stage
    from .mask import Grid
    if getattr(m, '_sample', None) is None:
        return None
    changed = ["Lattice of the sample updated"] if update_lattice(m) else []
    L = lattice_of(m)
    if L is None:
        return "; ".join(changed) or None
    want = {}                                                    # (uid родителя, уровень) -> множество клеток
    seen1, seenR = set(), set()                                  # чья кисть известна: уровень 1 / подсетки
    for par, f, d in _latest_grids(m):
        seen1.add(par['uid'])
        seenR.update((d.get('refine') or {}).keys())
        if d.get('selected'):                                    # квадраты литографии
            g = Grid(f)
            if g.ready():
                q = np.array([g.cr.point(i + 0.5, j + 0.5) for i, j in g.selected])
                if len(q):
                    uv = LT.to_uv(L, mosaic_to_stage(f, q))
                    pg = geom(m, par)
                    from shapely.geometry import Point
                    S_ = mosaic_to_stage(f, q)
                    cs = {(int(np.floor(u)), int(np.floor(v))) for (u, v), S1 in zip(uv, S_)   # середина внутри;
                          if pg.buffer(EPS).contains(Point(*S1))}   # контур черновика — по краю снятого (пирамидка)
                    want.setdefault((par['uid'], ()), set()).update(cs)
        for uid, rq in (d.get('refine') or {}).items():          # подсетки выделений
            k = idx(m, uid)
            if k is None:
                continue
            r = m._regions[k]
            n = int(rq.get('n') or 1)
            if n > 1:
                r['sub'] = n
            elif r.get('sub'):
                r.pop('sub')
            if n > 1 and rq.get('cells'):
                fac = tuple((r.get('factors') or []) + [n])
                inside = {tuple(c) for c in r.get('cells') or []}
                cs = {tuple(c) for c in rq['cells'] if not inside or (c[0] // n, c[1] // n) in inside}
                from shapely.geometry import Point
                rg_ = geom(m, r).buffer(EPS)                      # середина клетки — в родителе (как у квадратов)
                Nn = (int(np.prod(r.get('factors') or [])) if r.get('factors') else 1) * n
                cs = {c for c in cs if rg_.contains(Point(*LT.to_stage(L, [((c[0] + 0.5) / Nn, (c[1] + 0.5) / Nn)])[0]))}
                want.setdefault((uid, fac), set()).update(cs)
    for (puid, fac), cells in sorted(want.items(), key=lambda t: len(t[0][1])):   # сначала мельче уровень
        if idx(m, puid) is None:                                 # родитель ушёл в этом же проходе
            continue
        par = m._regions[idx(m, puid)]
        if fac and par.get('cells'):                             # родитель мог измениться в этом же проходе
            from shapely.geometry import Point
            n_, pc_ = fac[-1], {tuple(c) for c in par['cells']}
            Nn_, pgb_ = int(np.prod(fac)), geom(m, par).buffer(EPS)
            cells = {c for c in cells if (c[0] // n_, c[1] // n_) in pc_ and
                     pgb_.contains(Point(*LT.to_stage(L, [((c[0] + 0.5) / Nn_, (c[1] + 0.5) / Nn_)])[0]))}
        src = f"{puid}@{'x'.join(map(str, fac)) or '1'}"
        mine = [r for r in m._regions if r.get('src') == src]
        used = set()
        N = int(np.prod(fac)) if fac else 1
        pgeo = geom(m, par)

        def shape_of(fg_):                                       # клетки, обрезанные по краю родителя
            g_ = LT.cells_geom(L, fg_, N)
            return g_.intersection(pgeo) if not pgeo.is_empty and not pgeo.buffer(EPS).contains(g_) else g_
        for fg in LT.figures(cells):
            if shape_of(fg).is_empty:                            # снаружи родителя целиком — не черновик
                continue
            fs = set(fg)
            best = max(((len(fs & {tuple(c) for c in r.get('cells') or []}), i) for i, r in enumerate(mine)
                        if i not in used), default=(0, None))
            if best[0] > 0:
                r = mine[best[1]]
                used.add(best[1])
                if sorted(map(tuple, r.get('cells') or [])) != fg:
                    r['cells'] = [list(c) for c in fg]
                    set_geom(m, r, shape_of(fg))
                    changed.append(f"{_t(r)} follows the brush")
            else:
                r = dict(uid=uuid.uuid4().hex, num=new_num(m), obj=obj(m, par), fw=par.get('fw', m.frame_size[0]),
                         fh=par.get('fh', m.frame_size[1]), pts=[], thumbs=[], kind='draft', parent=puid, src=src,
                         cells=[list(c) for c in fg], factors=list(fac))
                set_geom(m, r, shape_of(fg))
                m._regions.append(r)
                changed.append(f"Draft {r['num']} ({LT.address(*fg[0], list(fac))}{' ...' if len(fg) > 1 else ''})")
        for i, r in enumerate(mine):
            if i not in used:
                _vanish(m, r)
                changed.append(f"{_t(r)} gone with its cells")
    renamed = False                                              # подсетки, заданные квадратам до черновика:
    for par, f, d in _latest_grids(m):                           # 'cell:X,Y' -> черновик с этим квадратом
        rf = d.get('refine') or {}
        for key in [k_ for k_ in rf if k_.startswith('cell:')]:
            c = tuple(map(int, key[5:].split(',')))
            r = next((r for r in m._regions if r.get('src') == f"{par['uid']}@1" and
                      c in {tuple(x) for x in r.get('cells') or []}), None)
            if r is None:
                continue
            d2 = json.load(open(os.path.join(f, 'grid.json'), encoding='utf-8'))
            d2['refine'][r['uid']] = d2['refine'].pop(key)
            with open(os.path.join(f, 'grid.json'), 'w', encoding='utf-8') as fo:
                json.dump(d2, fo, indent=1)
            renamed = True
    if renamed:
        more = sync_brush(m)
        if more:
            changed.append(more)
    for r in [r for r in m._regions if r.get('src') and is_mask(r)]:   # кисть очищена (или подсетка сменилась)
        if r not in m._regions or not r.get('src'):              # ушёл или отпущен в этом же проходе
            continue
        pu, lev = r['src'].split('@')
        fac = () if lev == '1' else tuple(map(int, lev.split('x')))
        if (pu, fac) not in want and (pu in seen1 if lev == '1' else pu in seenR):
            _vanish(m, r)
            changed.append(f"{_t(r)} gone with its cells")
    for r in [r for r in m._regions if r.get('parent') and idx(m, r['parent']) is None]:   # сирот не бывает
        par0 = next((p for p in m._regions if p.get('kind') not in ('group', 'draft') and not p.get('parent')
                     and p is not r and geom(m, p).buffer(EPS).contains(geom(m, r).representative_point())), None)
        r['parent'] = par0['uid'] if par0 is not None else None
    return "; ".join(changed) if changed else None


def _vanish(m, r):
    """Черновик ушёл вместе с клетками: всё, что внутри него, остаётся и переходит к его родителю."""
    if r not in m._regions:
        return
    release_children(m, r)
    m._regions.remove(r)


def brush_into_grid(regs, L, folder, g):
    """Новая сетка съёмки (по решётке образца) получает кисть из sample.json: квадраты черновиков уровня 1
    этого выделения -> g.selected; подсетки и клетки черновиков в них -> возвращаемый 'refine'."""
    from . import lattice as LT
    from .snap import stage_to_mosaic
    me = next((r for r in regs if not is_group(r) and not is_mask(r) and
               os.path.abspath(folder).startswith(os.path.abspath(_sel_dir(folder, r)) + os.sep)), None)
    if me is not None:
        for r in regs:
            if r.get('src') == f"{me['uid']}@1":
                for a, b in r.get('cells') or []:
                    q = stage_to_mosaic(folder, LT.to_stage(L, [(a + 0.5, b + 0.5)]))[0]
                    g.selected.add(g.cell_at(q))
    rf = {}
    for r in regs:
        if r.get('sub'):
            fac = 'x'.join(map(str, (r.get('factors') or []) + [r['sub']]))
            cs = [c for o in regs if o.get('src') == f"{r['uid']}@{fac}" for c in o.get('cells') or []]
            rf[r['uid']] = dict(n=int(r['sub']), cells=sorted(cs))
    return rf


def _sel_dir(folder, r):
    from .project import sel_folder, sample_of
    return sel_folder(sample_of(folder), r)


def _t(r):
    return f"{'Draft' if is_mask(r) else 'Selection'} {r['num']}"


def remove_brush_cells(m, reg):
    """Клетки выделения кисти уходят из кисти (удалили или изменили операцией): кисть его больше не создаст.
    Снятые квадраты уровня 1 — в grid.json 'selected_removed'."""
    from . import lattice as LT
    from .snap import stage_to_mosaic
    from .mask import Grid
    if not reg.get('src') or not reg.get('cells'):
        return
    L = lattice_of(m)
    puid, lev = reg['src'].split('@')
    gone = {tuple(c) for c in reg['cells']}
    for par, f, d in _latest_grids(m):
        if lev == '1' and par['uid'] == puid and L:
            g = Grid(f)
            if not g.ready():
                continue
            q = stage_to_mosaic(f, LT.to_stage(L, [(a + 0.5, b + 0.5) for a, b in gone]))
            drop = {g.cell_at(p) for p in q}
            d2 = json.load(open(os.path.join(f, 'grid.json'), encoding='utf-8'))
            d2['selected_removed'] = d2.get('selected_removed', []) + [list(c) for c in sorted(drop & g.selected)]
            d2['selected'] = [c for c in d2.get('selected', []) if tuple(c) not in drop]
            d2['brush_t'] = time.time()
            with open(os.path.join(f, 'grid.json'), 'w', encoding='utf-8') as fo:
                json.dump(d2, fo, indent=1)
        rq = (d.get('refine') or {}).get(puid)
        if lev != '1' and rq and rq.get('cells'):
            d2 = json.load(open(os.path.join(f, 'grid.json'), encoding='utf-8'))
            d2['refine'][puid]['cells'] = [c for c in rq['cells'] if tuple(c) not in gone]
            with open(os.path.join(f, 'grid.json'), 'w', encoding='utf-8') as fo:
                json.dump(d2, fo, indent=1)


def release_children(m, reg):
    """Выделение уходит, а то, что внутри него, остаётся — как после разгруппировки: черновики из его подсетки
    становятся самостоятельными (без связи с его клетками), всё переходит к его родителю (пирамидка цела)."""
    u = reg.get('uid')
    for c in m._regions:
        if c is reg or c.get('parent') != u:
            continue
        if is_mask(c) and (c.get('src') or '').startswith(f"{u}@"):
            c.pop('src', None)                                   # клетки остаются, кисть ими больше не управляет
        c['parent'] = reg.get('parent')
    for par, f, d in _latest_grids(m):                           # подсетка ушедшего — из кисти
        if u in (d.get('refine') or {}):
            d2 = json.load(open(os.path.join(f, 'grid.json'), encoding='utf-8'))
            d2['refine'].pop(u, None)
            with open(os.path.join(f, 'grid.json'), 'w', encoding='utf-8') as fo:
                json.dump(d2, fo, indent=1)


def delete_brush_cells(m, reg):
    """Черновик кисти удаляют: его клетки снимаются с кисти, вложенные остаются (release_children)."""
    release_children(m, reg)
    remove_brush_cells(m, reg)
