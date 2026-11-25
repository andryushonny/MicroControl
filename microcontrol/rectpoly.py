"""Прямоугольный (ортогональный) многоугольник по точкам: каждое ребро строго горизонтальное или вертикальное.

Всё — в одной системе координат (рамка: оси окна в момент начала). Функции чистые: окно просмотра даёт точки
мыши и кандидатов привязки, получает вершины, направляющие и замкнутый контур.

    next_point(verts, p, xs, ys, tol)  -> (q, guides)   точка, куда встанет следующая вершина
    add(verts, q)                      -> (verts', msg) вершина добавлена (или msg — почему нет)
    close(verts)                       -> (ring, msg)   замкнуть: угол достраивается сам
"""
import numpy as np
from shapely.geometry import Polygon, LineString, Point

EPS = 1e-6


def _near(v, cands, tol):
    """Ближайший к v кандидат в пределах tol (или None)."""
    best = None
    for c in cands:
        d = abs(c - v)
        if d <= tol and (best is None or d < best[0]):
            best = (d, c)
    return None if best is None else best[1]


def next_point(verts, p, xs=(), ys=(), tol=0.0, nodes=()):
    """Куда встанет следующая вершина под мышью p.
    Первая — p, привязанная к узлам nodes (точка) или к xs / ys (по осям). Следующие — по горизонтали или по
    вертикали от последней (что ближе к мыши); свободная координата привязывается к xs / ys (x и y прежних вершин,
    линии сетки), к первой вершине — в первую очередь. -> (q, guides): guides — [('x', значение) | ('y', значение)]."""
    p = np.asarray(p, float)
    xs = list(xs) + [v[0] for v in verts]
    ys = list(ys) + [v[1] for v in verts]
    guides = []
    if not verts:
        for n in nodes:
            if np.hypot(*(np.asarray(n, float) - p)) <= tol:
                return np.asarray(n, float), [('x', float(n[0])), ('y', float(n[1]))]
        q = p.copy()
        for a, cs in ((0, xs), (1, ys)):
            c = _near(q[a], cs, tol)
            if c is not None:
                q[a] = c
                guides.append(('xy'[a], float(c)))
        return q, guides
    L = np.asarray(verts[-1], float)
    d = p - L
    a = 0 if abs(d[0]) >= abs(d[1]) else 1                    # свободная ось: 0 — ребро горизонтальное
    q = L.copy()
    q[a] = p[a]
    F = np.asarray(verts[0], float)
    first = [F[a]] if len(verts) >= 2 else []
    c = _near(q[a], first, tol)
    if c is None:
        c = _near(q[a], [v for v in (xs, ys)[a] if abs(v - L[a]) > EPS], tol)
    if c is not None:
        q[a] = c
        guides.append(('xy'[a], float(c)))
    guides.append(('xy'[1 - a], float(L[1 - a])))           # само ребро
    return q, guides


def _axis(u, v):
    """Ось ребра u -> v: 0 — горизонтальное, 1 — вертикальное, None — нулевое или косое."""
    dx, dy = abs(v[0] - u[0]), abs(v[1] - u[1])
    if dx <= EPS and dy <= EPS:
        return None
    if dy <= EPS:
        return 0
    if dx <= EPS:
        return 1
    return None


def add(verts, q):
    """Добавить вершину q (уже от next_point). То же направление, что у последнего ребра, — та вершина просто
    сдвигается (продолжение прямой); назад по ребру и пересечения — нельзя. -> (новые вершины, сообщение или '')."""
    q = [float(q[0]), float(q[1])]
    vs = [list(map(float, v)) for v in verts]
    if not vs:
        return [q], ''
    L = vs[-1]
    a = _axis(L, q)
    if a is None:
        return vs, ("The same point" if abs(q[0] - L[0]) + abs(q[1] - L[1]) <= EPS else
                    "Only horizontal or vertical sides")
    if len(vs) >= 2 and _axis(vs[-2], L) == a:                 # по той же прямой
        s_old = np.sign(L[a] - vs[-2][a])
        s_new = np.sign(q[a] - L[a])
        if s_old != s_new:
            return vs, "The side would go back over itself"
        vs[-1] = q
        if len(vs) >= 2 and _crosses(vs[:-1], vs[-2], q):
            return [list(map(float, v)) for v in verts], "The sides would cross"
        return vs, ''
    if _crosses(vs, L, q):
        return vs, "The sides would cross"
    return vs + [q], ''


def _crosses(vs, L, q):
    """Пересекает ли ребро L -> q путь vs (кроме общей вершины L)."""
    if len(vs) < 2:
        return False
    path = LineString(vs)
    seg = LineString([L, q])
    x = path.intersection(seg)
    return not x.is_empty and not x.difference(Point(L).buffer(1e-4)).is_empty


def _clean(ring):
    """Без повторов и вершин посреди прямой (по кругу)."""
    r = [list(v) for v in ring]
    changed = True
    while changed and len(r) > 2:
        changed = False
        for i in range(len(r)):
            a, b, c = r[i - 1], r[i], r[(i + 1) % len(r)]
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) <= EPS or (_axis(a, b) is not None and _axis(a, b) == _axis(b, c)):
                r.pop(i)
                changed = True
                break
    return r


def valid(ring):
    """Контур годится: >= 4 вершин, все рёбра по осям, без самопересечений, площадь > 0."""
    if len(ring) < 4:
        return False
    if any(_axis(ring[i - 1], ring[i]) is None for i in range(len(ring))):
        return False
    pg = Polygon(ring)
    return pg.is_valid and pg.area > EPS and LineString(ring + [ring[0]]).is_simple


def close(verts):
    """Замкнуть: последняя вершина на одной прямой с первой — сразу; иначе угол достраивается (тот из двух,
    при котором контур не пересекает себя; при двух — тот, где стороны чередуются). -> (контур, '') или (None, msg)."""
    vs = [list(map(float, v)) for v in verts]
    if len(vs) >= 2 and abs(vs[-1][0] - vs[0][0]) + abs(vs[-1][1] - vs[0][1]) <= EPS:
        vs = vs[:-1]                                         # щёлкнули в первую вершину
    if len(vs) < 3:
        return None, "Put at least three points (the last corner is added by itself)"
    L, F = vs[-1], vs[0]
    if _axis(L, F) is not None:
        ring = _clean(vs)
        return (ring, '') if valid(ring) else (None, "The outline would cross itself")
    best = None
    for c in ([F[0], L[1]], [L[0], F[1]]):
        ring = _clean(vs + [c])
        if not valid(ring):
            continue
        alt = (_axis(vs[-2], L) != _axis(L, c)) + (_axis(c, F) != _axis(F, vs[1]))
        if best is None or alt > best[0]:
            best = (alt, ring)
    if best is None:
        return None, "Cannot close it without crossing: move the last point"
    return best[1], ''
