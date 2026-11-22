"""Решётка образца: кресты литографии и вложенные сетки выделений.

Кресты и квадраты нумеруются литографией от 1,1 (обычно до ~30,30, бывает до 58,58). В координатах решётки
(u, v) крест X,Y стоит в (X − 1, Y − 1), квадрат X,Y — клетка [X − 1, X] × [Y − 1, Y]. Решётка образца
(sample.json, 'lattice') — где это на столике: S = O + u·e1 + v·e2 (настоящие мкм; e1, e2 — шаг крестов).

Вложенные сетки. Выделение — набор клеток уровня N: клетка (a, b) — квадрат [a/N, (a+1)/N] × [b/N, (b+1)/N]
в (u, v). Квадраты литографии — уровень 1. У выделения может быть своя подсетка n = 2…5 (reg['sub']): его
дети — клетки уровня N·n, каждая лежит внутри клетки родителя (строгая вложенность). reg['factors'] —
цепочка подсеток от квадрата литографии ([] — сам квадрат, [4] — четверть, [4, 5] — двадцатая часть).
Адрес клетки: «12,7 › 2,1 › 1,3» — квадрат, затем номер подклетки на каждом уровне (с единицы).
"""
import os
import numpy as np


def from_grid(folder, g):
    """Решётка образца по размеченной съёмке (сетка g: кресты + подписи номеров) -> dict или None."""
    from .snap import mosaic_to_stage
    if not g.ready():
        return None
    num = g.numbering()
    pts_uv, pts_q = [], []
    for i, j in ((0, 0), (1, 0), (0, 1), (1, 1)):
        X, Y = g.name(i, j, num)
        pts_uv.append((X - 0.5, Y - 0.5))                         # центр квадрата X,Y
        pts_q.append(g.cr.point(i + 0.5, j + 0.5))
    S = mosaic_to_stage(folder, np.array(pts_q))
    A = np.c_[np.array(pts_uv), np.ones(4)]
    M = np.linalg.lstsq(A, S, rcond=None)[0]                     # (u, v, 1) -> S
    return dict(O=M[2].tolist(), e1=M[0].tolist(), e2=M[1].tolist(), pitch=float(np.hypot(*M[0])),
                numbered=bool(g.labels) and 'fixed' in num[2], src=os.path.abspath(folder))


def to_stage(L, uv):
    uv = np.atleast_2d(np.asarray(uv, float))
    return np.asarray(L['O']) + uv[:, :1] * np.asarray(L['e1']) + uv[:, 1:2] * np.asarray(L['e2'])


def to_uv(L, S):
    E = np.c_[L['e1'], L['e2']]
    return np.linalg.solve(E, (np.atleast_2d(np.asarray(S, float)) - np.asarray(L['O'])).T).T


def cell_uv(a, b, N):
    return np.array([(a, b), (a + 1, b), (a + 1, b + 1), (a, b + 1)], float) / N


def cell_stage(L, a, b, N):
    return to_stage(L, cell_uv(a, b, N))


def cells_geom(L, cells, N):
    """Контур набора клеток (shapely, настоящие мкм)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    return unary_union([Polygon(cell_stage(L, a, b, N)) for a, b in cells]).buffer(0)


def address(a, b, factors):
    """Адрес клетки уровня N = prod(factors): «12,7 › 2,1 › 1,3»."""
    N = int(np.prod(factors)) if factors else 1
    X, Y = a // N + 1, b // N + 1
    out = [f"{X},{Y}"]
    ra, rb, rest = a % N, b % N, N
    for f in factors:
        rest //= f
        out.append(f"{ra // rest + 1},{rb // rest + 1}")
        ra, rb = ra % rest, rb % rest
    return " > ".join(out)


def pitch_text(L, N):
    """Размер клетки уровня N: «25 um»."""
    return f"{L['pitch'] / N:.3g} um" if L else f"1/{N}"


def level(reg):
    """Уровень клеток выделения: N = произведение подсеток предков."""
    return int(np.prod(reg.get('factors') or [1]))


def seed_grid(folder, g, L):
    """Сетка съёмки без разметки — по решётке образца: три креста и подписи номеров (как будто отметили)."""
    from .snap import stage_to_mosaic
    S = to_stage(L, [(0, 0), (1, 0), (0, 1), (1, 1)])            # кресты 1,1; 2,1; 1,2
    q = stage_to_mosaic(folder, S)
    g.cr.marks, g.cr.ij = [], []
    g.cr._fit()
    for p in q[:3]:
        g.cr.add(np.asarray(p, float))
    n1, n2 = np.array(g.cr.ij[1], float), np.array(g.cr.ij[2], float)   # узлы крестов 2,1 и 1,2
    A = np.linalg.inv(np.c_[n1, n2])                             # узел -> смещение номера креста
    g.labels = {}
    for i, j in ((0, 0), (1, 0), (0, 1)):                        # три квадрата — нумерация однозначна
        nums = [np.rint(A @ np.array(c, float)).astype(int) + 1 for c in ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))]
        X, Y = np.min(nums, axis=0)
        g.labels[(i, j)] = (int(X), int(Y))
    return g


def figures(cells):
    """Клетки -> замкнутые фигуры (соседи по стороне), большие первыми."""
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
