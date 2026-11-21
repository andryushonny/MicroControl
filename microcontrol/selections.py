"""Отснятые выделения (selections) на диске: список деревом слоёв, открыть для просмотра, удалить совсем.

    m.list()          все отснятые выделения: карта 10x, под ней её участки 50x, под ними 100x ...
    m.open(3)         открыть №3 из списка в просмотре (mc.view)
    m.remove(3)       удалить №3 совсем (с вопросом); m.remove(3, confirm=False) — без вопроса

Вложенность: у новых сканов в meta.json записано, внутри какого выделения отмечено это (parent_uid);
у старых — по месту на столике: выделение меньшего увеличения, в котором лежит середина этого.
"""
import os
import json
import shutil
import numpy as np


def _root(root):
    if root is None:
        root = 'scans' if os.path.isdir('scans') else '.'
    return os.path.abspath(root)


def _is_sel(d):
    return os.path.exists(os.path.join(d, 'meta.json'))


def _folders(root):
    """Папки выделений: обычные сканы и выделения внутри папок сеансов (кроме _removed)."""
    out = []
    if not os.path.isdir(root):
        return out
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p) or d.startswith(('.', '_')):
            continue
        if _is_sel(p):
            out.append(p)
        elif os.path.exists(os.path.join(p, 'session.json')):
            out += [os.path.join(p, s) for s in sorted(os.listdir(p))
                    if not s.startswith(('.', '_')) and _is_sel(os.path.join(p, s))]
    return out


def _info(f, root):
    meta = json.load(open(os.path.join(f, 'meta.json'), encoding='utf-8'))
    try:
        import csv
        rows = list(csv.DictReader(open(os.path.join(f, 'index.csv'), encoding='utf-8')))
    except Exception:
        rows = []
    obj = meta.get('objective')
    fov = meta.get('fov_um') or {}
    ms = meta.get('motor_step_um') or 1.0
    box = meta.get('area_real_um')
    if box:                                                   # углы — центры крайних кадров
        x0, y0, x1, y1 = box
        fw, fh = fov.get('x', 0), fov.get('y', 0)
        box = [min(x0, x1) - fw / 2, min(y0, y1) - fh / 2, max(x0, x1) + fw / 2, max(y0, y1) + fh / 2]
    elif rows:                                                # по положениям кадров (настоящие шаги мотора)
        try:
            P = np.array([[float(r['x_p']), float(r['y_p'])] for r in rows]) * ms
            fw, fh = fov.get('x', 0), fov.get('y', 0)
            box = [P[:, 0].min() - fw / 2, P[:, 1].min() - fh / 2, P[:, 0].max() + fw / 2, P[:, 1].max() + fh / 2]
        except Exception:
            box = None
    ses = os.path.dirname(f) if os.path.exists(os.path.join(os.path.dirname(f), 'session.json')) else None
    n_sel = meta.get('selection') or (meta.get('sample_index') if ses else None)
    grid = {}
    try:
        grid = json.load(open(os.path.join(f, 'grid.json'), encoding='utf-8'))
    except Exception:
        pass
    state = []
    state.append("stitched" if os.path.exists(os.path.join(f, 'join', 'positions_all.csv')) else "not stitched")
    if grid.get('marks'):
        state.append("grid")
    if grid.get('selected'):
        state.append(f"{len(grid['selected'])} squares")
    if os.path.exists(os.path.join(f, 'export')):
        state.append("exported")
    return dict(folder=f, rel=os.path.relpath(f, root), objective=obj, frames=meta.get('frames', len(rows)),
                date=(meta.get('date') or '')[:16].replace('T', ' '), name=meta.get('sample') or '',
                selection=n_sel, uid=meta.get('sample_uid') or f, parent_uid=meta.get('parent_uid'),
                box=box, state=", ".join(state), session=ses)


def _link(items):
    """Родитель каждого выделения: по parent_uid (последний скан с таким uid), иначе по месту на столике."""
    by_uid = {}
    for k, it in enumerate(items):
        by_uid.setdefault(it['uid'], []).append(k)
    for k, it in enumerate(items):
        it['parent'] = None
        if it['parent_uid'] and it['parent_uid'] in by_uid:
            cand = [j for j in by_uid[it['parent_uid']] if j != k]
            if cand:
                it['parent'] = max(cand, key=lambda j: items[j]['date'])
                continue
        if not it['box'] or not it['objective']:
            continue
        c = ((it['box'][0] + it['box'][2]) / 2, (it['box'][1] + it['box'][3]) / 2)
        best = None
        for j, o in enumerate(items):
            if j == k or not o['box'] or not o['objective'] or o['objective'] >= it['objective']:
                continue
            b = o['box']
            if b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3]:
                key = (-o['objective'], (b[2] - b[0]) * (b[3] - b[1]), o['date'])
                if best is None or key < best[0]:
                    best = (key, j)
        if best:
            it['parent'] = best[1]


def _tree(items):
    order = []

    def walk(k, d):
        items[k]['depth'] = d
        order.append(k)
        for c in sorted((c for c, it in enumerate(items) if it['parent'] == k), key=lambda c: items[c]['date']):
            walk(c, d + 1)
    for k in sorted((k for k, it in enumerate(items) if it['parent'] is None), key=lambda k: items[k]['date']):
        walk(k, 0)
    for it in items:                                          # родитель — сама запись (а не номер)
        it['parent'] = items[it['parent']] if it['parent'] is not None else None
    return [items[k] for k in order]


class Selections(list):
    """Список отснятых выделений деревом слоёв; печатается таблицей. s[0]['folder'] — папка первого."""
    root = ''

    def __repr__(self):
        if not self:
            return f"No scanned selections in {self.root}"
        lab = []
        for it in self:
            t = f"Selection {it['selection']}" if it['selection'] else (it['name'] or "Scan")
            if it['selection'] and it['name'] and not it['name'].endswith(f" {it['selection']}"):
                t += f" ({it['name']})"
            lab.append(("    " * (it['depth'] - 1) + "  └ " if it['depth'] else "") + t)
        w = max(14, max(len(t) for t in lab) + 2)
        lines = [f"Scanned selections in {self.root}  (layers: a map and the details inside it)",
                 f"  {'#':>3}  {'Selection':<{w}}{'Obj':>5}{'Frames':>8}  {'Size, mm':<13}{'Date':<18}State"]
        for k, (it, t) in enumerate(zip(self, lab)):
            b = it['box']
            size = f"{(b[2] - b[0]) / 1000:.2f} x {(b[3] - b[1]) / 1000:.2f}" if b else "-"
            lines.append(f"  {k + 1:>3}  {t:<{w}}{str(it['objective'] or '-') + 'x':>5}{it['frames']:>8}  "
                         f"{size:<13}{it['date']:<18}{it['state']}")
            lines.append(f"  {'':>3}  {'':<{w}}{'':>5}{'':>8}  {it['rel']}")
        lines.append("  m.open(N): look at it   m.remove(N): delete it completely")
        return "\n".join(lines)

    __str__ = __repr__

    def _repr_pretty_(self, p, cycle):
        p.text(repr(self))


def list_selections(root=None):
    """Все отснятые выделения в папке сканов, деревом слоёв (карта -> участки -> подробные участки)."""
    root = _root(root)
    items = []
    for f in _folders(root):
        try:
            items.append(_info(f, root))
        except Exception:
            pass
    _link(items)
    out = Selections(_tree(items))
    out.root = root
    return out


def resolve(which, root=None):
    """Папка выделения: номер из m.list() (с 1), путь к папке или часть имени папки."""
    if isinstance(which, (int, np.integer)):
        lst = list_selections(root)
        if not 1 <= which <= len(lst):
            raise IndexError(f"No selection #{which}: the list has {len(lst)} (m.list()).")
        return lst[which - 1]['folder']
    which = str(which)
    if os.path.isdir(which):
        return os.path.abspath(which)
    r = _root(root)
    if os.path.isdir(os.path.join(r, which)):
        return os.path.join(r, which)
    hits = [it['folder'] for it in list_selections(root) if which in it['rel']]
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"'{which}': {'several' if hits else 'no'} selections match. Use the number from m.list().")


def open_selection(which, root=None, **kw):
    """Открыть отснятое выделение в просмотре (mc.view): крестики, квадраты, экспорт."""
    from .viewer import view
    return view(resolve(which, root), **kw)


def remove_selection(which, root=None, confirm=True):
    """Удалить отснятое выделение совсем (папку со всеми кадрами). Пустая после этого папка сеанса — тоже.
    confirm=True — сначала спросить (ввести yes)."""
    from .util import Result
    f = resolve(which, root)
    lst = list_selections(root)
    me = next((it for it in lst if os.path.abspath(it['folder']) == os.path.abspath(f)), None)
    kids = [it for it in lst if me is not None and it is not me and _is_child(lst, it, me)]
    size = sum(os.path.getsize(os.path.join(a, b)) for a, _d, fs in os.walk(f) for b in fs) / 2 ** 20
    if confirm:
        note = (f" Selections inside it stay: {', '.join(os.path.basename(k['folder']) for k in kids)}." if kids else "")
        ans = input(f"Delete {os.path.relpath(f, _root(root))} completely ({size:.0f} MB, cannot be undone)?{note} "
                    "Type yes: ")
        if ans.strip().lower() not in ('yes', 'y', 'да'):
            return Result("Not deleted", Folder=f)
    shutil.rmtree(f)
    ses = os.path.dirname(f)
    gone_ses = False
    if os.path.exists(os.path.join(ses, 'session.json')) and not any(
            _is_sel(os.path.join(ses, d)) for d in os.listdir(ses) if not d.startswith('_')):
        shutil.rmtree(ses)
        gone_ses = True
    from . import log as L
    L.event('SEL-001', f"Selection {f} deleted ({size:.0f} MB)" + (", empty session folder too" if gone_ses else "") + ".")
    r = Result("Deleted", Folder=f, Freed=f"{size:.0f} MB")
    if gone_ses:
        r['Session'] = f"{ses} (nothing left in it, deleted too)"
    return r


def _is_child(lst, it, me):
    """it лежит внутри me (на любой глубине)?"""
    for _ in range(50):
        it = it['parent']
        if it is None:
            return False
        if it is me:
            return True
    return False


# ------------------------------------------------------------------ выбранные квадраты (зелёные) на картах
_SQ = {}
_ALL = {}


def marked_squares(folder):
    """Выбранные в mc.view квадраты скана: dict(px=[4x2 мозаика], um=[4x2 настоящие мкм столика], size=(W, H)
    мозаики) или None. Мозаика -> столик: аффинно по положениям кадров (центр кадра = где стоял столик);
    скан в один ряд — по матрице камеры A. Форма и масштаб квадратов сохраняются."""
    gp = os.path.join(folder, 'grid.json')
    if not os.path.exists(gp):
        return None
    jp = os.path.join(folder, 'join', 'positions_all.csv')
    key = (os.path.getmtime(gp), os.path.getmtime(jp) if os.path.exists(jp) else 0)
    if folder in _SQ and _SQ[folder][0] == key:
        return _SQ[folder][1]
    out = None
    try:
        from .mask import Grid
        g = Grid(folder)
        if g.ready() and g.selected:
            from .storage import ScanData, meta_A
            from .stitch import load_positions
            st = ScanData(folder)
            try:
                H, W = st.shape[:2]
                P = load_positions(folder, st)
                Pm = P - P.min(0)
                ms = st.meta.get('motor_step_um') or 1.0
                S = np.array([[float(r['x_p']), float(r['y_p'])] for r in st.index]) * ms
                A = meta_A(st.meta, W, H)
            finally:
                st.close()
            Pc = Pm + (W / 2, H / 2)
            if len(Pc) >= 3 and np.linalg.matrix_rank(Pc - Pc.mean(0), tol=1.0) == 2:
                M = np.linalg.lstsq(np.c_[Pc, np.ones(len(Pc))], S, rcond=None)[0]          # 3 x 2
            else:
                Lin = -np.linalg.inv(A) * ms                                                # px -> мкм
                M = np.vstack([Lin.T, S.mean(0) - Pc.mean(0) @ Lin.T])
            px = [np.array([g.cr.point(i, j), g.cr.point(i + 1, j), g.cr.point(i + 1, j + 1), g.cr.point(i, j + 1)],
                           float) for i, j in sorted(g.selected)]
            out = dict(px=px, um=[np.c_[q, np.ones(4)] @ M for q in px], size=Pm.max(0) + (W, H))
    except Exception:
        out = None
    _SQ[folder] = (key, out)
    return out


def marked_all(root, every=3.0):
    """Все выбранные квадраты всех сканов в папке сканов (настоящие мкм столика): [4x2]. Обновляется не чаще
    раза в every секунд."""
    import time
    now = time.monotonic()
    c = _ALL.get(root)
    if c and now - c[0] < every:
        return c[1]
    polys = []
    for f in _folders(root):
        r = marked_squares(f)
        if r:
            polys += r['um']
    _ALL[root] = (now, polys)
    return polys
