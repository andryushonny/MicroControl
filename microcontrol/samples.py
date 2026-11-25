"""Выделения (selections) на столике: области скана, их цвета, слои по объективам, папка сеанса и
вертикальная карусель справа.

Выделения задаются в m.manual() строго по очереди: 1 — первый угол нового выделения, 2 — второй угол
(выделение зафиксировано). Каждое выделение помнит объектив, на котором его отметили (слой). Выделения
одного объектива не перекрываются; выделение большего увеличения можно отметить внутри выделения меньшего
(матрёшка: карта 10x -> участки 50x -> участки 100x). m.scan() снимает выделения текущего объектива. Точки фокуса нумеруются сквозь все области: у первой 1, 2, 3, 4,
у второй 5, 6, 7, 8 и т. д. Скан всех областей — одна папка сеанса, в ней по папке на образец:
    scans/<дата>_<объектив>_<имя>/session.json
    scans/<дата>_<объектив>_<имя>/sample_01/ ... (обычный скан: meta.json, кадры, grid.json, export/)
"""
import os
import json
import uuid
import shutil
import colorsys
import numpy as np
import cv2

# Палитра областей (BGR): различимые цвета, без красного (решётка) и жёлтого (поле зрения, подсказки);
# дальше — до 100 цветов, каждый следующий — самый непохожий на уже взятые.
_BASE = [(80, 200, 80), (230, 150, 50), (40, 150, 250), (200, 100, 180), (210, 200, 40),
         (170, 120, 245), (60, 190, 160), (90, 120, 180)]


def _palette(n=100):
    """n различимых цветов (BGR): первые — _BASE, дальше каждый следующий — самый далёкий от уже взятых
    в CIELAB (жадно); без красного (решётка), жёлтого (поле зрения, подсказки) и слишком тёмных (фон тёмный)."""
    cand = []
    for h in np.arange(0, 360, 4.0):
        if h < 22 or h > 330 or 38 < h < 72:                  # красный, жёлтый
            continue
        for sv in ((0.55, 0.95), (0.8, 0.95), (0.45, 0.75), (0.8, 0.75), (0.3, 1.0), (0.95, 0.85)):
            r, g, b = colorsys.hsv_to_rgb(h / 360, *sv)
            cand.append((int(b * 255), int(g * 255), int(r * 255)))
    cand = np.array(cand, np.uint8)
    lab = lambda c: cv2.cvtColor(np.asarray(c, np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    L = lab(cand)
    keep = (L[:, 0] > 105)                                    # не темнее ~40 % (OpenCV: L * 255 / 100)
    cand, L = cand[keep], L[keep]
    out = [tuple(int(v) for v in c) for c in _BASE]
    d = np.min([np.linalg.norm(L - x, axis=1) for x in lab(out)], axis=0)
    while len(out) < n:
        k = int(np.argmax(d))
        out.append(tuple(int(v) for v in cand[k]))
        d = np.minimum(d, np.linalg.norm(L - L[k], axis=1))
    return out


_PAL = _palette(100)


def color(i):
    """Цвет i-й области (с 0), BGR: 100 различимых, дальше — по кругу."""
    return _PAL[i % len(_PAL)]


def dim(c, k=0.45):
    return tuple(int(v * k) for v in c)


# ---------------------------------------------------------------- области в m.manual()

def rect(reg, fw, fh):
    """Прямоугольник области на образце (настоящие мкм): от края первого кадра до края последнего.
    Размер кадра — объектива самого выделения (если записан), иначе fw, fh."""
    fw, fh = reg.get('fw', fw), reg.get('fh', fh)
    (ax, ay), (bx, by) = reg['A'], reg['B']
    return min(ax, bx) - fw / 2, min(ay, by) - fh / 2, max(ax, bx) + fw / 2, max(ay, by) + fh / 2


def overlap(r1, r2):
    return min(r1[2], r2[2]) - max(r1[0], r2[0]) > 1e-6 and min(r1[3], r2[3]) - max(r1[1], r2[1]) > 1e-6


def inside(r, x, y):
    return r[0] <= x <= r[2] and r[1] <= y <= r[3]


def numbered(m):
    """Точки фокуса со сквозными номерами: [(номер, индекс области или -1 для новой, точка)]."""
    out, n = [], 0
    for k, reg in enumerate(getattr(m, '_regions', None) or []):
        for p in reg['pts']:
            n += 1
            out.append((n, k, p))
    pend = getattr(m, '_pending', None)
    for p in (pend['pts'] if pend else []):
        n += 1
        out.append((n, -1, p))
    return out


def thumb(img, w=84, h=63):
    """Маленькая копия кадра для карусели."""
    s = min(w / img.shape[1], h / img.shape[0])
    t = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return t if t.ndim == 3 else cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------- области: общая логика (ручной режим и агент)
def init_areas(m):
    """m._regions (зафиксированные области, настоящие мкм), m._pending (новая: есть угол 1), m._sel (выбранная)."""
    if getattr(m, '_regions', None) is None:
        m._regions = []
    if not m._regions and getattr(m, 'area', None) is not None and \
            not getattr(m.area, '_of_selection', False):      # зона m.area = Area(...) без выделений — выделение 1
        off_ = (m._off['x'], m._off['y'])
        fw_, fh_ = m.frame_size
        m._regions.append(dict(A=(m.area.x0 + off_[0], m.area.y0 + off_[1]),
                               B=(m.area.x1 + off_[0], m.area.y1 + off_[1]), obj=m.objective, fw=fw_, fh=fh_, prog=True,
                               pts=[dict(x=f_['x'], y=f_['y'], z=f_['z'])
                                    for f_ in (getattr(m, '_focus_pts', None) or [])], thumbs=[]))
    if not hasattr(m, '_pending'):
        m._pending = None
    seq = getattr(m, '_area_seq', 0)
    for rg in m._regions:                                  # у каждой области — свой скрытый uid и постоянный номер
        rg.setdefault('uid', uuid.uuid4().hex)
        if 'num' not in rg:
            seq += 1
            rg['num'] = seq
        seq = max(seq, rg['num'])
    m._area_seq = seq
    m._regions[:] = [rg for rg in m._regions if rg.get('kind') != 'group']   # список плоский: без групп
    for rg in m._regions:                                  # и без вложенности: все на одном уровне
        rg.pop('parent', None)
    rks_ = [rg['rank'] for rg in m._regions if rg.get('rank') is not None]
    top_ = min(rks_) if rks_ else 0
    for rg in m._regions:                                  # новое — наверх списка (верхний слой)
        if rg.get('rank') is None:
            top_ -= 1
            rg['rank'] = top_
    if getattr(m, '_sel', None) is None or m._sel >= len(m._regions):
        m._sel = len(m._regions) - 1 if m._regions else None


def persist(m):
    """Выделения -> project.json открытого проекта (переживают перезапуск)."""
    try:
        from .layers import refresh_groups
        refresh_groups(m)
        from .project import save_regions
        save_regions(m)
    except Exception:
        pass


def num(m, k):
    """Номер области k (индекс в m._regions; len(m._regions) — новая): постоянный, после удаления других
    не меняется и не используется снова."""
    if k is None:
        return None
    if k < len(m._regions):
        return m._regions[k]['num']
    return (m._pending or {}).get('num', used_max(m) + 1)


def col(m, k):
    """Цвет области k — по её постоянному номеру (не меняется после удаления других)."""
    return color(num(m, k) - 1)


def index_of(m, n):
    """Индекс области по её номеру n (как на экране); None — нет такой."""
    init_areas(m)
    k = next((i for i, rg in enumerate(m._regions) if rg['num'] == n), None)
    if k is None and m._pending and m._pending.get('num') == n:
        return len(m._regions)
    return k


def _rect(m, reg):
    """Охватывающий прямоугольник контура выделения (настоящие мкм)."""
    from .layers import bounds
    if reg.get('shape') or reg.get('kind') == 'group':
        return bounds(m, reg)
    fw, fh = m.frame_size
    return rect(reg, fw, fh)


def obj(m, reg):
    """Объектив выделения (на котором его отметили)."""
    return reg.get('obj', m.objective)


def parent(m, k):
    """Вложенности нет: все выделения — на одном уровне (порядок списка = слои). Всегда None."""
    return None


def _geo_parent(m, k):
    """Выделение меньшего увеличения, внутри которого лежит центр выделения k (по месту)."""
    rg = m._regions[k] if k < len(m._regions) else m._pending
    if rg is None or 'B' not in rg:
        return None
    r = _rect(m, rg)
    c = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
    best = None
    for i, o in enumerate(m._regions):
        if i == k or o.get('kind') == 'group' or (o.get('kind') != 'draft' and obj(m, o) > obj(m, rg)):
            continue
        q = _rect(m, o)
        if o.get('kind') != 'draft' and obj(m, o) == obj(m, rg) and \
                not (q[0] <= r[0] and q[1] <= r[1] and q[2] >= r[2] and q[3] >= r[3] and q != r):
            continue                                       # того же объектива — только если целиком внутри
        if inside(q, *c):
            key = (q[2] - q[0]) * (q[3] - q[1])                 # самое маленькое, в котором лежит
            if best is None or key < best[0]:
                best = (key, i)
    return best[1] if best else None


def shoot_mode(m, reg):
    """Как снимать выделение: 'quality' — для точного снимка (перекрытие 40 %, подъезд с одной стороны, PNG, запас
    вокруг рамки): рамки кристаллов (Ctrl+тянуть); иначе None — обычная съёмка. Сшивка после съёмки — всегда quick,
    точная (quality) — отдельно: Stitch."""
    if reg.get('mode') == 'quality' or reg.get('free'):
        return 'quality'
    return None


def frames_of(m, k):
    """(кадров по X, по Y) выделения k при съёмке его объективом (перекрытие — для того объектива)."""
    from .config import scan_overlap
    import math
    rg = m._regions[k]
    o = obj(m, rg)
    if o == m.objective:
        st = m._frame_step(1 - scan_overlap(o))
    else:
        s_ = m.motor_step
        st = {a: max(1, math.floor(rg.get(f, 1.0) * (1 - scan_overlap(o)) / s_ + 1e-9)) * s_
              for a, f in (('x', 'fw'), ('y', 'fh'))}
    if rg.get('kind') in ('group', 'draft'):                # группа и маска не снимаются
        return 0, 0
    key = (id(rg.get('shape')), tuple(rg['A']), tuple(rg['B']), rg.get('fw'), rg.get('fh'), m._off['x'], m._off['y'],
           st['x'], st['y'], m.motor_step, o == m.objective)
    c = _FCACHE.get(id(rg))
    if c is not None and c[0] == key and c[1] is rg.get('shape'):   # окно спрашивает каждый кадр — считаем раз
        return c[2]
    if not rg.get('shape'):
        res = _area(m, rg).size(st)
    else:
        from .layers import touches_frame
        fw, fh = rg.get('fw', m.frame_size[0]), rg.get('fh', m.frame_size[1])
        n = sum(touches_frame(m, rg, x + m._off['x'], y + m._off['y'], fw, fh) for _r, _c, x, y in _area(m, rg).grid(st))
        res = (n, 1)                                        # не прямоугольник: число кадров — первым
    if len(_FCACHE) > 2000:
        _FCACHE.clear()
    _FCACHE[id(rg)] = (key, rg.get('shape'), res)
    return res


_FCACHE = {}


def to_top(m, r):
    """Выделение — наверх списка (верхний слой)."""
    rks_ = [x.get('rank') for x in m._regions if x is not r and x.get('rank') is not None]
    r['rank'] = (min(rks_) if rks_ else 0) - 1


def tree(m):
    """Выделения по порядку списка (сверху — верхний слой): [(индекс, 0)]."""
    return [(k, 0) for k in order(m)]


def order(m):
    """Индексы выделений по порядку списка: сверху — верхний слой (рисуется поверх остальных)."""
    rk = lambda r: r.get('rank') if r.get('rank') is not None else -1e9
    return sorted(range(len(m._regions)), key=lambda c: (rk(m._regions[c]), -m._regions[c].get('num', 0)))


def siblings(m, p, par=None):
    """Выделения, лежащие прямо в выделении p (None — верхний слой), в порядке карусели: сначала по 'rank'
    (порядок, заданный перетаскиванием), новые — в конце, по объективу и номеру."""
    return order(m) if p is None else []


def reorder_ok(m, src, dst):
    """Можно ли поставить src рядом с dst: всегда (список плоский)."""
    return max(src, dst) < len(m._regions)


def reorder(m, src, dst, after=False, save=True):
    """Перетаскивание в карусели: поставить src перед dst (after=True — после). src, dst — индексы выделений
    одного слоя (лежат прямо в одном выделении или группе).
    Возвращает сообщение; порядок хранится в sample.json ('rank')."""
    if src == dst:
        return None
    if isinstance(src, int) and isinstance(dst, int) and max(src, dst) < len(m._regions):
        sib = order(m)
        sib.remove(src)
        sib.insert(sib.index(dst) + (1 if after else 0), src)
        for i, c in enumerate(sib):
            m._regions[c]['rank'] = i
        if save:
            persist(m)
        return f"Selection {num(m, src)} moved {'after' if after else 'before'} Selection {num(m, dst)}"
    return "Drop it on another card"


def title(m, k):
    """«Selection 3» или «Group 4» (или имя, если дали)."""
    rg = m._regions[k] if k < len(m._regions) else m._pending
    kind_ = 'Group' if rg.get('kind') == 'group' else 'Selection'      # черновик — выделение без объектива
    return rg.get('name') or f"{kind_} {num(m, k)}"


def layer_name(m, k):
    """«10x» или «50x inside Selection 1»."""
    rg = m._regions[k]
    return "draft" if rg.get('kind') == 'draft' else f"{obj(m, rg)}x"


def _area(m, reg):
    """Зона съёмки выделения (рабочие мкм): крайние кадры — наружу до целого шага мотора (столик встаёт только
    на шаги: край контура не теряется), число кадров — как у самой зоны (шаг чуть шире, перекрытие чуть меньше)."""
    import math
    from .scan import Area
    from .config import scan_overlap
    s_ = m.motor_step
    (ax, bx), (ay, by) = sorted((reg['A'][0], reg['B'][0])), sorted((reg['A'][1], reg['B'][1]))
    o = obj(m, reg)
    st = {a: max(1, math.floor(reg.get(f, 1.0) * (1 - scan_overlap(o)) / s_ + 1e-9)) * s_
          for a, f in (('x', 'fw'), ('y', 'fh'))} if 'fw' in reg else m._frame_step(1 - scan_overlap(o))
    n = list(Area(ax, ay, bx, by).size(st))
    ax0, ay0, bx0, by0 = ax, ay, bx, by
    ax, ay = math.floor(ax / s_ + 1e-6) * s_, math.floor(ay / s_ + 1e-6) * s_
    bx, by = math.ceil(bx / s_ - 1e-6) * s_, math.ceil(by / s_ - 1e-6) * s_
    gb = None
    if reg.get('shape'):                                    # контур меньше кадра (рамка с кристаллом): он сам
        from .layers import from_json
        gb = from_json(reg['shape']).bounds
    for i_, (lo, hi, lo0, hi0) in enumerate(((ax, bx, ax0, bx0), (ay, by, ay0, by0))):
        c_ = round((lo + hi) / 2 / s_) * s_                 # один кадр не накроет контур, если центр не на шаге
        if n[i_] == 1 and hi > lo and abs(c_ - (lo0 + hi0) / 2) > 1e-6:
            f_ = reg.get('fw' if i_ == 0 else 'fh', 0.0)
            if gb is not None:
                c_ = round((gb[i_] + gb[i_ + 2]) / 2 / s_) * s_     # шаг, ближайший к середине контура
            if gb is not None and c_ - f_ / 2 <= gb[i_] + 1e-6 and c_ + f_ / 2 >= gb[i_ + 2] - 1e-6:
                lo = hi = c_                                # кадр на ближнем шаге накрывает контур целиком
                if i_ == 0:
                    ax = bx = c_
                else:
                    ay = by = c_
                continue
            n[i_] = 2
    a_ = Area(ax - m._off['x'], ay - m._off['y'], bx - m._off['x'], by - m._off['y'])
    a_.n = tuple(n)
    return a_


def area_at(m, x, y, any_objective=False):
    """Индекс выделения текущего объектива, в которое попадает точка (настоящие мкм), или None.
    any_objective=True — любого (сначала текущего объектива, потом с наибольшим увеличением)."""
    from .layers import contains
    hits = [k for k, rg in enumerate(m._regions) if rg.get('kind') != 'group' and contains(m, rg, x, y)]
    same = [k for k in hits if obj(m, m._regions[k]) == m.objective]
    sel_ = getattr(m, '_sel', None)
    if sel_ in same:                                       # перекрываются: выбранное справа — первым,
        return sel_
    if same:                                               # дальше — по порядку списка (верхний слой)
        rk_ = {k: i for i, k in enumerate(order(m))}
        same.sort(key=lambda k: rk_.get(k, 1e9))
    if same or not any_objective:
        return same[0] if same else None
    return max(hits, key=lambda k: obj(m, m._regions[k])) if hits else None


def corner(m, which, img=None):
    """Угол области в текущем положении столика: 1 — начать новую область, 2 — зафиксировать её.
    Z в этот момент запоминается как точка фокуса. Возвращает сообщение."""
    from .errors import MicroControlError
    from .config import scan_overlap
    from .manual import snap
    init_areas(m)
    rp, p = m.real_position, m.position
    fpt = dict(x=rp.x, y=rp.y, z=rp.z)
    th = [thumb(img)] if img is not None else []
    if which == 1:
        if m._pending:                                     # угол 1 заново
            m._pending['A'] = (rp.x, rp.y)
            m._pending['pts'][0] = fpt
            m._pending['thumbs'][:1] = th
        else:
            fw_, fh_ = m.frame_size
            m._pending = dict(A=(rp.x, rp.y), pts=[fpt], thumbs=th, num=used_max(m) + 1,
                              uid=uuid.uuid4().hex, obj=m.objective, fw=fw_, fh=fh_)
        k = len(m._regions)
        persist(m)
        return (f"Selection {num(m, k)} ({m.objective}x): corner 1 at X {p.x:.0f}, Y {p.y:.0f} um, focus point "
                f"{len(numbered(m)) - len(m._pending['pts']) + 1}. Go to the other corner, press 2 (Del: cancel)")
    if which != 2:
        raise MicroControlError('E703', "Corner is 1 or 2")
    if not m._pending:
        raise MicroControlError('E701', "Press 1 first: the first corner of a new selection")
    if obj(m, m._pending) != m.objective:
        raise MicroControlError('E705', f"Corner 1 was set with {obj(m, m._pending)}x, now the objective is "
                                        f"{m.objective}x: switch back (O) or start again (Del, then 1)")
    Aw = (m._pending['A'][0] - m._off['x'], m._pending['A'][1] - m._off['y'])
    Bw = snap(m, Aw, (p.x, p.y))                           # угол 2 — ровно на сетке кадров от угла 1
    new = dict(A=m._pending['A'], B=(Bw[0] + m._off['x'], Bw[1] + m._off['y']),
               pts=[m._pending['pts'][0], fpt] + m._pending['pts'][1:],
               thumbs=m._pending['thumbs'][:1] + th, num=m._pending['num'], uid=m._pending['uid'],
               obj=m.objective, fw=m._pending.get('fw', m.frame_size[0]), fh=m._pending.get('fh', m.frame_size[1]))
    m._regions.append(new)                                 # перекрытия — любые (список плоский, порядок = слои)
    rks_ = [rg.get('rank') for rg in m._regions if rg.get('rank') is not None]
    new['rank'] = (min(rks_) if rks_ else 0) - 1           # новое — наверх списка
    m._area_seq = max(getattr(m, '_area_seq', 0), new['num'])         # номер больше не выдаётся
    m._pending, m._sel = None, len(m._regions) - 1
    persist(m)
    nx_, ny_ = _area(m, new).size(m._frame_step(1 - scan_overlap(m.objective)))
    return (f"Selection {new['num']} fixed ({m.objective}x): {nx_} x {ny_} = {nx_ * ny_} frames. "
            "Key 1: the next selection")


def add_point(m):
    """Точка фокуса (Z здесь): новой (угол 1 уже есть), иначе ВЫБРАННОЙ справа (её объектив должен стоять),
    иначе той, что под столиком. Возвращает сообщение."""
    from .errors import MicroControlError
    init_areas(m)
    rp, p = m.real_position, m.position
    k = None
    sel_ = m._sel if m._sel is not None and m._sel < len(m._regions) and \
        m._regions[m._sel].get('kind') not in ('group', 'draft') else None
    if m._pending:
        m._pending['pts'].append(dict(x=rp.x, y=rp.y, z=rp.z))
    elif sel_ is not None:                                 # выбранная — ей (выделения перекрываются: не по месту)
        if obj(m, m._regions[sel_]) != m.objective:
            raise MicroControlError('E701', f"Selection {num(m, sel_)} is for {obj(m, m._regions[sel_])}x, the objective "
                                            f"is {m.objective}x: its focus points need its objective (O)")
        k = sel_
        m._regions[k]['pts'].append(dict(x=rp.x, y=rp.y, z=rp.z))
    else:
        k = area_at(m, rp.x, rp.y)                         # ничего не выбрано — та, что под столиком
        if k is None:
            raise MicroControlError('E701', f"Choose a selection (its card) or mark one with {m.objective}x: keys 1 and 2")
        m._regions[k]['pts'].append(dict(x=rp.x, y=rp.y, z=rp.z))
    fpn = len(numbered(m)) if k is None else [n_ for n_, kk, _p in numbered(m) if kk == k][-1]
    persist(m)
    who = f"Selection {num(m, k if k is not None else len(m._regions))}"
    return f"Focus point {fpn} ({who}): Z {p.z:.1f} um at X {p.x:.0f}, Y {p.y:.0f}"


def follow_z(m):
    """Флажок «Z по точкам фокуса» включён и столик внутри области с ≥ 2 точками фокуса: поставить Z по ним
    (как в скане: 4 ближайшие, веса 1/d²). Вверх — не больше max_move_z за раз. Возвращает сообщение или ''."""
    from .scan import focus_z
    if not getattr(m, '_use_fp', True):
        return ''
    init_areas(m)
    rp = m.real_position
    k = area_at(m, rp.x, rp.y)
    pts = m._regions[k]['pts'] if k is not None else []
    if len(pts) < 2:
        return ''
    z_t = focus_z(pts, rp.x, rp.y)
    dz = z_t - rp.z
    if abs(dz) < 0.5 * m.motor_step:
        return ''
    if dz > m.max_move_z:
        return f"Z from the focus points is {dz:.0f} um higher (more than max_move_z): not raised, check the points"
    m.goto(z=z_t - m._off['z'])
    return f"Z {m.position.z:.1f} um from the focus points of Selection {num(m, k)} ({dz:+.1f})"


def delete_area(m, k):
    """Удалить область k (0…); k = len(m._regions) — отменить новую. Возвращает сообщение."""
    from .errors import MicroControlError
    init_areas(m)
    if k == len(m._regions) and m._pending:
        m._pending = None
        persist(m)
        return "New selection cancelled"
    if not 0 <= k < len(m._regions):
        raise MicroControlError('E703', "No such selection")
    from .layers import delete_brush_cells, _drop
    rg_ = m._regions[k]
    n_ = rg_['num']
    delete_brush_cells(m, rg_)
    _drop(m, [k])
    m._sel = min(k, len(m._regions) - 1) if m._regions else None
    persist(m)
    return f"Selection {n_} deleted (the numbers of the others stay)"


def goto_area(m, k):
    """Поехать в ближайшую к столику клетку (кадр) области k: сначала вниз (если её фокус ниже), потом по X, Y
    отрезками не длиннее max_jump, потом вверх до её фокуса — подъём кусками не больше max_move_z.
    Возвращает сообщение."""
    from .scan import focus_z
    from .errors import MicroControlError
    from .manual import snap
    init_areas(m)
    if not 0 <= k < len(m._regions):
        raise MicroControlError('E703', "No such selection")
    reg = m._regions[k]
    rp = m.real_position
    from .layers import geom
    from shapely.geometry import Point
    from shapely.ops import nearest_points
    g = geom(m, reg)
    if g.is_empty:
        raise MicroControlError('E703', f"{title(m, k)} is empty")
    c = nearest_points(g, Point(rp.x, rp.y))[0]                   # ближайшая точка контура ...
    X, Y = snap(m, reg.get('A', (c.x, c.y)), (c.x, c.y))          # ... и центр её клетки (кадра)
    if g.distance(Point(X, Y)) > 0.5 * min(m.frame_size):        # клетка вне контура (узкое место) — сама точка
        X, Y = c.x, c.y
    z_t = focus_z(reg['pts'], X, Y) - m._off['z'] if reg.get('pts') and obj(m, reg) == m.objective else None
    if z_t is not None and z_t < m.position.z:
        m.goto(z=z_t)
    x_t, y_t = X - m._off['x'], Y - m._off['y']
    p0 = m.position
    nleg = max(1, int(np.ceil(np.hypot(x_t - p0.x, y_t - p0.y) / (m.max_jump * 0.95))))
    for i_ in range(1, nleg + 1):
        m.goto(x=p0.x + (x_t - p0.x) * i_ / nleg, y=p0.y + (y_t - p0.y) * i_ / nleg)
    while z_t is not None and z_t > m.position.z + 0.5 * m.motor_step:
        m.goto(z=min(z_t, m.position.z + m.max_move_z * 0.95))
    m._sel = k
    other = obj(m, reg) != m.objective
    return (f"{title(m, k)} ({obj(m, reg)}x): at its nearest frame"
            + (f", focus Z {z_t:.1f} um (from its points)" if z_t is not None else "")
            + (f". It was marked with {obj(m, reg)}x, now {m.objective}x: Z not changed" if other else ""))


def drive_to(m, X, Y, z=None, trace=None):
    """Безопасно поехать в точку (настоящие мкм): вниз (если z ниже), X/Y отрезками ≤ max_jump, вверх кусками.
    trace(x, y) (рабочие мкм) -> снять ли здесь кадр: столик едет, как всегда, по X, потом по Y, но там, где
    trace просит, останавливается и снимает (туман войны карты — след переезда)."""
    if z is not None and z < m.position.z:
        m.goto(z=z)
    x_t, y_t = X - m._off['x'], Y - m._off['y']
    if trace is not None:
        p0 = m.position
        st_ = 0.9 * min(m.frame_size)                       # шаг следа — чуть меньше кадра: без щелей
        pts_ = []
        nx_ = int(abs(x_t - p0.x) // st_)
        pts_ += [(p0.x + np.sign(x_t - p0.x) * st_ * i, p0.y) for i in range(1, nx_ + 1)]
        ny_ = int(abs(y_t - p0.y) // st_)
        pts_ += [(x_t, p0.y + np.sign(y_t - p0.y) * st_ * i) for i in range(0, ny_ + 1)]
        for wx, wy in pts_:
            if abs(wx - x_t) + abs(wy - y_t) < st_ * 0.1:
                break
            try:
                if trace(wx, wy, False):
                    m.goto(x=wx, y=wy)
                    trace(wx, wy, True)
            except Exception:
                break
    p0 = m.position
    nleg = max(1, int(np.ceil(np.hypot(x_t - p0.x, y_t - p0.y) / (m.max_jump * 0.95))))
    for i_ in range(1, nleg + 1):
        m.goto(x=p0.x + (x_t - p0.x) * i_ / nleg, y=p0.y + (y_t - p0.y) * i_ / nleg)
    while z is not None and z > m.position.z + 0.5 * m.motor_step:
        m.goto(z=min(z, m.position.z + m.max_move_z * 0.95))


def areas_info(m):
    """Области для людей и агента: номер, углы и размер (рабочие мкм), кадры, точки фокуса со сквозными номерами."""
    init_areas(m)
    nums = numbered(m)
    w = lambda x, y: (round(x - m._off['x'], 1), round(y - m._off['y'], 1))
    out = []
    for k, reg in enumerate(m._regions):
        x0, y0, x1, y1 = _rect(m, reg)
        nx, ny = frames_of(m, k)
        out.append(dict(selection=reg['num'], objective=obj(m, reg),
                        inside=(m._regions[parent(m, k)]['num'] if parent(m, k) is not None else None),
                        state='fixed', selected=(k == m._sel),
                        corner1_um=w(*reg['A']), corner2_um=w(*reg['B']),
                        size_mm=[round((x1 - x0) / 1000, 3), round((y1 - y0) / 1000, 3)], frames=nx * ny,
                        focus_points=[dict(number=n, x_um=w(p['x'], p['y'])[0], y_um=w(p['x'], p['y'])[1],
                                           z_um=round(p['z'] - m._off['z'], 1)) for n, kk, p in nums if kk == k]))
    if m._pending:
        out.append(dict(selection=num(m, len(m._regions)), objective=obj(m, m._pending),
                        state='new (corner 2 not set yet)', selected=False,
                        corner1_um=w(*m._pending['A']), corner2_um=None,
                        focus_points=[dict(number=n, x_um=w(p['x'], p['y'])[0], y_um=w(p['x'], p['y'])[1],
                                           z_um=round(p['z'] - m._off['z'], 1)) for n, kk, p in nums if kk == -1]))
    return out


# ---------------------------------------------------------------- сеанс из нескольких образцов

def is_sample(folder):
    return os.path.exists(os.path.join(folder, 'meta.json'))


def session_of(folder):
    """(папка группы или None, список съёмок). Проект: последняя съёмка каждого выделения (открытая версия —
    вместо последней своего выделения). Старый сеанс (session.json): его образцы. Один скан — (None, [скан])."""
    from . import project as P_
    folder = os.path.abspath(folder)
    pf = P_.sample_of(folder)
    if pf:
        subs = P_.latest_scans(pf)
        if is_sample(folder):                              # открыли конкретную (может быть старую) версию
            u_ = P_.scan_uid(folder)
            subs = [folder if (u_ and P_.scan_uid(x) == u_) else x for x in subs]
            if folder not in subs:
                subs.append(folder)
        return pf, subs
    if os.path.exists(os.path.join(folder, 'session.json')):
        ses = folder
    elif is_sample(folder) and os.path.exists(os.path.join(os.path.dirname(folder), 'session.json')):
        ses = os.path.dirname(folder)
    else:
        return None, [folder]
    subs = sorted(os.path.join(ses, d) for d in os.listdir(ses)
                  if not d.startswith('_') and is_sample(os.path.join(ses, d)))
    return ses, subs


def samples(folder):
    """Папки образцов: для сеанса — все образцы, для одного скана — [скан]."""
    return session_of(folder)[1]


def used_max(m):
    """Наибольший номер, который уже занят: у выделений и у съёмок образца (в т. ч. убранных). Выделение, которое
    создали и удалили, ничего не сняв, номер не занимает — следующее получит его же."""
    import glob as _gl, re as _re
    n = max([int(r.get('num', 0) or 0) for r in getattr(m, '_regions', [])] + [0])
    smp = getattr(m, '_sample', None)
    if smp is not None:
        pats = ('scans/*_S[0-9]*', 'scans/_removed/*_S[0-9]*', '_removed/*_S[0-9]*', '_removed/*/*_S[0-9]*')
        for f in [x for p_ in pats for x in _gl.glob(os.path.join(smp.folder, *p_.split('/')))]:
            mo = _re.search(r'_S(\d+)(?:_|$)', os.path.basename(f))
            if mo and os.path.isdir(f):
                n = max(n, int(mo.group(1)))
    return n


def next_num(m):
    """Номер нового выделения: следующий за занятыми (used_max)."""
    m._area_seq = used_max(m) + 1
    return m._area_seq


def sample_index(folder):
    """Номер образца в сеансе (с 1), как при съёмке."""
    try:
        return int(json.load(open(os.path.join(folder, 'meta.json'), encoding='utf-8')).get('sample_index', 1))
    except Exception:
        return 1


def sample_uid(folder):
    """Скрытый уникальный номер образца (uid) из meta.json; '' у старых сканов."""
    try:
        return str(json.load(open(os.path.join(folder, 'meta.json'), encoding='utf-8')).get('sample_uid', ''))
    except Exception:
        return ''


def sample_color(folder):
    """Цвет образца (по номеру в сеансе)."""
    return color(sample_index(folder) - 1)


def remove_sample(folder):
    """Убрать образец из сеанса: папка переносится в <сеанс>/_removed/ (не удаляется — можно вернуть)."""
    ses = os.path.dirname(os.path.abspath(folder))
    dst = os.path.join(ses, '_removed', os.path.basename(folder))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):                                # уже убирали такой же: ничего не стираем
        dst += "_" + (sample_uid(folder) or uuid.uuid4().hex)[:8]
    shutil.move(folder, dst)
    return dst


# ---------------------------------------------------------------- карусель справа

CAR_W = 360          # ширина карусели, px (кнопки карточки и операции помещаются и у вложенных)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _button(img, x0, y0, x1, y1, label, hover=False, danger=False, active=False):
    """Кнопка: тёмная заливка, при наведении светлее; текст строго по центру."""
    base = (62, 62, 70) if not active else (60, 60, 190)
    if hover:
        base = (60, 60, 200) if danger else (96, 96, 104)
    cv2.rectangle(img, (x0, y0), (x1, y1), base, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (120, 120, 130) if hover else (84, 84, 92), 1)
    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
    cv2.putText(img, label, (x0 + (x1 - x0 - tw) // 2 + 1, y0 + (y1 - y0 + th) // 2), _FONT, 0.45,
                (245, 245, 245), 1, cv2.LINE_AA)


def _eye(img, cx, cy, open_=True):
    """Значок глаза: открытый — видно на карте, перечёркнутый — скрыто."""
    c = (235, 235, 235) if open_ else (130, 130, 136)
    cv2.ellipse(img, (cx, cy), (8, 5), 0, 0, 360, c, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, c, -1, cv2.LINE_AA)
    if not open_:
        cv2.line(img, (cx - 8, cy + 6), (cx + 8, cy - 6), (235, 235, 235), 1, cv2.LINE_AA)


def hidden(m, k):
    """Выделение скрыто (само или его группа / родитель)?"""
    from .layers import idx
    seen = 0
    while k is not None and k < len(m._regions) and seen < 30:
        if m._regions[k].get('hidden'):
            return True
        p_ = m._regions[k].get('parent')
        k = idx(m, p_) if p_ else None
        seen += 1
    return False


def wrap(s, scale, max_w, th=1):
    """Строку — в несколько, по словам, чтобы каждая влезала в max_w пикселей (слишком длинное слово режется)."""
    out, cur = [], ""
    for w_ in str(s).split(" "):
        t_ = (cur + " " + w_) if cur else w_
        if cv2.getTextSize(t_, _FONT, scale, th)[0][0] <= max_w or not cur:
            cur = t_
        else:
            out.append(cur)
            cur = w_
        while cv2.getTextSize(cur, _FONT, scale, th)[0][0] > max_w and len(cur) > 4:   # слово длиннее строки
            n_ = len(cur) - 1
            while n_ > 1 and cv2.getTextSize(cur[:n_], _FONT, scale, th)[0][0] > max_w:
                n_ -= 1
            out.append(cur[:n_])
            cur = cur[n_:]
    if cur or not out:
        out.append(cur)
    return out


def _fit_w(t, w):
    """Картинку — не шире w (уменьшить)."""
    if t is None or t.shape[1] <= w or w < 10:
        return t
    k = w / t.shape[1]
    return cv2.resize(t, (max(1, int(t.shape[1] * k)), max(1, int(t.shape[0] * k))), interpolation=cv2.INTER_AREA)


def carousel(H, items, sel, scroll=0, title="Selections", go=True, hover=None, tools=None):
    """Вертикальная карусель на всю высоту. items: [dict(title, lines, color, thumbs, confirm, depth, obj, muted)].
    depth — слой (матрёшка): карточка участка лежит внутри рамки выделения, в котором её отметили, а та — внутри
    своего; obj — объектив («50x») плашкой у заголовка; muted — объектив не тот, что стоит сейчас.
    hover — (x, y) мыши в координатах карусели (подсветка кнопок и карточки) или None.
    Возвращает (картинка H x CAR_W, области щелчков [(x0, y0, x1, y1, действие, индекс)], высота содержимого).
    Действия: 'sel' — выбрать, 'go' — поехать (если go), 'del' — удалить, 'eye' — скрыть / показать (если у карточки
    есть hidden: True / False); 'tool' — кнопка tools[i] под заголовком
    (операции над отмеченными карточками). Карточка с chosen=True отмечена (Ctrl+щелчок)."""
    W = CAR_W
    img = np.full((H, W, 3), 30, np.uint8)
    hits = []
    over = lambda x0, y0, x1, y1: hover is not None and x0 <= hover[0] <= x1 and y0 <= hover[1] <= y1
    cv2.putText(img, title, (14, 25), _FONT, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    if items:
        n_ = str(len(items))
        (tw, _), _ = cv2.getTextSize(n_, _FONT, 0.45, 1)
        cv2.rectangle(img, (W - tw - 26, 10), (W - 12, 30), (62, 62, 70), -1)
        cv2.putText(img, n_, (W - tw - 19, 25), _FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(img, (0, 38), (W, 38), (62, 62, 70), 1)
    rows_, x_ = [[]], 8                                                   # кнопки операций — в строки по ширине
    for t in tools or []:
        tw_ = cv2.getTextSize(t, _FONT, 0.45, 1)[0][0] + 12
        if x_ + tw_ > W - 8 and rows_[-1]:
            rows_.append([])
            x_ = 8
        rows_[-1].append((t, x_, tw_))
        x_ += tw_ + 4
    top = 39 + (26 * len(rows_) + 4 if tools else 0)
    PAD, IND = 6, 10                                                      # поле рамки слоя, отступ на слой
    # 1) раскладка: высота карточек, место каждой рамки (своя карточка + все вложенные)
    depth = [min(4, it.get('depth', 0)) for it in items]
    items = [dict(it, lines=[w_ for ln in it.get('lines', []) for w_ in wrap(ln, 0.4, W - 60 - IND * depth[k])],
                  thumbs=[_fit_w(t, W - 60 - IND * depth[k]) for t in it.get('thumbs') or []])
             for k, it in enumerate(items)]                               # строки — переносом, картинки — по ширине
    ys, y = [], top + 7 - scroll
    for k, it in enumerate(items):
        th = it.get('thumbs') or []
        h_th = max((t.shape[0] for t in th), default=0)
        h = 46 + 15 * len(it.get('lines', [])) + (h_th + 6 if th else 0) + (15 if it.get('confirm') else 0)
        nxt = depth[k + 1] if k + 1 < len(items) else 0
        ys.append((y, y + h))
        y = y + h + 6 + PAD * max(0, depth[k] - nxt)                       # закрыть рамки слоёв ниже
    content = y + scroll
    ends = []                                                             # нижний край рамки k (с вложенными)
    for k in range(len(items)):
        e = k
        while e + 1 < len(items) and depth[e + 1] > depth[k]:
            e += 1
        ends.append(ys[e][1] + PAD * (depth[e] - depth[k]))
    # 2) рамки-матрёшки: от внешних к внутренним
    for k, it in enumerate(items):
        if ends[k] > ys[k][1]:                                            # есть вложенные
            x0 = 6 + IND * depth[k]
            c = it['color'] if not it.get('muted') else dim(it['color'], 0.7)
            y0_, y1_ = max(top, ys[k][0]), min(H - 1, ends[k] + 2)
            if y1_ > y0_:
                xr = W - 7 - 4 * depth[k]
                g = img[y0_:y1_, x0:xr]
                g[:] = (g * 0.6 + np.array(dim(c, 0.35)) * 0.4).astype(np.uint8)
                cv2.rectangle(img, (x0, ys[k][0]), (xr, ends[k] + 2), dim(c, 0.8), 1)
    # 3) карточки
    dmax = max(depth) if depth else 0
    for k, it in enumerate(items):
        y0, y1 = ys[k]
        if y1 <= top or y0 >= H:
            continue
        ix = 6 + IND * depth[k]
        x1c = W - 7 - 4 * dmax                                             # правый край — общий: кнопки в один столбец
        th = it.get('thumbs') or []
        c = it['color'] if not it.get('muted') else dim(it['color'], 0.7)
        card = img[max(top, y0):min(H, y1), ix:x1c]
        hv = over(ix, y0, x1c, y1)
        card[:] = np.clip(card.astype(int) * 0.55 + 30 * 0.45 + (16 if k == sel else (9 if hv else 0)), 0, 255).astype(np.uint8)
        cv2.rectangle(img, (ix, y0 + 2), (x1c, y1 - 2), c if k == sel else (66, 66, 74), 1)
        if it.get('chosen'):                                               # отмечена для операции
            cv2.rectangle(img, (ix + 1, y0 + 3), (x1c - 1, y1 - 3), (235, 235, 235), 1)
            cv2.circle(img, (x1c - 8, y1 - 10), 4, (235, 235, 235), -1)
        cv2.rectangle(img, (ix, y0 + 2), (ix + 4, y1 - 2), c, -1)            # цветная полоска выделения
        bx1 = x1c - 7                                                      # кнопка «удалить»
        bx0 = bx1 - 22
        gx1 = bx0 - 6                                                      # кнопка «поехать»
        gx0 = gx1 - 36
        right = gx0 - 6 if (go and it.get('go', True)) else bx0 - 6
        if it.get('hidden') is not None:                                   # кнопка «глаз»: скрыть / показать
            ex1 = right
            ex0 = ex1 - 24
            right = ex0 - 6
        if it.get('more'):                                                 # кнопка «...»: все съёмки, основная
            mx1_ = right
            mx0_ = mx1_ - 26
            right = mx0_ - 6
        tx = ix + 12
        if it.get('check') is not None:                                   # галочка: идёт ли в экспорт
            cx0, cy0 = tx, y0 + 10
            cv2.rectangle(img, (cx0, cy0), (cx0 + 15, cy0 + 15), (200, 200, 205) if over(cx0 - 3, cy0 - 3, cx0 + 18, cy0 + 18)
                          else (150, 150, 156), 1)
            if it['check']:
                cv2.line(img, (cx0 + 3, cy0 + 8), (cx0 + 6, cy0 + 12), (110, 215, 110), 2, cv2.LINE_AA)
                cv2.line(img, (cx0 + 6, cy0 + 12), (cx0 + 12, cy0 + 3), (110, 215, 110), 2, cv2.LINE_AA)
            hits.append((cx0 - 3, cy0 - 3, cx0 + 18, cy0 + 18, 'chk', k))
            tx += 22
        ob = it.get('obj')
        if ob:                                                             # плашка объектива
            (ow, oh), _ = cv2.getTextSize(ob, _FONT, 0.42, 1)
            right -= ow + 10
            if it.get('mode'):                                             # и вида сшивки
                right -= cv2.getTextSize(it['mode'], _FONT, 0.42, 1)[0][0] + 14
        ttl, fs_ = it['title'], 0.5
        while fs_ > 0.36 and cv2.getTextSize(ttl, _FONT, fs_, 1)[0][0] > right - tx:
            fs_ -= 0.02
        if cv2.getTextSize(ttl, _FONT, fs_, 1)[0][0] > right - tx:
            ttl = ttl.replace("Selection", "Sel.")
        cv2.putText(img, ttl, (tx, y0 + 22), _FONT, fs_, c, 1, cv2.LINE_AA)
        if ob:
            ox0 = tx + cv2.getTextSize(ttl, _FONT, fs_, 1)[0][0] + 7
            cv2.rectangle(img, (ox0, y0 + 9), (ox0 + ow + 8, y0 + 27), dim(c, 0.55), -1)
            cv2.putText(img, ob, (ox0 + 4, y0 + 23), _FONT, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
            hits.append((ox0, y0 + 9, ox0 + ow + 8, y0 + 27, 'obj', k))     # щелчок — другой объектив
            md_ = it.get('mode')
            if md_:                                                        # плашка вида сшивки: quick / quality
                (mw_, _), _ = cv2.getTextSize(md_, _FONT, 0.42, 1)
                mx0 = ox0 + ow + 14
                if mx0 + mw_ + 8 < right + ow + 10:
                    mc_ = (60, 150, 225) if md_ == 'quality' else (120, 120, 128)
                    cv2.rectangle(img, (mx0, y0 + 9), (mx0 + mw_ + 8, y0 + 27), mc_, -1 if md_ == 'quality' else 1)
                    cv2.putText(img, md_, (mx0 + 4, y0 + 23), _FONT, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
                    hits.append((mx0, y0 + 9, mx0 + mw_ + 8, y0 + 27, 'mode', k))   # щелчок — другой вид
        conf = it.get('confirm')
        _button(img, bx0, y0 + 8, bx1, y0 + 28, 'x', over(bx0, y0 + 8, bx1, y0 + 28), True, bool(conf))
        hits.append((bx0, y0 + 8, bx1, y0 + 28, 'del', k))
        if go and it.get('go', True):
            _button(img, gx0, y0 + 8, gx1, y0 + 28, 'Go', over(gx0, y0 + 8, gx1, y0 + 28))
            hits.append((gx0, y0 + 8, gx1, y0 + 28, 'go', k))
        if it.get('hidden') is not None:
            _button(img, ex0, y0 + 8, ex1, y0 + 28, '', over(ex0, y0 + 8, ex1, y0 + 28))
            _eye(img, (ex0 + ex1) // 2, y0 + 18, not it['hidden'])
            hits.append((ex0, y0 + 8, ex1, y0 + 28, 'eye', k))
        if it.get('more'):
            _button(img, mx0_, y0 + 8, mx1_, y0 + 28, '...', over(mx0_, y0 + 8, mx1_, y0 + 28))
            hits.append((mx0_, y0 + 8, mx1_, y0 + 28, 'more', k))
        yy = y0 + 44
        for ln in it.get('lines', []):
            cv2.putText(img, ln, (tx, yy), _FONT, 0.4, (200, 200, 205) if not it.get('muted') else (160, 160, 165),
                        1, cv2.LINE_AA)
            yy += 15
        if conf:
            cv2.putText(img, "click x again to delete", (tx, yy), _FONT, 0.38, (120, 120, 255), 1, cv2.LINE_AA)
            yy += 15
        if th:
            xx = tx
            for t in th:
                if xx + t.shape[1] > W - 8 and W - 8 - xx > 20:          # не влезает по ширине — уменьшить
                    k_t = (W - 8 - xx) / t.shape[1]
                    t = cv2.resize(t, (max(1, int(t.shape[1] * k_t)), max(1, int(t.shape[0] * k_t))),
                                   interpolation=cv2.INTER_AREA)
                ty0 = yy - 6
                a, b = max(0, top - ty0), max(0, ty0 + t.shape[0] - H)
                if t.shape[0] - a - b > 0 and xx + t.shape[1] <= W - 8:
                    img[ty0 + a:ty0 + t.shape[0] - b, xx:xx + t.shape[1]] = t[a:t.shape[0] - b]
                    cv2.rectangle(img, (xx, ty0), (xx + t.shape[1], ty0 + t.shape[0]), dim(c, 0.8), 1)
                xx += t.shape[1] + 6
        hits.append((ix, y0, x1c, y1, 'sel', k))
    img[:top] = np.full((top, W, 3), 30, np.uint8)                        # шапка поверх прокрученного
    cv2.putText(img, title, (14, 25), _FONT, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    tool_hits = []
    if tools:                                                             # операции над отмеченными
        i = 0
        for r_, row in enumerate(rows_):
            yb = 44 + 26 * r_
            for t, x_, tw_ in row:
                _button(img, x_, yb, x_ + tw_, yb + 20, t, over(x_, yb, x_ + tw_, yb + 20))
                tool_hits.append((x_, yb, x_ + tw_, yb + 20, 'tool', i))
                i += 1
    if items:
        cv2.rectangle(img, (W - tw - 26, 10), (W - 12, 30), (62, 62, 70), -1)
        cv2.putText(img, n_, (W - tw - 19, 25), _FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(img, (0, 38), (W, 38), (62, 62, 70), 1)
    hits = [h_ for h_ in hits if h_[3] > top] + tool_hits
    if not items:
        cv2.putText(img, "none yet", (14, 66), _FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "keys 1 and 2: corners", (14, 88), _FONT, 0.4, (120, 120, 125), 1, cv2.LINE_AA) \
            if go else None
    if content > H:                                                       # полоса прокрутки: тянуть мышью
        hgt = max(24, int((H - top) * (H - top) / max(1, content - top)))
        top_ = top + int((H - top - hgt) * scroll / max(1, content - H))
        cv2.rectangle(img, (W - 9, top + 1), (W - 1, H - 1), (44, 44, 50), -1)
        cv2.rectangle(img, (W - 9, max(top + 1, top_)), (W - 1, top_ + hgt), (150, 150, 156), -1)
        hits.insert(0, (W - 11, top + 1, W, H - 1, 'scroll', 0))
    cv2.line(img, (0, 0), (0, H), (70, 70, 78), 1)
    return img, hits, content


def hit(hits, x, y):
    """Что под щелчком: (действие, индекс) — кнопки важнее карточки; или None."""
    best = None
    for x0, y0, x1, y1, act, k in hits:
        if x0 <= x <= x1 and y0 <= y <= y1:
            if act != 'sel':
                return act, k
            best = (act, k)
    return best


def per_sample(fn):
    """Для папки сеанса — выполнить fn для каждого образца; для одного скана — как обычно."""
    import functools

    @functools.wraps(fn)
    def wrap(folder, *a, **kw):
        if folder and (os.path.exists(os.path.join(folder, 'session.json')) or
                       os.path.exists(os.path.join(folder, 'sample.json'))):
            from .util import Result
            out = Result(f"{fn.__name__}: {len(samples(folder))} samples")
            for f in samples(folder):
                try:
                    r = fn(f, *a, **kw)
                    out[os.path.basename(f)] = "; ".join(f"{k} {v}" for k, v in list(r.items())[:4]) \
                        if isinstance(r, dict) else str(r)
                except Exception as e:
                    out[os.path.basename(f)] = f"error: {e}"
            return out
        return fn(folder, *a, **kw)
    return wrap
