"""Несколько образцов на столике: области (зоны) скана, их цвета, папка сеанса и вертикальная карусель справа.

Области задаются в m.manual() строго по очереди: 1 — первый угол новой области, 2 — второй угол (область
зафиксирована). Области не перекрываются. Точки фокуса нумеруются сквозь все области: у первой 1, 2, 3, 4,
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
# после 8-й — цвета по «золотому углу» оттенка, не повторяются.
_BASE = [(80, 200, 80), (230, 150, 50), (40, 150, 250), (200, 100, 180), (210, 200, 40),
         (170, 120, 245), (60, 190, 160), (90, 120, 180)]


def color(i):
    """Цвет i-й области (с 0), BGR."""
    if i < len(_BASE):
        return _BASE[i]
    h = (0.13 + (i - len(_BASE)) * 0.618034) % 1.0
    if 0.10 < h < 0.20 or h > 0.95 or h < 0.03:          # не жёлтый и не красный
        h = (h + 0.33) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def dim(c, k=0.45):
    return tuple(int(v * k) for v in c)


# ---------------------------------------------------------------- области в m.manual()

def rect(reg, fw, fh):
    """Прямоугольник области на образце (настоящие мкм): от края первого кадра до края последнего."""
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
        if getattr(m, 'area', None) is not None:          # зона, заданная раньше одной парой углов, — область 1
            off_ = (m._off['x'], m._off['y'])
            m._regions.append(dict(A=(m.area.x0 + off_[0], m.area.y0 + off_[1]),
                                   B=(m.area.x1 + off_[0], m.area.y1 + off_[1]),
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
    if getattr(m, '_sel', None) is None or m._sel >= len(m._regions):
        m._sel = len(m._regions) - 1 if m._regions else None


def num(m, k):
    """Номер области k (индекс в m._regions; len(m._regions) — новая): постоянный, после удаления других
    не меняется и не используется снова."""
    if k is None:
        return None
    if k < len(m._regions):
        return m._regions[k]['num']
    return (m._pending or {}).get('num', getattr(m, '_area_seq', 0) + 1)


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
    fw, fh = m.frame_size
    return rect(reg, fw, fh)


def _area(m, reg):
    from .scan import Area
    w = lambda P: (P[0] - m._off['x'], P[1] - m._off['y'])
    return Area(*w(reg['A']), *w(reg['B']))


def area_at(m, x, y):
    """Номер области (0…), в которую попадает точка (настоящие мкм), или None."""
    return next((k for k, rg in enumerate(m._regions) if inside(_rect(m, rg), x, y)), None)


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
        hit_ = area_at(m, rp.x, rp.y)
        if hit_ is not None:
            raise MicroControlError('E704', f"Here is sample {num(m, hit_)}: a new area must not overlap it")
        if m._pending:                                     # угол 1 заново
            m._pending['A'] = (rp.x, rp.y)
            m._pending['pts'][0] = fpt
            m._pending['thumbs'][:1] = th
        else:
            m._pending = dict(A=(rp.x, rp.y), pts=[fpt], thumbs=th, num=getattr(m, '_area_seq', 0) + 1,
                              uid=uuid.uuid4().hex)
        k = len(m._regions)
        return (f"Sample {num(m, k)}: corner 1 at X {p.x:.0f}, Y {p.y:.0f} um, focus point "
                f"{len(numbered(m)) - len(m._pending['pts']) + 1}. Go to the other corner, press 2")
    if which != 2:
        raise MicroControlError('E703', "Corner is 1 or 2")
    if not m._pending:
        raise MicroControlError('E701', "Press 1 first: the first corner of a new sample area")
    Aw = (m._pending['A'][0] - m._off['x'], m._pending['A'][1] - m._off['y'])
    Bw = snap(m, Aw, (p.x, p.y))                           # угол 2 — ровно на сетке кадров от угла 1
    new = dict(A=m._pending['A'], B=(Bw[0] + m._off['x'], Bw[1] + m._off['y']),
               pts=[m._pending['pts'][0], fpt] + m._pending['pts'][1:],
               thumbs=m._pending['thumbs'][:1] + th, num=m._pending['num'], uid=m._pending['uid'])
    bad = [rg['num'] for rg in m._regions if overlap(_rect(m, new), _rect(m, rg))]
    if bad:
        raise MicroControlError('E704', f"This area overlaps sample {', '.join(map(str, bad))}: "
                                        f"choose corner 2 so that the areas do not overlap")
    m._regions.append(new)
    m._area_seq = max(getattr(m, '_area_seq', 0), new['num'])         # номер больше не выдаётся
    m._pending, m._sel = None, len(m._regions) - 1
    nx_, ny_ = _area(m, new).size(m._frame_step(1 - scan_overlap(m.objective)))
    return f"Sample {new['num']} fixed: {nx_} x {ny_} = {nx_ * ny_} frames. Key 1: the next sample"


def add_point(m):
    """Точка фокуса (Z здесь): области под столиком, иначе новой, иначе выбранной. Возвращает сообщение."""
    from .errors import MicroControlError
    init_areas(m)
    rp, p = m.real_position, m.position
    k = area_at(m, rp.x, rp.y)
    if k is not None:
        m._regions[k]['pts'].append(dict(x=rp.x, y=rp.y, z=rp.z))
    elif m._pending:
        m._pending['pts'].append(dict(x=rp.x, y=rp.y, z=rp.z))
    elif m._sel is not None:
        k = m._sel
        m._regions[k]['pts'].append(dict(x=rp.x, y=rp.y, z=rp.z))
    else:
        raise MicroControlError('E701', "Mark a sample area first: keys 1 and 2")
    fpn = len(numbered(m)) if k is None else [n_ for n_, kk, _p in numbered(m) if kk == k][-1]
    who = f"sample {num(m, k if k is not None else len(m._regions))}"
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
    return f"Z {m.position.z:.1f} um from the focus points of sample {num(m, k)} ({dz:+.1f})"


def delete_area(m, k):
    """Удалить область k (0…); k = len(m._regions) — отменить новую. Возвращает сообщение."""
    from .errors import MicroControlError
    init_areas(m)
    if k == len(m._regions) and m._pending:
        m._pending = None
        return "New area cancelled"
    if not 0 <= k < len(m._regions):
        raise MicroControlError('E703', "No such sample")
    n_ = m._regions.pop(k)['num']
    m._sel = min(k, len(m._regions) - 1) if m._regions else None
    return f"Sample {n_} deleted (the numbers of the others stay)"


def goto_area(m, k):
    """Поехать в ближайшую к столику клетку (кадр) области k: сначала вниз (если её фокус ниже), потом по X, Y
    отрезками не длиннее max_jump, потом вверх до её фокуса — подъём кусками не больше max_move_z.
    Возвращает сообщение."""
    from .scan import focus_z
    from .errors import MicroControlError
    from .manual import snap
    init_areas(m)
    if not 0 <= k < len(m._regions):
        raise MicroControlError('E703', "No such sample")
    reg = m._regions[k]
    rp = m.real_position
    (ax, ay), (bx, by) = reg['A'], reg['B']
    cx = min(max(rp.x, min(ax, bx)), max(ax, bx))                # ближайшая точка области ...
    cy = min(max(rp.y, min(ay, by)), max(ay, by))
    X, Y = snap(m, (ax, ay), (cx, cy))                            # ... и центр её клетки (кадра)
    z_t = focus_z(reg['pts'], X, Y) - m._off['z'] if reg['pts'] else None
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
    return f"Sample {reg['num']}: at its nearest frame" + (f", focus Z {z_t:.1f} um (from its points)" if z_t is not None else "")


def areas_info(m):
    """Области для людей и агента: номер, углы и размер (рабочие мкм), кадры, точки фокуса со сквозными номерами."""
    from .config import scan_overlap
    init_areas(m)
    st = m._frame_step(1 - scan_overlap(m.objective))
    nums = numbered(m)
    w = lambda x, y: (round(x - m._off['x'], 1), round(y - m._off['y'], 1))
    out = []
    for k, reg in enumerate(m._regions):
        x0, y0, x1, y1 = _rect(m, reg)
        nx, ny = _area(m, reg).size(st)
        out.append(dict(sample=reg['num'], state='fixed', selected=(k == m._sel),
                        corner1_um=w(*reg['A']), corner2_um=w(*reg['B']),
                        size_mm=[round((x1 - x0) / 1000, 3), round((y1 - y0) / 1000, 3)], frames=nx * ny,
                        focus_points=[dict(number=n, x_um=w(p['x'], p['y'])[0], y_um=w(p['x'], p['y'])[1],
                                           z_um=round(p['z'] - m._off['z'], 1)) for n, kk, p in nums if kk == k]))
    if m._pending:
        out.append(dict(sample=num(m, len(m._regions)), state='new (corner 2 not set yet)', selected=False,
                        corner1_um=w(*m._pending['A']), corner2_um=None,
                        focus_points=[dict(number=n, x_um=w(p['x'], p['y'])[0], y_um=w(p['x'], p['y'])[1],
                                           z_um=round(p['z'] - m._off['z'], 1)) for n, kk, p in nums if kk == -1]))
    return out


# ---------------------------------------------------------------- сеанс из нескольких образцов

def is_sample(folder):
    return os.path.exists(os.path.join(folder, 'meta.json'))


def session_of(folder):
    """(папка сеанса или None, список папок образцов). Для обычного скана без сеанса — (None, [folder])."""
    folder = os.path.abspath(folder)
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

CAR_W = 210          # ширина карусели, px
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


def carousel(H, items, sel, scroll=0, title="Samples", go=True, hover=None):
    """Вертикальная карусель на всю высоту. items: [dict(title, lines, color, thumbs, confirm)].
    hover — (x, y) мыши в координатах карусели (подсветка кнопок и карточки) или None.
    Возвращает (картинка H x CAR_W, области щелчков [(x0, y0, x1, y1, действие, индекс)], высота содержимого).
    Действия: 'sel' — выбрать, 'go' — поехать (если go), 'del' — удалить (второй щелчок — подтверждение)."""
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
    top = 39
    y = 46 - scroll
    for k, it in enumerate(items):
        th = it.get('thumbs') or []
        h_th = max((t.shape[0] for t in th), default=0)
        h = 46 + 15 * len(it.get('lines', [])) + (h_th + 6 if th else 0)
        y0, y1 = y, y + h
        if y1 > top and y0 < H:
            c = it['color']
            card = img[max(top, y0):min(H, y1), 6:W - 6]
            hv = over(6, y0, W - 6, y1)
            if k == sel or hv:
                card[:] = np.clip(card.astype(int) + (16 if k == sel else 9), 0, 255).astype(np.uint8)
            cv2.rectangle(img, (6, y0 + 2), (W - 7, y1 - 2), c if k == sel else (66, 66, 74), 1)
            cv2.rectangle(img, (6, y0 + 2), (10, y1 - 2), c, -1)            # цветная полоска области
            cv2.putText(img, it['title'], (18, y0 + 22), _FONT, 0.52, c, 1, cv2.LINE_AA)
            conf = it.get('confirm')
            bx0, bx1 = W - 36, W - 14                                      # кнопка «удалить»
            _button(img, bx0, y0 + 8, bx1, y0 + 28, 'x', over(bx0, y0 + 8, bx1, y0 + 28), True, bool(conf))
            hits.append((bx0, y0 + 8, bx1, y0 + 28, 'del', k))
            if go and it.get('go', True):                                 # кнопка «поехать»
                gx0, gx1 = W - 78, W - 42
                _button(img, gx0, y0 + 8, gx1, y0 + 28, 'Go', over(gx0, y0 + 8, gx1, y0 + 28))
                hits.append((gx0, y0 + 8, gx1, y0 + 28, 'go', k))
            yy = y0 + 44
            for ln in it.get('lines', []):
                cv2.putText(img, ln, (18, yy), _FONT, 0.4, (200, 200, 205), 1, cv2.LINE_AA)
                yy += 15
            if conf:
                cv2.putText(img, "click x again to delete", (18, yy), _FONT, 0.38, (120, 120, 255), 1, cv2.LINE_AA)
            if th:
                xx = 18
                for t in th:
                    ty0 = yy - 6
                    a, b = max(0, top - ty0), max(0, ty0 + t.shape[0] - H)
                    if t.shape[0] - a - b > 0 and xx + t.shape[1] <= W - 8:
                        img[ty0 + a:ty0 + t.shape[0] - b, xx:xx + t.shape[1]] = t[a:t.shape[0] - b]
                        cv2.rectangle(img, (xx, ty0), (xx + t.shape[1], ty0 + t.shape[0]), dim(c, 0.8), 1)
                    xx += t.shape[1] + 6
            hits.append((6, y0, W - 6, y1, 'sel', k))
        y = y1 + 6
    content = y + scroll
    if not items:
        cv2.putText(img, "none yet", (14, 66), _FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "keys 1 and 2: area corners", (14, 88), _FONT, 0.4, (120, 120, 125), 1, cv2.LINE_AA) \
            if go else None
    if content > H:                                                       # полоса прокрутки
        hgt = max(20, int(H * H / content))
        top_ = int((H - hgt) * scroll / max(1, content - H))
        cv2.rectangle(img, (W - 3, top_), (W - 1, top_ + hgt), (120, 120, 120), -1)
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
        if folder and os.path.exists(os.path.join(folder, 'session.json')):
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
