"""Несколько образцов на столике: области (зоны) скана, их цвета, папка сеанса и вертикальная карусель справа.

Области задаются в m.manual() строго по очереди: 1 — первый угол новой области, 2 — второй угол (область
зафиксирована). Области не перекрываются. Точки фокуса нумеруются сквозь все области: у первой 1, 2, 3, 4,
у второй 5, 6, 7, 8 и т. д. Скан всех областей — одна папка сеанса, в ней по папке на образец:
    scans/<дата>_<объектив>_<имя>/session.json
    scans/<дата>_<объектив>_<имя>/sample_01/ ... (обычный скан: meta.json, кадры, grid.json, export/)
"""
import os
import json
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
    if getattr(m, '_sel', None) is None or m._sel >= len(m._regions):
        m._sel = len(m._regions) - 1 if m._regions else None


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
            raise MicroControlError('E704', f"Here is sample {hit_ + 1}: a new area must not overlap it")
        if m._pending:                                     # угол 1 заново
            m._pending['A'] = (rp.x, rp.y)
            m._pending['pts'][0] = fpt
            m._pending['thumbs'][:1] = th
        else:
            m._pending = dict(A=(rp.x, rp.y), pts=[fpt], thumbs=th)
        k = len(m._regions)
        return (f"Sample {k + 1}: corner 1 at X {p.x:.0f}, Y {p.y:.0f} um, focus point "
                f"{len(numbered(m)) - len(m._pending['pts']) + 1}. Go to the other corner, press 2")
    if which != 2:
        raise MicroControlError('E703', "Corner is 1 or 2")
    if not m._pending:
        raise MicroControlError('E701', "Press 1 first: the first corner of a new sample area")
    Aw = (m._pending['A'][0] - m._off['x'], m._pending['A'][1] - m._off['y'])
    Bw = snap(m, Aw, (p.x, p.y))                           # угол 2 — ровно на сетке кадров от угла 1
    new = dict(A=m._pending['A'], B=(Bw[0] + m._off['x'], Bw[1] + m._off['y']),
               pts=[m._pending['pts'][0], fpt] + m._pending['pts'][1:],
               thumbs=m._pending['thumbs'][:1] + th)
    bad = [k_ + 1 for k_, rg in enumerate(m._regions) if overlap(_rect(m, new), _rect(m, rg))]
    if bad:
        raise MicroControlError('E704', f"This area overlaps sample {', '.join(map(str, bad))}: "
                                        f"choose corner 2 so that the areas do not overlap")
    m._regions.append(new)
    m._pending, m._sel = None, len(m._regions) - 1
    nx_, ny_ = _area(m, new).size(m._frame_step(1 - scan_overlap(m.objective)))
    return f"Sample {m._sel + 1} fixed: {nx_} x {ny_} = {nx_ * ny_} frames. Key 1: the next sample"


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
    num = len(numbered(m)) if k is None else [n_ for n_, kk, _p in numbered(m) if kk == k][-1]
    who = f"sample {k + 1}" if k is not None else f"sample {len(m._regions) + 1}"
    return f"Focus point {num} ({who}): Z {p.z:.1f} um at X {p.x:.0f}, Y {p.y:.0f}"


def delete_area(m, k):
    """Удалить область k (0…); k = len(m._regions) — отменить новую. Возвращает сообщение."""
    from .errors import MicroControlError
    init_areas(m)
    if k == len(m._regions) and m._pending:
        m._pending = None
        return "New area cancelled"
    if not 0 <= k < len(m._regions):
        raise MicroControlError('E703', f"No sample {k + 1}")
    m._regions.pop(k)
    m._sel = min(k, len(m._regions) - 1) if m._regions else None
    return f"Sample {k + 1} deleted; the others are renumbered"


def goto_area(m, k):
    """Поехать в центр области k: сначала вниз (если её фокус ниже), потом по X, Y отрезками не длиннее
    max_jump, потом вверх до её фокуса — подъём кусками не больше max_move_z. Возвращает сообщение."""
    from .scan import focus_z
    from .errors import MicroControlError
    init_areas(m)
    if not 0 <= k < len(m._regions):
        raise MicroControlError('E703', f"No sample {k + 1}")
    reg = m._regions[k]
    X, Y = (reg['A'][0] + reg['B'][0]) / 2, (reg['A'][1] + reg['B'][1]) / 2
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
    return f"Sample {k + 1}: at its centre" + (f", focus Z {z_t:.1f} um (from its points)" if z_t is not None else "")


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
        out.append(dict(sample=k + 1, state='fixed', selected=(k == m._sel),
                        corner1_um=w(*reg['A']), corner2_um=w(*reg['B']),
                        size_mm=[round((x1 - x0) / 1000, 3), round((y1 - y0) / 1000, 3)], frames=nx * ny,
                        focus_points=[dict(number=n, x_um=w(p['x'], p['y'])[0], y_um=w(p['x'], p['y'])[1],
                                           z_um=round(p['z'] - m._off['z'], 1)) for n, kk, p in nums if kk == k]))
    if m._pending:
        out.append(dict(sample=len(m._regions) + 1, state='new (corner 2 not set yet)', selected=False,
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


def sample_color(folder):
    """Цвет образца (по номеру в сеансе)."""
    return color(sample_index(folder) - 1)


def remove_sample(folder):
    """Убрать образец из сеанса: папка переносится в <сеанс>/_removed/ (не удаляется — можно вернуть)."""
    ses = os.path.dirname(os.path.abspath(folder))
    dst = os.path.join(ses, '_removed', os.path.basename(folder))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(folder, dst)
    return dst


# ---------------------------------------------------------------- карусель справа

CAR_W = 210          # ширина карусели, px
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def carousel(H, items, sel, scroll=0, title="Samples", go=True):
    """Вертикальная карусель на всю высоту. items: [dict(title, lines, color, thumbs, confirm)].
    Возвращает (картинка H x CAR_W, области щелчков [(x0, y0, x1, y1, действие, индекс)], высота содержимого).
    Действия: 'sel' — выбрать, 'go' — поехать (если go), 'del' — удалить (второй щелчок — подтверждение)."""
    W = CAR_W
    img = np.full((H, W, 3), 28, np.uint8)
    hits = []
    cv2.putText(img, title, (10, 22), _FONT, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.line(img, (0, 32), (W, 32), (70, 70, 70), 1)
    y = 40 - scroll
    for k, it in enumerate(items):
        th = it.get('thumbs') or []
        h_th = max((t.shape[0] for t in th), default=0)
        h = 46 + 15 * len(it.get('lines', [])) + (h_th + 6 if th else 0)
        y0, y1 = y, y + h
        if y1 > 34 and y0 < H:
            c = it['color']
            card = img[max(34, y0):min(H, y1)]
            if k == sel:
                card[:] = np.clip(card.astype(int) + 22, 0, 255).astype(np.uint8)
            cv2.rectangle(img, (4, y0 + 2), (W - 5, y1 - 2), c if k == sel else (70, 70, 70), 2 if k == sel else 1)
            cv2.rectangle(img, (4, y0 + 2), (12, y1 - 2), c, -1)          # цветная полоска области
            cv2.putText(img, it['title'], (18, y0 + 20), _FONT, 0.5, c, 1, cv2.LINE_AA)
            bx = W - 30                                                   # кнопка «удалить»
            conf = it.get('confirm')
            cv2.rectangle(img, (bx, y0 + 6), (bx + 22, y0 + 26), (60, 60, 200) if conf else (90, 90, 90), -1 if conf else 1)
            cv2.putText(img, 'x', (bx + 6, y0 + 21), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            hits.append((bx, y0 + 6, bx + 22, y0 + 26, 'del', k))
            if go and it.get('go', True):                                 # кнопка «поехать»
                gx = W - 64
                cv2.rectangle(img, (gx, y0 + 6), (gx + 30, y0 + 26), (90, 90, 90), 1)
                cv2.putText(img, 'Go', (gx + 4, y0 + 21), _FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
                hits.append((gx, y0 + 6, gx + 30, y0 + 26, 'go', k))
            yy = y0 + 40
            for ln in it.get('lines', []):
                cv2.putText(img, ln, (18, yy), _FONT, 0.4, (210, 210, 210), 1, cv2.LINE_AA)
                yy += 15
            if conf:
                cv2.putText(img, "click x again to delete", (18, yy), _FONT, 0.38, (120, 120, 255), 1, cv2.LINE_AA)
            if th:
                xx = 18
                for t in th:
                    ty0 = yy - 6
                    a, b = max(0, 34 - ty0), max(0, ty0 + t.shape[0] - H)
                    if t.shape[0] - a - b > 0 and xx + t.shape[1] <= W - 6:
                        img[ty0 + a:ty0 + t.shape[0] - b, xx:xx + t.shape[1]] = t[a:t.shape[0] - b]
                        cv2.rectangle(img, (xx, ty0), (xx + t.shape[1], ty0 + t.shape[0]), dim(c, 0.8), 1)
                    xx += t.shape[1] + 6
            hits.append((4, y0, W - 4, y1, 'sel', k))
        y = y1 + 4
    content = y + scroll
    if not items:
        cv2.putText(img, "none yet", (10, 60), _FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    if content > H:                                                       # полоса прокрутки
        hgt = max(20, int(H * H / content))
        top = int((H - hgt) * scroll / max(1, content - H))
        cv2.rectangle(img, (W - 3, top), (W - 1, top + hgt), (120, 120, 120), -1)
    cv2.line(img, (0, 0), (0, H), (90, 90, 90), 1)
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
