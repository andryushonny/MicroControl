"""Просмотр скана в окне OpenCV (лёгкий, для слабого ПК).

  мышь (тянуть любой кнопкой) / стрелки / WASD   перемещение (Shift — быстрее)
  двойной щелчок            этот участок — в центр
  колесо / + - / PgUp PgDn  масштаб
  0                         показать весь скан
  Esc                       выход

Мелкий масштаб собирается из превью (1/16), которые пишутся при скане;
крупный — из нужных кусков кадров. Поэтому окно открывается сразу и на тысячах кадров.
"""
from .samples import per_sample as _per_sample
import csv, json, math, os
from collections import OrderedDict
import cv2
import numpy as np
from .storage import ScanData, THUMB, meta_A
from .util import wheel_delta, open_window, window_size, exit_key, EXIT, set_icon
from . import ui as UI
from . import anomaly as _AD
from . import config as _cfg

WIN = "MicroControl scan"


class _Scene:
    def __init__(self, folder):
        self.st = ScanData(folder)
        self.H, self.W = self.st.shape[:2]
        A = meta_A(self.st.meta, self.W, self.H)
        p = np.array([[r['x_p'], r['y_p']] for r in self.st.index], float)
        P = -(p @ A.T)
        jp = os.path.join(folder, 'join', 'positions_all.csv')
        if os.path.exists(jp):                              # уточнённые позиции после join
            with open(jp, encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    P[int(r['id'])] = (float(r['x_px']), float(r['y_px']))
        self.P = P - P.min(0)
        self.size = self.P.max(0) + (self.W, self.H)
        self.thumbs = self.st.thumbs()
        self.cache, self.cache_b = OrderedDict(), 0         # (i, уровень) -> уменьшенный кадр; байт в кэше
        self.over, self.ok_ = None, 0.0                     # обзорная мозаика (join/preview_all.jpg)
        pv = os.path.join(folder, 'join', 'preview_all.jpg')
        if os.path.exists(pv) and (not os.path.exists(jp) or os.path.getmtime(pv) >= os.path.getmtime(jp)):
            img = cv2.imdecode(np.fromfile(pv, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self.over, self.ok_ = img, (img.shape[1] - 2) / self.size[0]

    def level_img(self, i, lv):
        key = (i, lv)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        f = 2 ** lv
        p = os.path.join(self.st.folder, 'viewer', 'levels', f"{i:05d}_{lv}.jpg")
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) if os.path.exists(p) else None
        if img is None:                                     # нет готового уровня — из кадра
            img = cv2.resize(self.st.frame(i), (max(1, self.W // f), max(1, self.H // f)), interpolation=cv2.INTER_AREA)
        self.cache[key] = img
        self.cache_b += img.nbytes
        while self.cache_b > 400e6 and len(self.cache) > 1:  # не больше ~400 МБ на старом ПК
            self.cache_b -= self.cache.popitem(last=False)[1].nbytes
        return img

    def render(self, cx, cy, s, Wv, Hv, fast=False):
        """fast — во время перетаскивания: из уменьшенных копий (плавно), затем — в полном качестве."""
        out = np.full((Hv, Wv, 3), 20, np.uint8)
        x0, y0 = cx - Wv / 2 / s, cy - Hv / 2 / s
        x1, y1 = x0 + Wv / s, y0 + Hv / s
        if self.over is not None and s <= self.ok_ * 1.05:  # мелкий масштаб: один вырез из обзора, а не тысячи кадров
            k, ov = self.ok_, self.over
            ax0, ay0 = max(0, int(x0 * k)), max(0, int(y0 * k))
            ax1, ay1 = min(ov.shape[1], int(math.ceil(x1 * k)) + 1), min(ov.shape[0], int(math.ceil(y1 * k)) + 1)
            if ax1 > ax0 and ay1 > ay0:
                ox, oy = int(round((ax0 / k - x0) * s)), int(round((ay0 / k - y0) * s))   # место выреза в окне
                dw, dh = max(1, int(round((ax1 - ax0) / k * s))), max(1, int(round((ay1 - ay0) / k * s)))
                part = cv2.resize(ov[ay0:ay1, ax0:ax1], (dw, dh), interpolation=cv2.INTER_AREA)
                u0, v0 = max(0, -ox), max(0, -oy)
                u1, v1 = min(dw, Wv - ox), min(dh, Hv - oy)
                if u1 > u0 and v1 > v0:
                    out[oy + v0:oy + v1, ox + u0:ox + u1] = part[v0:v1, u0:u1]
            return out
        P, W, H = self.P, self.W, self.H
        idx = np.flatnonzero((P[:, 0] < x1) & (P[:, 0] + W > x0) & (P[:, 1] < y1) & (P[:, 1] + H > y0))
        lv = 0 if s >= 1 else min(int(math.floor(math.log2(1 / s))), 4)   # 4 = превью 1/16
        if fast:
            lv = max(lv, 2 if s >= 0.25 else 4)
        for i in idx:
            px, py = P[i]
            ax0, ay0, ax1, ay1 = max(x0, px), max(y0, py), min(x1, px + W), min(y1, py + H)
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            if lv >= 4:
                src, k = self.thumbs[i], self.thumbs[i].shape[1] / W
            elif lv == 0:
                src, k = None, 1.0
            else:
                src = self.level_img(i, lv)
                k = src.shape[1] / W
            sx0, sy0 = int((ax0 - px) * k), int((ay0 - py) * k)
            sx1, sy1 = max(sx0 + 1, int(math.ceil((ax1 - px) * k))), max(sy0 + 1, int(math.ceil((ay1 - py) * k)))
            crop = self.st.frame(i, (sx0, sy0, sx1, sy1)) if src is None else src[sy0:sy1, sx0:sx1]
            dx0, dy0 = int(round((ax0 - x0) * s)), int(round((ay0 - y0) * s))
            dx1, dy1 = int(round((ax1 - x0) * s)), int(round((ay1 - y0) * s))
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            crop = cv2.resize(np.ascontiguousarray(crop), (dx1 - dx0, dy1 - dy0),
                              interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            out[dy0:dy1, dx0:dx1] = crop[:dy1 - dy0, :dx1 - dx0]
        return out


HELP_V2 = "Drag: move  Wheel: zoom  0: whole  R: straight  , / .: sample  H: all keys  Shift+Esc: exit"
HELP_MODE = {
    'look': "G: mark crosses  F: fit grid  S: select  B: brush  P: polygon  N: number squares  Y: hints  E: export",
    'cross': "Click: cross (2 in a row = tilt)  Right click: remove  Backspace: undo  C: start again  F: fit grid  Esc: done",
    'select': "Click: square  Drag: box  Shift: unselect  Right click: remove green area  A: all  X: none  Esc: done",
    'paint': "Drag: paint  Shift+drag: erase  A: all  X: none  Esc: done",
    'poly': "Click: corner  Enter: select inside  Del: unselect inside  Backspace: undo corner  Esc: done",
    'label': "Click a square, type its number X Y  Enter: set  Esc: done",
}
HELP_HINTS = "Y: hints on / off  Shift+Y: select them  < / >: sensitivity  U: type it"
HELP_V = ("G: 2 crosses in a row = tilt (right click = remove, C = again)   F: fit grid   S: select (click, box; Shift = "
          "unselect; right click = remove the green area)   B: brush   P: polygon   N: number squares (X Y)   Y: hints "
          "(yellow), Shift+Y: select them, U, < / >: sensitivity   A / X: all / none   E: export   Esc: exit")
HELP_ALL = [
    ("Look", "Drag (right / middle in any mode): move   Wheel, + / -: zoom   0: whole   Arrows, W A S D: move"),
    ("", "R: straight / as shot   , / .: previous / next sample   H: this help   Esc: leave the mode"),
    ("Exit", "Shift+Esc (or Ctrl+Esc), or close the window"),
    ("Grid", "G: mark crosses: click two neighbouring crosses in one row = tilt; more crosses far apart = more exact"),
    ("", "Right click: remove a cross   Backspace: undo   C: start again   F: fit the grid to all crosses"),
    ("Numbers", "N: click a square, type the number written next to it (X Y), Enter. Two squares fix all numbers"),
    ("Select", "S: click / drag a box (Shift: unselect)   B: brush   P: polygon   A / X: all / none"),
    ("", "Right click on green: remove that green area"),
    ("Hints", "Y: yellow hints on / off   Shift+Y or Ctrl+Y: select them   < / >: sensitivity -/+ 5   U: type it"),
    ("Export", "E: export the selected squares (PNG + DXF)"),
]


_PREV = {}


def _preview(folder, w):
    """Картинка образца для карусели: обзорная мозаика (после сшивки) или первый кадр."""
    pv = os.path.join(folder, 'join', 'preview_all.jpg')
    key = (folder, os.path.getmtime(pv) if os.path.exists(pv) else 0, w)
    if key not in _PREV:
        img = cv2.imdecode(np.fromfile(pv, np.uint8), cv2.IMREAD_COLOR) if os.path.exists(pv) else None
        if img is None:
            try:
                st_ = ScanData(folder)
                img = st_.thumbs()[0] if len(st_) else None
                st_.close()
            except Exception:
                img = None
        if img is None:
            img = np.full((60, w, 3), 40, np.uint8)
        k = min(w / img.shape[1], 130 / img.shape[0])
        _PREV[key] = cv2.resize(img, (max(1, int(img.shape[1] * k)), max(1, int(img.shape[0] * k))),
                                interpolation=cv2.INTER_AREA)
    return _PREV[key]


def _car_items(subs, cur, car):
    from . import samples as SM
    items = []
    for k, f in enumerate(subs):
        lines = []
        try:
            n = sum(1 for _ in open(os.path.join(f, 'index.csv'), encoding='utf-8')) - 1
            lines.append(f"{n} frames" + ("" if os.path.exists(os.path.join(f, 'join', 'positions_all.csv'))
                                          else "   not stitched"))
        except Exception:
            pass
        gp = os.path.join(f, 'grid.json')
        if os.path.exists(gp):
            try:
                d = json.load(open(gp, encoding='utf-8'))
                lines.append(f"selected {len(d.get('selected', []))}   hints {len(d.get('useful', []))}")
            except Exception:
                pass
        else:
            lines.append("no grid yet (G)")
        items.append(dict(title=f"Sample {SM.sample_index(f)}", color=SM.sample_color(f),
                          lines=lines, thumbs=[_preview(f, SM.CAR_W - 36)], confirm=car.get('confirm') == f, go=False))
    return items


def view(folder, fullscreen=False, sensitivity=None, _keys=None):
    """Открыть сшитый скан (или папку сеанса из нескольких образцов). Справа — карусель образцов:
    щелчок — открыть образец, x — убрать образец (папка уходит в <сеанс>/_removed/), , / . — предыдущий / следующий."""
    from . import samples as SM
    from .manual import screen_size
    ses, subs = SM.session_of(folder)
    cur = subs[0] if (ses and os.path.abspath(folder) == ses) else os.path.abspath(folder)
    if not subs:
        raise FileNotFoundError(f"No samples in {folder}")
    keys = iter(_keys) if _keys is not None else None
    Wv, Hv = open_window(WIN, fullscreen) or screen_size()
    car = dict(scroll=0, confirm=None, size=(Wv, Hv))
    while cur:
        cur = _view_one(cur, fullscreen, sensitivity, keys, car)
    if not fullscreen:
        window_size(WIN, remember=True)                                    # запомнить размер для следующего окна
    cv2.destroyWindow(WIN)
    cv2.waitKey(1)


def _view_one(folder, fullscreen, sensitivity, keys, car):
    """Открыть сшитый скан. Перемещение — левая кнопка (в режимах выделения — правая или средняя), колесо — масштаб.
    G — отметить кресты (щелчок у креста: центр находится сам по симметрии креста); соседний крест по ряду,
    по столбцу и дальние — решётка уточняется. R — повернуть картинку так, чтобы квадраты стали ровными.
    S — выделять квадраты: щелчок — квадрат, рамка — много сразу (Shift — снять); P — многоугольник:
    щелчки — вершины, Enter — выделить внутри, Del — снять внутри. A — все, X — ни одного.
    E — вывести выделенное (mc.export). Разметка хранится в <скан>/grid.json."""
    from .mask import Grid, snap_cross, export, sens_frac
    from . import samples as SM
    sc = _Scene(folder)
    g = Grid(folder)
    if sensitivity is not None:
        g.sensitivity = float(min(100, max(0, sensitivity)))
    px_um = sc.st.meta.get('pixel_size_um') or 1.0
    ses, subs = SM.session_of(folder)
    Wv, Hv = car['size']
    st = dict(cx=sc.size[0] / 2, cy=sc.size[1] / 2, s=None, rot=0.0, mode='look', down=None, box=None,
              poly=[], dirty=True, Wv=Wv - SM.CAR_W, Hv=Hv, straight=False, switch=None,
              msg=(f"Sample {SM.sample_index(folder)} ({subs.index(folder) + 1} of {len(subs)})" if len(subs) > 1 else ""))

    def car_click(x, y):
        h_ = SM.hit(car.get('hits', []), x - st['Wv'], y)
        if not h_:
            return
        act, k = h_
        f = subs[k]
        if act == 'sel':
            car['confirm'] = None
            if f != folder:
                st['switch'] = f
        elif act == 'del':
            if car.get('confirm') == f:
                car['confirm'] = None
                g.save(px_um) if f == folder else None
                if f == folder:
                    sc.st.close()
                dst = SM.remove_sample(f)
                rest = [s_ for s_ in subs if s_ != f]
                st['msg'] = f"Sample {SM.sample_index(dst)} removed to {dst}"
                st['switch'] = (folder if f != folder else (rest[min(k, len(rest) - 1)] if rest else False))
            else:
                car['confirm'] = f
                st['msg'] = f"Click x again to remove sample {SM.sample_index(f)} (its folder goes to _removed/)"
        st['dirty'] = True

    def R():
        t = np.radians(st['rot'])
        return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    mid = lambda: np.array([st['Wv'] / 2, st['Hv'] / 2])
    to_view = lambda M: mid() + st['s'] * (R() @ (np.asarray(M, float) - (st['cx'], st['cy'])))
    from_view = lambda v: np.array([st['cx'], st['cy']]) + R().T @ ((np.asarray(v, float) - mid()) / st['s'])

    def fit():
        st['s'] = min(st['Wv'] / sc.size[0], st['Hv'] / sc.size[1]) * 0.95
        st['cx'], st['cy'] = sc.size[0] / 2, sc.size[1] / 2

    def zoom(f, x=None, y=None):
        v = np.array([st['Wv'] / 2 if x is None else x, st['Hv'] / 2 if y is None else y], float)
        Mc = from_view(v)
        st['s'] = min(8.0, max(1e-4, st['s'] * f))
        c = Mc - R().T @ ((v - mid()) / st['s'])
        st['cx'], st['cy'] = c

    def set_straight(on):
        st['straight'] = on and g.ready()
        if st['straight']:
            th = np.degrees(np.arctan2(*g.cr.vectors()[0][::-1]))
            st['rot'] = -(((th + 90) % 180) - 90)
        else:
            st['rot'] = 0.0

    def add_cross(v):
        Mc = from_view(v)
        r = int(min(150, max(8, 15.0 / px_um)))
        reg = sc.render(Mc[0], Mc[1], 1.0, 4 * r + 1, 4 * r + 1)
        d, resp = snap_cross(cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY), r, arm=g.arm)
        try:
            ij = g.cr.add(Mc + d)
            g.refined = None                                # решётка — по отметкам (квадратная)
            if len(g.cr.marks) == 1:
                st['msg'] = "Cross 1 marked. Now click its neighbour in the same row"
            else:
                v1, _ = g.cr.vectors()
                st['msg'] = (f"{len(g.cr.marks)} crosses: tilt {g.cr.angle():+.3f} deg, pitch "
                             f"{np.hypot(*v1) * px_um:.2f} um. Picture turned level (R: as shot). "
                             f"A far cross in the row makes the tilt more exact")
                if len(g.cr.marks) == 2:                    # два креста: прямая наклона -> полотно докручено
                    set_straight(True)
                if _cfg.AUTO_HINTS or (getattr(g, 'scores', None) and st.get('hints')):   # подсказки — по Y
                    hints()
            g.save(px_um)
            if st['straight']:
                set_straight(True)
        except ValueError as e:
            st['msg'] = str(e)

    def hints():
        """Жёлтая подсказка: квадраты, где что-то есть (mask.Grid.find_useful) — по уменьшенной мозаике."""
        g.find_useful(sc.render, sc.size, model=_AD.default_model(sc.st.meta.get('objective')))
        st['hints'] = True

    def set_sens(S):
        if not getattr(g, 'scores', None):
            hints()
        g.pick_useful(S); g.save(px_um); st['hints'] = True
        st['msg'] = (f"Sensitivity {g.sensitivity:g}: {len(g.useful)} squares in yellow "
                     f"(something covers > {sens_frac(g.sensitivity) * 100:.2g} % of the square)")

    def apply_poly(poly_view, on, touch=False):
        pts = [from_view(v) for v in poly_view]
        cells = g.cells_touching(pts) if touch else g.cells_in(pts, sc.size)
        if on:
            g.selected |= cells
        else:
            g.selected -= cells
        g.save(px_um)
        st['msg'] = f"{'Selected' if on else 'Unselected'} {len(cells)} squares, total {len(g.selected)}"

    def mouse(ev, x, y, flags, _=None):
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        if x >= st['Wv'] and not st['down']:              # карусель образцов справа
            if ev == cv2.EVENT_MOUSEMOVE:                   # наведение: подсветка кнопок и карточек
                h_ = SM.hit(car['hits'], x - st['Wv'], y)
                if h_ != car.get('hov'):
                    car['hov'], car['hxy'], st['car_dirty'] = h_, (x - st['Wv'], y), True
                return
            if ev == cv2.EVENT_LBUTTONUP:
                car_click(x, y)
            elif ev == cv2.EVENT_MOUSEWHEEL:
                car['scroll'] = int(max(0, min(car.get('content', 0) - st['Hv'] + 20,
                                               car['scroll'] - 60 * np.sign(wheel_delta(flags)))))
                st['dirty'] = True
            return
        if car.get('hov') is not None:                      # мышь ушла с карусели
            car['hov'], car['hxy'], st['car_dirty'] = None, None, True
        if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            st['down'] = (x, y, st['cx'], st['cy'], ev, shift, False)
            st['ctrl'] = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
        elif ev == cv2.EVENT_MOUSEMOVE and st['down']:
            x0, y0, cx, cy, ev0, sh0, moved = st['down']
            if abs(x - x0) + abs(y - y0) > 4:
                st['down'] = (x0, y0, cx, cy, ev0, sh0, True)
                # сдвиг: правой / средней кнопкой — в любом режиме; левой — везде, кроме рамки (S) и кисти (B),
                # а там — с Ctrl
                pan = ev0 != cv2.EVENT_LBUTTONDOWN or st['mode'] in ('look', 'cross', 'poly', 'label') or st.get('ctrl')
                if st['mode'] == 'paint' and ev0 == cv2.EVENT_LBUTTONDOWN and g.ready() and not st.get('ctrl'):
                    for t_ in np.linspace(0, 1, 8):          # мазок: все квадраты на отрезке от прошлой точки
                        q_ = np.array(st.get('last', (x0, y0)), float) * (1 - t_) + np.array([x, y], float) * t_
                        c_ = g.cell_at(from_view(q_))
                        (g.selected.discard if sh0 else g.selected.add)(c_)
                    st['last'] = (x, y)
                    st['dirty'] = True
                    return
                if pan:
                    c = np.array([cx, cy]) - R().T @ (np.array([x - x0, y - y0], float) / st['s'])
                    st['cx'], st['cy'] = c
                elif st['mode'] == 'select':
                    st['box'] = (x0, y0, x, y, sh0)
                st['dirty'] = True
        elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP) and st['down']:
            x0, y0, cx, cy, ev0, sh0, moved = st['down']
            st['down'] = None
            if st['box'] is not None:
                bx0, by0, bx1, by1, shb = st['box']
                st['box'] = None
                if g.ready():                               # рамка берёт всё, чего касается
                    apply_poly([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)], not shb, touch=True)
            elif st['mode'] == 'paint' and ev0 == cv2.EVENT_LBUTTONDOWN and not (moved and st.get('ctrl')):
                if not moved and g.ready():
                    c_ = g.cell_at(from_view((x, y)))
                    (g.selected.discard if sh0 else g.selected.add)(c_)
                st.pop('last', None)
                g.save(px_um)
                st['msg'] = f"Selected {len(g.selected)} squares"
            elif not moved and ev0 == cv2.EVENT_LBUTTONDOWN:
                if st['mode'] == 'cross':
                    add_cross((x, y))
                elif st['mode'] == 'select' and g.ready():
                    c = g.cell_at(from_view((x, y)))
                    g.selected.symmetric_difference_update({c})
                    g.save(px_um)
                    st['msg'] = f"Square {c} {'selected' if c in g.selected else 'unselected'}, total {len(g.selected)}"
                elif st['mode'] == 'poly':
                    st['poly'].append((x, y))
                elif st['mode'] == 'label' and g.ready():
                    c = g.cell_at(from_view((x, y)))
                    X, Y = g.labels.get(c, g.name(*c))
                    st['lab'] = [c, ""]
                    st['msg'] = f"Square {c}: type its number X Y (now {X} {Y}), Enter = keep, Esc = cancel"
            elif not moved and ev0 == cv2.EVENT_RBUTTONDOWN:
                if st['mode'] == 'cross' and g.cr.marks:
                    dists = [np.hypot(*(to_view(q) - (x, y))) for q in g.cr.marks]
                    k = int(np.argmin(dists))
                    if dists[k] < 25:
                        g.cr.remove(k)
                        g.save(px_um)
                        st['msg'] = "Cross mark removed"
                elif st['mode'] == 'poly' and st['poly']:
                    st['poly'].pop()
                elif st['mode'] in ('look', 'select', 'paint', 'poly') and g.ready() and g.selected:
                    c = g.cell_at(from_view((x, y)))        # правый щелчок по зелёному — убрать всю эту область
                    if c in g.selected:
                        area, todo = set(), [c]
                        while todo:
                            q = todo.pop()
                            if q in g.selected and q not in area:
                                area.add(q)
                                todo += [(q[0] + 1, q[1]), (q[0] - 1, q[1]), (q[0], q[1] + 1), (q[0], q[1] - 1)]
                        g.selected -= area
                        g.save(px_um)
                        st['msg'] = f"Area of {len(area)} squares removed, {len(g.selected)} left"
                elif st['mode'] == 'label' and g.ready():
                    c = g.cell_at(from_view((x, y)))
                    if g.labels.pop(c, None) is not None:
                        g.save(px_um)
                        st['msg'] = f"Label removed. {g.numbering()[2]}"
            st['dirty'] = True
        elif ev == cv2.EVENT_LBUTTONDBLCLK and st['mode'] == 'look':
            c = from_view((x, y))
            st['cx'], st['cy'] = c
            st['dirty'] = True
        elif ev == cv2.EVENT_MOUSEWHEEL:
            zoom(1.25 if wheel_delta(flags) > 0 else 0.8, x, y)
            st['dirty'] = True

    def render():
        Wv, Hv = st['Wv'], st['Hv']
        if abs(st['rot']) < 1e-6:
            img = sc.render(st['cx'], st['cy'], st['s'], Wv, Hv, fast=st['down'] is not None)
        else:                                               # повёрнутый вид: квадрат с запасом, затем поворот
            D = int(math.hypot(Wv, Hv)) + 2
            big = sc.render(st['cx'], st['cy'], st['s'], D, D, fast=st['down'] is not None)
            Rm = R()
            t = mid() - Rm @ np.array([D / 2, D / 2])
            img = cv2.warpAffine(big, np.float32(np.c_[Rm, t]), (Wv, Hv), flags=cv2.INTER_LINEAR)
        red, green, yellow = (40, 40, 255), (60, 200, 60), (0, 220, 255)
        base_, void_ = img.copy(), (img <= 22).all(2)                # вне мозаики сетку и заливку не рисуем
        if g.ready():
            lo = np.min([from_view(v) for v in ((0, 0), (Wv, 0), (0, Hv), (Wv, Hv))], 0)
            hi = np.max([from_view(v) for v in ((0, 0), (Wv, 0), (0, Hv), (Wv, Hv))], 0)
            v1, v2 = g.cr.vectors()
            step = min(np.hypot(*v1), np.hypot(*v2)) * st['s']
            lo, hi = np.maximum(lo, 0), np.minimum(hi, sc.size)            # решётка — только над мозаикой
            e_ = 0.25 * min(np.hypot(*v1), np.hypot(*v2))                   # квадрат почти весь над мозаикой
            cells = [c for c in g.cells_range(lo, hi)                       # (до ~40 тыс. квадратов на экране)
                     if (lambda q: e_ <= q[0] <= sc.size[0] - e_ and e_ <= q[1] <= sc.size[1] - e_)(
                         g.cr.point(c[0] + 0.5, c[1] + 0.5))] if step >= 1.5 else []
            if cells and (g.selected or (st.get('hints', _cfg.AUTO_HINTS) and g.useful)):   # зелёные — выделены,
                ov = img.copy()                                                   # жёлтые — подсказка
                for c in cells:
                    if c in g.selected:
                        cv2.fillPoly(ov, [np.int32([to_view(q) for q in g.corners(*c)])], green)
                    elif st.get('hints', _cfg.AUTO_HINTS) and c in g.useful:
                        cv2.fillPoly(ov, [np.int32([to_view(q) for q in g.corners(*c)])], (0, 215, 255))
                img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
            if step >= 34 and len(cells) <= 3000:           # номер квадрата в уголке: (X, Y) по подписям
                num = g.numbering()
                fs = min(0.5, step / 160)
                for c in cells:
                    pts = np.array([to_view(q) for q in g.corners(*c)])
                    u, v = pts[np.argmin(pts.sum(1))] + (4, 4 + 14 * fs / 0.5)
                    txt = "{},{}".format(*g.name(*c, num))
                    col = (0, 230, 255) if c in g.labels else ((255, 255, 255) if g.labels else (170, 170, 170))
                    cv2.putText(img, txt, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, txt, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
                if st.get('lab'):                           # вводимый номер — у квадрата
                    pts = np.array([to_view(q) for q in g.corners(*st['lab'][0])])
                    cv2.polylines(img, [np.int32(pts)], True, (0, 230, 255), 2, cv2.LINE_AA)
            if cells and step >= 7:                         # крупно: контур каждого квадрата над мозаикой
                polys = [np.int32(np.round([to_view(q) for q in g.corners(*c)])) for c in cells]
                cv2.polylines(img, polys, True, red, 1, cv2.LINE_AA if step >= 30 else cv2.LINE_8)
            elif cells:                                     # мелко: линии через кресты
                # сплошные линии без сглаживания (сглаженная линия в 1 px на дробной позиции бледнеет),
                # совсем мелко — каждая k-я, чтобы не слиться в сплошной красный
                k_ = max(1, int(math.ceil(6 / step)))
                I = sorted({c[0] for c in cells} | {cells[-1][0] + 1})
                J = sorted({c[1] for c in cells} | {cells[-1][1] + 1})
                for i in I:
                    if i % k_ == 0:
                        a_, b_ = to_view(g.cr.point(i, J[0])), to_view(g.cr.point(i, J[-1]))
                        cv2.line(img, tuple(np.int32(np.round(a_))), tuple(np.int32(np.round(b_))), red, 1, cv2.LINE_8)
                for j in J:
                    if j % k_ == 0:
                        a_, b_ = to_view(g.cr.point(I[0], j)), to_view(g.cr.point(I[-1], j))
                        cv2.line(img, tuple(np.int32(np.round(a_))), tuple(np.int32(np.round(b_))), red, 1, cv2.LINE_8)
            elif g.selected:
                st['msg'] = st['msg'] if 'zoom in' in st['msg'] else st['msg']
        img[void_] = base_[void_]
        for q in g.cr.marks:
            cv2.circle(img, tuple(np.int32(to_view(q))), 6, red, 1, cv2.LINE_AA)
        if st['poly']:
            pts = np.int32(st['poly'] + ([st['hover']] if st.get('hover') else []))
            cv2.polylines(img, [pts], False, yellow, 1, cv2.LINE_AA)
        if st['box']:
            bx0, by0, bx1, by1, shb = st['box']
            cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 140, 255) if shb else yellow, 1)
        mode = {'look': 'Look', 'cross': 'Mark crosses', 'select': 'Select', 'poly': 'Polygon',
                'paint': 'Brush', 'label': 'Number squares'}[st['mode']]
        mcol = {'look': (170, 170, 178), 'cross': (90, 90, 255), 'select': (90, 210, 90), 'poly': (90, 210, 90),
                'paint': (90, 210, 90), 'label': UI.ACCENT}[st['mode']]
        if st.get('lab'):
            st['msg'] = f"Square number X Y: {st['lab'][1]}_   (Enter = set, Esc = cancel)"
        if st.get('sens') is not None:
            st['msg'] = f"Hint sensitivity 0-100 (now {g.sensitivity:g}, more = more yellow): {st['sens']}_   Enter / Esc"
        um = px_um / st['s']
        fs = 0.5
        th_ = UI.size("Xg", fs)[1]
        bh = th_ + 22                                       # высота верхней и нижней полос
        UI.panel(img, 0, 0, Wv, bh, 0.82)                   # верх: режим и главное о скане
        x_ = UI.chip(img, mode, 10, bh - 11, fs, mcol) + 14
        it = [f"{len(sc.st)} frames", f"{um:.2f} um / px"]
        if g.ready():
            it += [f"pitch {np.hypot(*g.cr.vectors()[0]) * px_um:.1f} um", f"tilt {g.cr.angle():+.2f} deg"]
        else:
            it += ["no grid yet: G"]
        it += [f"selected {len(g.selected)}"]
        if g.useful and st.get('hints', _cfg.AUTO_HINTS):
            it += [f"hints {len(g.useful)} (sensitivity {g.sensitivity:g})"]
        UI.facts(img, it, x_, bh - 11, fs)
        if st['straight']:
            w_ = UI.size("Straight", fs)[0] + 18
            UI.chip(img, "Straight", Wv - w_ - 10, bh - 11, fs, (235, 170, 60))
        hl = [HELP_MODE[st['mode']]] + ([HELP_HINTS] if g.useful and st['mode'] == 'look' else []) + [HELP_V2]
        hb = len(hl) * (th_ + 12) + 12
        UI.panel(img, 0, Hv - hb, Wv, Hv, 0.82)             # низ: клавиши этого режима
        for k_, h_ in enumerate(hl):
            UI.keys_line(img, h_, 12, Hv - hb + 6 + (k_ + 1) * (th_ + 12) - 4, fs, Wv - 10)
        import time as _t
        if st['msg'] != st.get('_lm'):
            st['_lm'], st['_lt'] = st['msg'], _t.monotonic()
        prompt = bool(st.get('lab')) or st.get('sens') is not None
        if st['msg'] and (prompt or _t.monotonic() - st['_lt'] < 8):
            low = st['msg'].lower()
            kind = 'err' if (st['msg'].startswith('[E') or 'no ' in low[:4] or 'not ' in low or 'disagree' in low) \
                else ('info' if st.get('lab') or st.get('sens') is not None else 'ok')
            UI.toast(img, st['msg'], Wv / 2, Hv - hb - 18, 0.55, kind, Wv - 40)
        if st.get('help'):                                  # H: все клавиши
            fsh = 0.52
            lh = UI.size("Xg", fsh)[1] + 14
            wl = max(UI.size(a_, fsh, 2)[0] for a_, _b in HELP_ALL) + 24
            wh = min(Wv - 60, wl + max(UI.size(b_, fsh)[0] for _a, b_ in HELP_ALL) + 40)
            hh = lh * len(HELP_ALL) + 50
            x0_, y0_ = int((Wv - wh) / 2), int((Hv - hh) / 2)
            UI.panel(img, x0_, y0_, x0_ + wh, y0_ + hh, 0.92, UI.LINE)
            UI.put(img, "Keys", x0_ + 18, y0_ + 30, 0.6, UI.ACCENT, 2)
            for k_, (a_, b_) in enumerate(HELP_ALL):
                yy = y0_ + 30 + (k_ + 1) * lh
                UI.put(img, a_, x0_ + 18, yy, fsh, UI.ACCENT, 1)
                UI.put(img, b_, x0_ + 18 + wl, yy, fsh, UI.TEXT)
        st['scene'] = img
        return np.hstack([img, carousel_img()])

    def carousel_img():
        pan_, car['hits'], car['content'] = SM.carousel(st['Hv'], _car_items(subs, folder, car), subs.index(folder),
                                                        car['scroll'], title="Samples", go=False, hover=car.get('hxy'))
        return pan_

    cv2.setMouseCallback(WIN, mouse)
    while True:
        if st['switch'] is not None:                      # другой образец из карусели
            break
        if not fullscreen:
            W_, H_ = window_size(WIN, (st['Wv'] + SM.CAR_W, st['Hv']))
            if W_ >= 100 + SM.CAR_W and H_ >= 100 and (W_ - SM.CAR_W, H_) != (st['Wv'], st['Hv']):
                st['Wv'], st['Hv'], st['dirty'] = W_ - SM.CAR_W, H_, True
                car['size'] = (W_, H_)
        if st['s'] is None:
            fit()
        if st['dirty']:
            cv2.imshow(WIN, render())
            st['dirty'] = st['car_dirty'] = False
            if not st.get('icon'):                          # окно уже на экране: значок программы (Windows)
                st['icon'] = True
                set_icon(WIN)
        elif st.get('car_dirty') and st.get('scene') is not None:   # только карусель (наведение мыши)
            cv2.imshow(WIN, np.hstack([st['scene'], carousel_img()]))
            st['car_dirty'] = False
        key = next(keys, EXIT) if keys is not None else cv2.waitKeyEx(8 if st['down'] else 20)
        if keys is not None:
            cv2.waitKey(1)
            if isinstance(key, tuple):                      # тест: ('click'|'rclick', x, y[, shift]) или ('drag', ...)
                kind, *a = key
                fl = cv2.EVENT_FLAG_SHIFTKEY if (len(a) > 2 and a[-1] == 'shift') else 0
                if kind in ('click', 'rclick'):
                    dn, up = ((cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP) if kind == 'click' else
                              (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONUP))
                    mouse(dn, a[0], a[1], fl); mouse(up, a[0], a[1], fl)
                elif kind == 'drag':
                    mouse(cv2.EVENT_LBUTTONDOWN, a[0], a[1], fl)
                    mouse(cv2.EVENT_MOUSEMOVE, a[2], a[3], fl | cv2.EVENT_FLAG_LBUTTON)
                    mouse(cv2.EVENT_LBUTTONUP, a[2], a[3], fl)
                key = -1
        if key == -1:
            if keys is None and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                st['switch'] = False
                break
            continue
        if exit_key(key):                                   # Shift+Esc / Ctrl+Esc — выйти из просмотра
            st['switch'] = False
            break
        if st.get('sens') is not None:                      # ввод чувствительности 0…100
            if 48 <= key <= 57 and len(st['sens']) < 3:
                st['sens'] += chr(key)
            elif key == 8:
                st['sens'] = st['sens'][:-1]
            elif key in (13, 10):
                if st['sens']:
                    set_sens(int(st['sens']))
                st['sens'] = None
            elif key == 27:
                st['sens'] = None; st['msg'] = "Cancelled"
            st['dirty'] = True
            continue
        if st.get('lab'):                                   # ввод номера квадрата: цифры, пробел / запятая, минус
            c, buf = st['lab']
            if 48 <= key <= 57 or (key == 45 and (not buf or buf[-1] == ' ')):
                st['lab'][1] = buf + chr(key)
            elif key in (32, 44) and buf and buf[-1] != ' ' and ' ' not in buf:
                st['lab'][1] = buf + ' '
            elif key == 8:
                st['lab'][1] = buf[:-1]
            elif key == 27:
                st['lab'] = None; st['msg'] = "Cancelled"
            elif key in (13, 10):
                parts = buf.split()
                if not buf:
                    g.labels[c] = g.name(*c)                # Enter без цифр — подтвердить показанный номер
                elif len(parts) == 2 and all(p_.lstrip('-').isdigit() for p_ in parts):
                    g.labels[c] = (int(parts[0]), int(parts[1]))
                else:
                    st['msg'] = "Two numbers: X Y"; st['dirty'] = True; continue
                g.save(px_um)
                st['lab'] = None
                st['msg'] = f"Square {c} = {g.labels[c]}.   {g.numbering()[2]}"
            st['dirty'] = True
            continue
        ch = chr(key & 0xFF) if 0 <= key < 256 else ''
        raw = ch
        step = (0.8 if ch in 'WASD' and ch else 0.25) * min(st['Wv'], st['Hv']) / st['s']
        ch = ch.lower()
        if key == 27:
            if st.get('help'):
                st['help'] = False
            elif st['poly']:
                st['poly'] = []
            elif st['mode'] != 'look':
                st['mode'] = 'look'
            else:
                st['msg'] = "Exit: Shift+Esc or close the window"
            st['dirty'] = True
        elif raw in (',', '.') and len(subs) > 1:           # предыдущий / следующий образец
            st['switch'] = subs[(subs.index(folder) + (1 if raw == '.' else -1)) % len(subs)]
        elif ch == 'c' and st['mode'] == 'cross':           # кресты заново
            g.cr.clear(); g.refined = None; set_straight(False); g.save(px_um)
            st['msg'] = "Crosses cleared. Click a cross, then its neighbour in the same row"
        elif ch == 'g':
            st['mode'] = 'cross'
            st['msg'] = "Click a cross, then its neighbour in the same row: the picture turns level. C: start again"
        elif ch == 's' and not raw.isupper():
            st['mode'] = 'select' if g.ready() else st['mode']
            st['msg'] = "Click = square, drag = box (Shift = unselect), right button = move" if g.ready() \
                else "Mark at least 2 crosses first (G)"
        elif ch == 'b':
            st['mode'] = 'paint' if g.ready() else st['mode']
            st['msg'] = "Brush: drag over the squares (Shift = rub out), right button = move" if g.ready() \
                else "Mark at least 2 crosses first (G)"
        elif key == 25 or raw == 'Y':                       # Ctrl+Y / Shift+Y: жёлтое -> зелёное
            if g.ready():
                if not g.useful and not getattr(g, 'scores', None):
                    hints()
                n0 = len(g.selected)
                g.selected |= set(g.useful)
                g.save(px_um)
                st['mode'] = 'paint'                        # сразу кисть: поправить выделение руками
                st['msg'] = f"Hints selected: +{len(g.selected) - n0} squares, total {len(g.selected)}. " \
                            "Brush: drag to add, Shift+drag to erase"
        elif ch == 'y' and g.ready():                       # Y: подсказка вкл. / выкл. (первый раз — посчитать)
            if not st.get('hints', _cfg.AUTO_HINTS) and not getattr(g, 'scores', None):
                st['msg'] = "Looking for squares with something in them..."
                cv2.imshow(WIN, render()); cv2.waitKey(1)
                hints(); g.save(px_um)
            else:
                st['hints'] = not st.get('hints', _cfg.AUTO_HINTS)
            st['msg'] = (f"Hints: {len(g.useful)} squares in yellow, sensitivity {g.sensitivity:g}. Shift+Y or Ctrl+Y: "
                         f"select them; U: type the sensitivity 0-100, < / >: -5 / +5") if st.get('hints', _cfg.AUTO_HINTS) \
                else "Hints hidden"
        elif raw in ('<', '>') and g.ready():                # чувствительность подсказки -5 / +5
            set_sens(g.sensitivity + (5 if raw in '.>' else -5))
        elif ch == 'u' and g.ready():                       # ввести чувствительность числом
            st['sens'] = ""
        elif ch == 'n':
            st['mode'] = 'label' if g.ready() else st['mode']
            st['msg'] = ("Click a square with a number written on it and type the number X Y. Two squares in another "
                         "row and column fix the directions; right click = remove a label") if g.ready() \
                else "Mark at least 2 crosses first (G)"
        elif ch == 'p':
            st['mode'] = 'poly' if g.ready() else st['mode']
            st['poly'] = []
            st['msg'] = "Click the corners, Enter = select inside, Del = unselect inside, right click = undo a corner"
        elif key in (13, 10) and st['mode'] == 'poly' and len(st['poly']) >= 3:
            apply_poly(st['poly'], True); st['poly'] = []
        elif key in (3014656, 65535, 127) and st['mode'] == 'poly' and len(st['poly']) >= 3:
            apply_poly(st['poly'], False); st['poly'] = []
        elif key == 8:
            if st['mode'] == 'poly' and st['poly']:
                st['poly'].pop()
            elif st['mode'] == 'cross':
                g.cr.undo(); g.save(px_um); st['msg'] = "Last cross mark removed"
        elif ch == 'f' and g.ready():
            n_, rms = g.refine(lambda x, y, w, h: sc.render(x, y, 1.0, w, h), sc.size, px_um)
            if _cfg.AUTO_HINTS or (getattr(g, 'scores', None) and st.get('hints')):
                hints()
            g.save(px_um)
            st['msg'] = f"Grid fitted to all {n_} crosses" + (f": {rms:.2f} px off" if rms is not None else ": too few") + \
                f"   hints: {len(g.useful)} squares (Y)"
        elif ch == 'h':
            st['help'] = not st.get('help')
        elif ch == 'r':
            set_straight(not st['straight'])
            st['msg'] = "Straight: squares stand level" if st['straight'] else "Picture as shot"
        elif ch == 'a' and st['mode'] in ('select', 'poly', 'paint') and g.ready():
            W_, H_ = sc.size
            g.selected |= g.cells_in([(0, 0), (W_, 0), (W_, H_), (0, H_)], sc.size)
            g.save(px_um); st['msg'] = f"All {len(g.selected)} squares selected"
        elif ch == 'x' and st['mode'] in ('select', 'poly', 'paint'):
            g.selected.clear(); g.save(px_um); st['msg'] = "Selection cleared"
        elif ch == 'e':
            try:
                r = export(folder)
                st['msg'] = f"Exported {r['Squares']} squares: {r['File']}"
            except Exception as e:
                st['msg'] = f"Export: {e}"
        elif ch and ch in '+=' or key in {2162688, 65365}:
            zoom(1.25)
        elif ch and ch in '-_' or key in {2228224, 65366}:
            zoom(0.8)
        elif ch == '0':
            fit()
        elif key in {2424832, 65361} or (ch == 'a' and st['mode'] in ('look', 'cross')):
            st['cx'] -= step
        elif key in {2555904, 65363} or ch == 'd':
            st['cx'] += step
        elif key in {2490368, 65362} or ch == 'w':
            st['cy'] -= step
        elif key in {2621440, 65364} or (ch == 's' and raw.isupper()):
            st['cy'] += step
        st['dirty'] = True
    if os.path.isdir(folder):
        g.save(px_um)
        sc.st.close()
    return st['switch'] or None


@_per_sample
def hints(folder, sensitivity=None, model=None):
    """Жёлтая подсказка без окна: квадраты, где что-то есть (поиск аномалий: всё, что не похоже на пустую
    подложку). Нужна решётка (кресты в mc.view). sensitivity 0…100 (больше — больше квадратов).
    model — модель пустой подложки (папка скана после mc.train_hints или файл); по умолчанию учится на самом
    скане. Пишется в grid.json, видна в mc.view (Y)."""
    from .mask import Grid
    from .util import Result
    g = Grid(folder)
    if not g.ready():
        raise ValueError("No grid yet: mark the crosses in mc.view(folder) (G).")
    sc = _Scene(folder)
    info = {}
    g.find_useful(sc.render, sc.size, sensitivity=sensitivity, model=model or _AD.default_model(sc.st.meta.get('objective')), info=info)
    g.save(sc.st.meta.get('pixel_size_um') or 1.0)
    sc.st.close()
    return Result("Hints", Squares=len(g.useful), Sensitivity=g.sensitivity, Model=info.get('model'),
                  Time=f"{info.get('seconds')} s",
                  List=", ".join("{},{}".format(*g.name(*c)) for c in sorted(g.useful)) or "-")


@_per_sample
def train_hints(folder, default=False):
    """Модель пустой подложки для подсказок: снимите скан чистой подложки (без чешуек, с крестами) и
    отметьте на нём кресты (mc.view, G). Модель пишется в <скан>/hint_model.npz; default=True — ещё и как модель
    по умолчанию для этого объектива (~/.microcontrol/hint_model_<объектив>x.npz): все подсказки с этим
    объективом будут сравнивать с ней. Без модели подсказка учится на самом скане (тоже работает)."""
    from .mask import Grid
    from .util import Result
    g = Grid(folder)
    if not g.ready():
        raise ValueError("No grid yet: mark the crosses in mc.view(folder) (G).")
    sc = _Scene(folder)
    info = {}
    g.find_useful(sc.render, sc.size, info=info)
    path = os.path.join(folder, _AD.MODEL_FILE)
    g.hint_model.save(path)
    out = Result("Hint model", File=path, Components=len(g.hint_model.means), Time=f"{info.get('seconds')} s",
                 Found=f"{len(g.useful)} squares with something (should be ~0 on an empty substrate)")
    if default:
        obj = sc.st.meta.get('objective')
        d = os.path.join(os.path.expanduser('~'), '.microcontrol')
        os.makedirs(d, exist_ok=True)
        dp = os.path.join(d, f"hint_model_{obj}x.npz")
        g.hint_model.save(dp)
        out['Default'] = f"{dp} (objective {obj}x)"
    sc.st.close()
    return out
