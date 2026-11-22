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
import copy, csv, json, math, os
import time as _t
from collections import OrderedDict
import cv2
import numpy as np
from .storage import ScanData, THUMB, meta_A
from .util import wheel_delta, open_window, window_size, exit_key, EXIT, set_icon
from .undo import History, undo_key
from . import ui as UI
from . import log as L
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
                down_ = dw < ax1 - ax0                         # уменьшение — AREA (без ряби); тянут или
                part = cv2.resize(ov[ay0:ay1, ax0:ax1], (dw, dh),   # увеличение — LINEAR (в разы быстрее)
                                  interpolation=cv2.INTER_AREA if down_ and not fast else cv2.INTER_LINEAR)
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


HELP_V2 = "Arrows / drag: move  Wheel: zoom  0: whole  , / .: selection  H: all keys  Shift+Esc: exit"
HELP_MODE = {
    'look': "S: stitch  C: crosses  N: numbers  T: level  M: select cells (Drafts)  2-5: finer grid of a selection  Y: hints  R: ruler  E: export",
    'cross': "Click: cross exactly there (2 in a row = tilt)  Shift+click: snap to centre  Drag a mark: move  Right click: remove  Shift+C: again  F: fit grid  C / Esc: done",
    'select': "Click: square  Drag: box  Shift: unselect  Right click: remove a square (Shift: the whole area)  A: all  X: none  Esc: done",
    'paint': "Drag a rectangle: select all squares in it  Shift+drag: unselect  Click: one square  Right click: remove one (Shift: whole area)  A: all  X: none  Esc: done",
    'label': "Click a square, type its number X Y  Enter: set  Esc: done",
    'ruler': "Click two points: the distance (a third click starts again)  Drag: move  R or Esc: off",
    'refine': "Over a selection: 2-5 its finer grid (1: none), then M selects its cells: they become Drafts inside it",
}
HELP_HINTS = "Y: hints on / off  Shift+Y: select them  < / >: sensitivity  U: type it"
HELP_V = ("S: stitch   C: crosses (2 in a row = tilt; right click = remove, Shift+C = again)   F: fit grid   M: select "
          "(click, box; Shift = unselect; right click = remove a square, Shift + right click = the whole area)   N: number squares (X Y)   "
          "Y: hints, Shift+Y: select them, U, < / >: sensitivity   A / X: all / none   R: ruler   E: export   Esc: exit")
HELP_ALL = [
    ("Look", "Arrows or drag (right / middle in any mode): move   Wheel, + / -: zoom   0: whole"),
    ("", "T: level (tilt removed, from the crosses) / as shot   , / .: previous / next selection   H: this help   Esc: leave the mode"),
    ("Undo", "Ctrl+Z: undo any change (crosses, grid, squares, numbers, hints, removed selection)   Ctrl+Y: redo"),
    ("Exit", "Shift+Esc (or Ctrl+Esc), or close the window"),
    ("Stitch", "S: stitch this selection right here (progress in the window), then the view reopens"),
    ("Ruler", "R: click two points -> distance in um (a third click starts again); R or Esc: off"),
    ("Crosses", "C: mark crosses: click two neighbouring crosses in one row = tilt; more crosses far apart = more exact"),
    ("", "The mark goes exactly where you click   Shift+click: snap to the cross centre   Drag a mark: move it"),
    ("", "Right click: remove a cross   Backspace: the last one   Shift+C: start again   F: fit the grid to all crosses"),
    ("Numbers", "N: click a square, type the number written next to it (X Y), Enter. Two squares fix all numbers"),
    ("Select", "M: click a square (again: off) / drag a box (Shift: unselect)   A / X: all / none"),
    ("", "Right click on green: remove that square (or cell)   Shift + right click: the whole green area"),
    ("Hints", "Y: yellow hints on / off   Shift+Y: select them   < / >: sensitivity -/+ 5   U: type it"),
    ("Export", "E: export the selected squares (PNG + DXF)"),
]


_PREV = {}


def _preview(folder, w):
    """Картинка образца для карусели: обзорная мозаика (после сшивки) или первый кадр."""
    pv = os.path.join(folder, 'join', 'preview_all.jpg')
    from . import samples as _SM
    gp_ = os.path.join(folder, 'grid.json')
    key = (folder, _SM.sample_uid(folder), os.path.getmtime(pv) if os.path.exists(pv) else 0, w,   # uid: не путать
           os.path.getmtime(gp_) if os.path.exists(gp_) else 0)
                                                                                                   # с убранным
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
        small = cv2.resize(img, (max(1, int(img.shape[1] * k)), max(1, int(img.shape[0] * k))),
                           interpolation=cv2.INTER_AREA)
        if os.path.exists(pv):                              # выбранные квадраты — тёмно-зелёным, своей формы
            try:
                from .selections import marked_squares
                ms_ = marked_squares(folder)
                if ms_:
                    kk = (img.shape[1] - 2) / ms_['size'][0] * k
                    ov = small.copy()
                    for q in ms_['px']:
                        cv2.fillPoly(ov, [np.int32(np.round(q * kk * 16))], _SM.dim(_SM.sample_color(folder), 0.6),
                                     cv2.LINE_AA, 4)
                    small = cv2.addWeighted(ov, 0.7, small, 0.3, 0)
            except Exception:
                pass
        _PREV[key] = small
    return _PREV[key]


def _connected(cells, c):
    """Связная (по сторонам) область множества клеток, в которой лежит c."""
    area, todo = set(), [c]
    while todo:
        q = todo.pop()
        if q in cells and q not in area:
            area.add(q)
            todo += [(q[0] + 1, q[1]), (q[0] - 1, q[1]), (q[0], q[1] + 1), (q[0], q[1] - 1)]
    return area


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
            lines.append("no grid yet (C)")
        items.append(dict(title=f"Selection {SM.sample_index(f)}", color=SM.sample_color(f),
                          lines=lines, thumbs=[_preview(f, SM.CAR_W - 36)], confirm=car.get('confirm') == f, go=False))
    return items


def view(folder, fullscreen=False, sensitivity=None, _keys=None):
    """Окно; если что-то упало — полный текст ошибки пишется в журнал (logs/MC_*.log), окно закрывается."""
    try:
        return _view(folder, fullscreen=fullscreen, sensitivity=sensitivity, _keys=_keys)
    except Exception:
        import traceback
        L.event('VIEW-999', "The window stopped with an error:\n" + traceback.format_exc(), 40)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        raise


def view_window(folder):
    """Открыть съёмку в просмотре в НОВОМ окне (отдельный процесс): на каждый проект — своё окно, текущее
    остаётся открытым. Возвращает процесс."""
    import subprocess
    import sys
    from . import project as P_
    folder = os.path.abspath(folder)
    smp = P_.sample_obj(folder)
    base = smp.project.base if smp else os.path.dirname(folder)
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=pkg + os.pathsep + os.environ.get('PYTHONPATH', ''))
    kw = dict(creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)) if os.name == 'nt' else \
        dict(start_new_session=True)
    return subprocess.Popen([sys.executable, '-c', f"import microcontrol as mc; mc.view({folder!r})"], cwd=base,
                            env=env, **kw)


def _view(folder, fullscreen=False, sensitivity=None, _keys=None):
    """Открыть сшитый скан (или папку сеанса из нескольких образцов). Справа — карусель образцов:
    щелчок — открыть образец, x — убрать образец (папка уходит в <сеанс>/_removed/), , / . — предыдущий / следующий."""
    from . import samples as SM
    from .manual import screen_size
    ses, subs = SM.session_of(folder)
    cur = subs[0] if (ses and os.path.abspath(folder) == ses) else os.path.abspath(folder)
    if not subs:
        raise FileNotFoundError(f"No selections in {folder}")
    keys = iter(_keys) if _keys is not None else None
    Wv, Hv = open_window(WIN, fullscreen) or screen_size()
    car = dict(scroll=0, confirm=None, size=(Wv, Hv), hist=History())   # история — общая для всех выделений
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
    S — выделять квадраты: щелчок — квадрат, рамка — много сразу (Shift — снять); B — кисть: прямоугольники.
    A — все, X — ни одного.
    E — вывести выделенное (mc.export). Разметка хранится в <скан>/grid.json."""
    from .mask import Grid, snap_cross, export, sens_frac
    from . import samples as SM
    sc = _Scene(folder)
    g = Grid(folder)
    if sensitivity is not None:
        g.sensitivity = float(min(100, max(0, sensitivity)))
    px_um = sc.st.meta.get('pixel_size_um') or 1.0
    moved_ = g.sync(sc.P, sc.W, sc.H) if g.cr.marks else 0   # кадры встали иначе (сшивка): кресты — за ними
    if moved_:
        g.refined = None
    if g.cr.marks:
        g.save(px_um)
    from . import project as P0_
    from . import lattice as LT
    smp0 = P0_.sample_obj(folder)
    LAT = smp0.data.get('lattice') if smp0 is not None else None
    seeded_ = False
    if not g.ready() and LAT is not None and LAT.get('numbered'):   # решётка образца с номерами есть: кресты не нужны
        try:
            LT.seed_grid(folder, g, LAT)
            g.save(px_um)
            d_ = json.load(open(os.path.join(folder, 'grid.json'), encoding='utf-8'))
            d_['seeded'] = True
            d_['seeded_labels'] = [list(c_) for c_ in g.labels]
            try:                                                # кисть прежних съёмок — в новую сетку
                from .layers import brush_into_grid
                g2_ = Grid(folder)
                rf0_ = brush_into_grid(smp0.data.get('selections', []), LAT, folder, g2_)
                d_['selected'] = [list(c) for c in sorted(g2_.selected)]
                if rf0_:
                    d_['refine'] = rf0_
            except Exception:
                pass
            with open(os.path.join(folder, 'grid.json'), 'w', encoding='utf-8') as f_:
                json.dump(d_, f_, indent=1)
            g = Grid(folder)
            seeded_ = True
        except Exception:
            pass
    ses, subs = SM.session_of(folder)
    Wv, Hv = car['size']
    Hv -= UI.MENU_H                                         # сверху — строка меню
    st = dict(cx=sc.size[0] / 2, cy=sc.size[1] / 2, s=None, rot=0.0, mode='look', down=None, box=None,
              poly=[], dirty=True, Wv=Wv - SM.CAR_W, Hv=Hv, straight=False, switch=None, ruler=[], mouse=None,
              msg=car.pop('msg', None) or (f"Selection {SM.sample_index(folder)} ({subs.index(folder) + 1} of {len(subs)})" if len(subs) > 1 else "")
              + (f"   Crosses moved with the stitched frames ({moved_}); F: fit the grid again" if moved_ else "")
              + ("   The grid comes from the sample lattice (marked once): no crosses to mark here" if seeded_ else ""))

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
        elif act == 'del':                                  # сразу, без подтверждения (Ctrl+Z вернёт)
            car['confirm'] = None
            g.save(px_um) if f == folder else None
            if f == folder:
                sc.st.close()
            dst = SM.remove_sample(f)
            car['hist'].push(f"remove Selection {SM.sample_index(dst)}", None, None, kind='remove', src=f, dst=dst)
            rest = [s_ for s_ in subs if s_ != f]
            st['msg'] = f"Selection {SM.sample_index(dst)} removed to {dst} (Ctrl+Z: bring it back)"
            st['switch'] = (folder if f != folder else (rest[min(k, len(rest) - 1)] if rest else False))
        st['dirty'] = True

    def R():
        t = np.radians(st['rot'])
        return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    mid = lambda: np.array([st['Wv'] / 2, st['Hv'] / 2])
    to_view = lambda M: mid() + st['s'] * (R() @ (np.asarray(M, float) - (st['cx'], st['cy'])))
    from_view = lambda v: np.array([st['cx'], st['cy']]) + R().T @ ((np.asarray(v, float) - mid()) / st['s'])

    def to_view_arr(M):
        """to_view для массива точек (..., 2) разом."""
        M = np.asarray(M, float)
        return mid() + st['s'] * ((M - (st['cx'], st['cy'])) @ R().T)

    def cell_quads(cells_):
        """Углы квадратов сетки (N, 4, 2) в окне — одним вычислением, без цикла по квадратам."""
        v1_, v2_ = g.cr.vectors()
        C_ = np.asarray(cells_, float).reshape(-1, 2)
        offs = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)
        ij_ = C_[:, None, :] + offs[None]
        Mq = g.cr.O + ij_[..., :1] * v1_ + ij_[..., 1:] * v2_
        return to_view_arr(Mq)

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

    STAGES = {'light': "light field (uneven lighting)", 'seams': "seams (where each frame really is)",
              'brightness': "frame brightness", 'saving': "saving the preview and small copies",
              'export: squares': "cutting the squares", 'export: saving': "writing the selection picture",
              'export: drawing': "drawing: pictures of the squares in micrometres (DXF / DWG)"}

    menu = dict(hover=None, hits=[])
    from . import project as P_
    smp = P_.sample_obj(folder)                             # образец этой съёмки (если она в проекте)
    try:                                                    # заголовок окна — проект и образец (окон может быть несколько)
        cv2.setWindowTitle(WIN, f"{smp.title} [{smp.project.uid}] - MicroControl" if smp else
                           f"{os.path.basename(folder)} - MicroControl")
    except cv2.error:
        pass

    def show(img):
        ttl = (smp.title + f"  [{smp.project.uid}]" + (f"  —  {smp.description}" if smp.description else "")) if smp else os.path.basename(folder)
        bar, menu['hits'] = UI.menu_bar(img.shape[1], ttl, ["Open", "Scans", "Info", "Help"], menu['hover'])
        cv2.imshow(WIN, np.vstack([bar, img]))

    def menu_do(item):
        """Меню просмотра: открыть образец (последние съёмки), выбрать съёмку, сведения, подсказка."""
        from . import dialogs as D
        from .selections import list_selections
        base_ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(smp.folder)))) if smp else '.'
        try:
            if item == "Open":
                ss = P_.all_samples(base_)
                head, rows = P_.sample_table(ss)
                r_ = D.choose("Open a sample (its latest scans): every project in its own window", rows,
                              next((i for i, x in enumerate(ss) if x == smp), None), ("New window", "Open here"),
                              header=head)
                if r_:
                    ls = P_.latest_scans(ss[r_[1]].folder)
                    if not ls:
                        st['msg'] = f"{ss[r_[1]].title}: nothing scanned yet"
                    elif r_[0] == "New window":
                        view_window(ls[0])
                        st['msg'] = f"{ss[r_[1]].title}: opened in a new window"
                    else:
                        g.save(px_um)
                        st['switch'] = ls[0]
            elif item == "Scans":
                lst = list_selections(smp.folder if smp else os.path.dirname(folder))
                rows = repr(lst).split("\n")[2:-1:2]
                cur = next((i for i, it in enumerate(lst) if os.path.abspath(it['folder']) == os.path.abspath(folder)), None)
                r_ = D.choose("Scans (every scan of every selection)", rows, cur, ("Open",))
                if r_ and os.path.abspath(lst[r_[1]]['folder']) != os.path.abspath(folder):
                    g.save(px_um)
                    st['switch'] = lst[r_[1]]['folder']
            elif item == "Info":
                if smp is None:
                    st['msg'] = f"Scan {folder} (not in a project)"
                    return
                lines = [f"Project    {smp.project.name}   (id {smp.project.uid})", f"Sample     {smp.name}",
                         f"This scan  {os.path.relpath(folder, smp.folder)}",
                         f"Frames     {len(sc.st)}", f"Scans      {len(P_.all_scans(smp.folder))}"]
                r_ = D.describe(smp.title, lines, [("Project description", smp.project.description),
                                                  ("Sample description", smp.description)])
                if r_:
                    smp.project.description, smp.description = r_
                    st['msg'] = "Descriptions saved"
            elif item == "Help":
                st['help'] = not st.get('help')
        except Exception as e:
            st['msg'] = f"{item}: {e}"

    def busy(title, fn, close_scene=False, footer=""):
        """Долгая работа прямо в окне: fn() в фоновом потоке, в окне — этап, полоса 0-100 %, время.
        Возвращает (результат, ошибка или None)."""
        import threading
        from .util import Progress
        base = st.get('scene')
        if base is None:
            base = render()[:, :st['Wv']]
        base = (base.astype(np.float32) * 0.35).astype(np.uint8)
        pan = carousel_img()
        if close_scene:                                     # файлы скана перепишутся (Windows не даст при открытых)
            sc.st.close()
            sc.thumbs, sc.over = None, None
            sc.cache.clear()
        prog = dict(title=title, f=0.0, text="starting", eta=None)
        res = {}

        def hook(t, f, txt, el, eta):
            prog.update(title=t, f=f, text=txt, eta=eta)

        def work():
            try:
                res['r'] = fn()
            except Exception as e:                          # покажем в окне
                res['e'] = e

        t0 = _t.monotonic()
        Progress.hook = hook
        th = threading.Thread(target=work, daemon=True)
        th.start()
        try:
            while th.is_alive():
                img = base.copy()
                nm_ = prog['title'].replace("Stitch: ", "").lower()
                order_ = ['light', 'seams', 'brightness', 'saving']
                stage = (f"Step {order_.index(nm_) + 1} of 4:  " if nm_ in order_ else "") + STAGES.get(nm_, prog['title'])
                UI.progress_panel(img, title, stage, prog['f'], prog['text'], _t.monotonic() - t0, prog['eta'], footer)
                show(np.hstack([img, pan]) if pan.shape[0] == img.shape[0] else img)
                cv2.waitKey(60)
        finally:
            Progress.hook = None
            th.join()
        st['busy_s'] = _t.monotonic() - t0
        return res.get('r'), res.get('e')

    def stitch_here():
        """S: сшить это выделение (mc.stitch) прямо из просмотра; затем окно открывается заново уже со сшитой
        картинкой (кресты переезжают вместе с кадрами)."""
        from .stitch import stitch as _stitch
        g.save(px_um)
        r, e = busy(f"Stitching Selection {SM.sample_index(folder)}" if len(subs) > 1 else "Stitching",
                    lambda: _stitch(folder, verbose=True), close_scene=True, footer="(the view reopens by itself)")
        st['stitched'] = True
        if e is not None:
            car['msg'] = f"Stitching failed: {type(e).__name__}: {e}"
        else:
            car['msg'] = (f"Stitched in {st['busy_s']:.0f} s: seams {r.get('Seams accepted', '-')}, "
                          f"error {r.get('Seam error, px', '-')} px" + ("; crosses moved with the frames" if 'Crosses' in r else ""))
        st['switch'] = folder

    def ruler_text(a=None, b=None):
        """Подпись линейки: расстояние, мкм, и его проекции по осям картинки."""
        if a is None:
            if len(st['ruler']) < 2:
                return ""
            a, b = st['ruler']
        d = (np.asarray(b, float) - np.asarray(a, float)) * px_um
        return f"{np.hypot(*d):.2f} um   (dx {d[0]:+.2f}, dy {d[1]:+.2f})"

    def unseed():
        """Сетку поставили по решётке образца, а человек сам правит кресты или номера: номера, подставленные
        программой, убираются (иначе они спорят с его подписями: «labels disagree»), сетка — уже его."""
        gp_ = os.path.join(folder, 'grid.json')
        try:
            d_ = json.load(open(gp_, encoding='utf-8')) if os.path.exists(gp_) else {}
        except (OSError, ValueError):
            d_ = {}
        if not d_.get('seeded'):
            return
        for c_ in d_.get('seeded_labels') or [tuple(k_) for k_ in g.labels]:
            g.labels.pop(tuple(c_), None)
        g.save(px_um)
        d_ = json.load(open(gp_, encoding='utf-8'))
        d_.pop('seeded', None)
        d_.pop('seeded_labels', None)
        with open(gp_, 'w', encoding='utf-8') as f_:
            json.dump(d_, f_, indent=1)

    def add_cross(v, snap=False):
        unseed()
        """Крест — ровно там, где щёлкнули (середина пикселя экрана); snap (Shift+щелчок) — к центру креста
        на снимке (поиск по шаблону «+»)."""
        Mc = from_view((v[0] + 0.5, v[1] + 0.5))
        d = np.zeros(2)
        if snap:
            r = int(min(150, max(8, 15.0 / px_um)))
            reg = sc.render(Mc[0], Mc[1], 1.0, 4 * r + 1, 4 * r + 1)
            d, resp = snap_cross(cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY), r, arm=g.arm)
        try:
            ij = g.cr.add(Mc + d)
            g.refined = None                                # решётка — по отметкам (квадратная)
            if len(g.cr.marks) == 1:
                st['msg'] = ("Cross 1 marked. Now click its neighbour in the same row. Drag a mark to move it, "
                             "Shift+click: snap to the cross centre")
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
        tg_ = refine_target(np.mean(pts, axis=0))
        if tg_ is not None:                                  # внутри выделения с подсеткой — её клетки
            paint_refine(tg_, pts, on)
            return
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
        if ev == cv2.EVENT_MOUSEMOVE:
            st['hov'] = (x, y)
        if ev == cv2.EVENT_MOUSEMOVE and st['mode'] == 'ruler' and len(st['ruler']) == 1:
            st['mouse'] = (x + 0.5, y + 0.5)               # линейка тянется за мышью до второй точки
            st['dirty'] = True
        if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            st['down'] = (x, y, st['cx'], st['cy'], ev, shift, False)
            st['ctrl'] = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
            st['mdrag'] = None                                 # левой по отметке креста — тащить её
            if ev == cv2.EVENT_LBUTTONDOWN and st['mode'] == 'cross' and g.cr.marks:
                dists = [np.hypot(*(to_view(q) - (x + 0.5, y + 0.5))) for q in g.cr.marks]
                k = int(np.argmin(dists))
                if dists[k] <= 10:
                    st['mdrag'] = k
        elif ev == cv2.EVENT_MOUSEMOVE and st['down']:
            x0, y0, cx, cy, ev0, sh0, moved = st['down']
            if st.get('mdrag') is not None:                    # отметка едет за мышью точно
                if abs(x - x0) + abs(y - y0) > 2 or moved:
                    st['down'] = (x0, y0, cx, cy, ev0, sh0, True)
                    g.cr.marks[st['mdrag']] = from_view((x + 0.5, y + 0.5))
                    g.cr._fit()
                    st['dirty'] = True
            elif abs(x - x0) + abs(y - y0) > 4:
                st['down'] = (x0, y0, cx, cy, ev0, sh0, True)
                # сдвиг: правой / средней кнопкой — в любом режиме; левой — везде, кроме рамки (S) и кисти (B),
                # а там — с Ctrl
                pan = ev0 != cv2.EVENT_LBUTTONDOWN or st['mode'] in ('look', 'cross', 'label', 'ruler') or st.get('ctrl')
                if pan:
                    c = np.array([cx, cy]) - R().T @ (np.array([x - x0, y - y0], float) / st['s'])
                    st['cx'], st['cy'] = c
                elif st['mode'] in ('select', 'paint'):        # рамка: всё, чего она касается
                    st['box'] = (x0, y0, x, y, sh0)
                st['dirty'] = True
        elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP) and st['down']:
            x0, y0, cx, cy, ev0, sh0, moved = st['down']
            st['down'] = None
            mk, st['mdrag'] = st.get('mdrag'), None
            if mk is not None:
                if moved:
                    unseed()
                    g.refined = None
                    g.save(px_um)
                    st['msg'] = f"Cross mark moved; tilt {g.cr.angle():+.3f} deg" if len(g.cr.marks) > 1 else "Cross mark moved"
                else:
                    st['msg'] = "Drag the mark to move it; right click removes it"
                st['dirty'] = True
            elif st['box'] is not None:
                bx0, by0, bx1, by1, shb = st['box']
                st['box'] = None
                if g.ready():                               # рамка берёт всё, чего касается
                    apply_poly([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)], not shb, touch=True)
            elif st['mode'] in ('paint', 'select') and ev0 == cv2.EVENT_LBUTTONDOWN and not moved and \
                    refine_target(from_view((x, y))) is not None:  # щелчок по клетке подсетки
                q_ = from_view((x, y))
                paint_refine(refine_target(q_), [q_], not sh0, single=True)
            elif st['mode'] == 'paint' and ev0 == cv2.EVENT_LBUTTONDOWN and not moved:
                if g.ready():                               # щелчок кистью: один квадрат (Shift — снять)
                    c_ = g.cell_at(from_view((x, y)))
                    (g.selected.discard if sh0 else g.selected.add)(c_)
                    g.save(px_um)
                st['msg'] = f"Selected {len(g.selected)} squares"
            elif not moved and ev0 == cv2.EVENT_LBUTTONDOWN:
                if st['mode'] == 'cross':
                    add_cross((x, y), snap=sh0)
                elif st['mode'] == 'select' and g.ready():
                    c = g.cell_at(from_view((x, y)))
                    g.selected.symmetric_difference_update({c})
                    g.save(px_um)
                    st['msg'] = f"Square {c} {'selected' if c in g.selected else 'unselected'}, total {len(g.selected)}"
                elif st['mode'] == 'ruler':                 # линейка: точка ровно там, где щёлкнули
                    q_ = from_view((x + 0.5, y + 0.5))
                    st['ruler'] = [q_] if len(st['ruler']) != 1 else st['ruler'] + [q_]
                    st['msg'] = ruler_text() if len(st['ruler']) == 2 else "Ruler: click the second point"
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
                        unseed()
                        g.cr.remove(k)
                        g.save(px_um)
                        st['msg'] = "Cross mark removed"
                elif st['mode'] in ('look', 'select', 'paint') and g.ready() and \
                        unpaint_refine(from_view((x, y)), sh0):   # клетка подсетки (Shift — вся её область)
                    pass
                elif st['mode'] in ('look', 'select', 'paint') and g.ready() and g.selected:
                    c = g.cell_at(from_view((x, y)))        # правый щелчок по зелёному: этот квадрат,
                    if c in g.selected:                     # с Shift — вся связная область
                        area = _connected(g.selected, c) if sh0 else {c}
                        g.selected -= area
                        g.save(px_um)
                        st['msg'] = (f"Area of {len(area)} squares removed" if sh0 else f"Square {c} removed "
                                     "(Shift + right click: the whole area)") + f", {len(g.selected)} left"
                elif st['mode'] == 'label' and g.ready():
                    c = g.cell_at(from_view((x, y)))
                    unseed()
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
        img = draw_finer(img)                               # снятое крупнее — поверх, в своих контурах
        red, green, yellow = (40, 40, 255), SM.sample_color(folder), (0, 220, 255)   # выбранные — цветом выделения
        base_, void_ = img.copy(), cv2.inRange(img, (0, 0, 0), (22, 22, 22))   # вне мозаики сетку не рисуем
        if g.ready():
            lo = np.min([from_view(v) for v in ((0, 0), (Wv, 0), (0, Hv), (Wv, Hv))], 0)
            hi = np.max([from_view(v) for v in ((0, 0), (Wv, 0), (0, Hv), (Wv, Hv))], 0)
            v1, v2 = g.cr.vectors()
            step = min(np.hypot(*v1), np.hypot(*v2)) * st['s']
            lo, hi = np.maximum(lo, 0), np.minimum(hi, sc.size)            # решётка — только над мозаикой
            e_ = 0.25 * min(np.hypot(*v1), np.hypot(*v2))                   # квадрат почти весь над мозаикой
            cells = []
            if step >= 1.5:                                               # (до ~40 тыс. квадратов на экране)
                cr_ = np.array(list(g.cells_range(lo, hi)), float).reshape(-1, 2)
                cc_ = g.cr.O + (cr_[:, :1] + 0.5) * v1 + (cr_[:, 1:] + 0.5) * v2   # центры — разом
                ok_ = (cc_[:, 0] >= e_) & (cc_[:, 0] <= sc.size[0] - e_) & (cc_[:, 1] >= e_) & (cc_[:, 1] <= sc.size[1] - e_)
                cells = [(int(a), int(b)) for a, b in cr_[ok_]]
            quads = cell_quads(cells) if cells else np.zeros((0, 4, 2))
            sel_ = [i_ for i_, c in enumerate(cells) if c in g.selected]
            if g.selected:                                  # выбранное — чистое (там ищут), всё вокруг — под цветом
                clear_ = [quads[i_] for i_ in sel_] if sel_ else []
                if not clear_ and step < 1.5:               # мелко: квадраты целиком, без отбора
                    clear_ = list(cell_quads(sorted(g.selected)))
                UI.tint_outside(img, [(q_, []) for q_ in clear_] or [(np.zeros((3, 2)), [])], green, 0.3)
            hint_ = st.get('hints', _cfg.AUTO_HINTS)
            use_ = [i_ for i_, c in enumerate(cells) if hint_ and c not in g.selected and c in g.useful]
            if use_:                                        # жёлтые — подсказка
                ov = img.copy()
                cv2.fillPoly(ov, list(np.int32(quads[use_])), (0, 215, 255))
                img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
            if sel_ and step >= 7:                          # край выбранного — зелёной линией
                cv2.polylines(img, list(np.int32(np.round(quads[sel_]))), True, green, 1, cv2.LINE_AA)
            if step >= 34 and len(cells) <= 3000:           # номер квадрата в уголке: (X, Y) по подписям
                num = g.numbering()
                fs = min(0.5, step / 160)
                for c, pts in zip(cells, quads):
                    u, v = pts[np.argmin(pts.sum(1))] + (4, 4 + 14 * fs / 0.5)
                    txt = "{},{}".format(*g.name(*c, num))
                    col = (0, 230, 255) if c in g.labels else ((255, 255, 255) if g.labels else (170, 170, 170))
                    cv2.putText(img, txt, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, txt, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
                if st.get('lab'):                           # вводимый номер — у квадрата
                    pts = np.array([to_view(q) for q in g.corners(*st['lab'][0])])
                    cv2.polylines(img, [np.int32(pts)], True, (0, 230, 255), 2, cv2.LINE_AA)
            if cells and step >= 7:                         # крупно: контур каждого квадрата над мозаикой
                cv2.polylines(img, list(np.int32(np.round(quads))), True, red, 1, cv2.LINE_AA if step >= 30 else cv2.LINE_8)
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
        img = cv2.copyTo(base_, void_, img)
        pyramid_tint(img)                                     # внутри выделений — всё, кроме выбранного в них
        H_img, W_img = img.shape[:2]
        for q_, c_, lab_, dash_, new_ in sample_outlines():      # выделения образца на этой картинке
            pv_ = np.array([to_view(p_) for p_ in q_], float)
            if pv_[:, 0].max() < 0 or pv_[:, 1].max() < 0 or pv_[:, 0].min() > W_img or pv_[:, 1].min() > H_img:
                continue
            if dash_:
                UI.dashed_poly(img, pv_, c_, 2 if new_ else 1)
            else:
                cv2.polylines(img, [np.int32(np.round(pv_ * 16))], True, c_, 2, cv2.LINE_AA, 4)
            x_l, y_l = pv_[:, 0].min(), pv_[:, 1].min()
            if 0 <= x_l < W_img - 60 and 30 <= y_l < H_img:
                tw_ = UI.size(lab_, 0.42)[0]
                UI.panel(img, x_l, y_l - 20, x_l + tw_ + 10, y_l - 2, 0.8, c_)
                UI.put(img, lab_, x_l + 5, y_l - 6, 0.42, c_)
        draw_new_drafts(img)                                  # выбранное здесь — черновиком сразу
        draw_subgrids(img)                                    # подсетки выделений и их клетки кисти
        for k_, q in enumerate(g.cr.marks):
            u_, v_ = to_view(q)
            UI.plus(img, u_, v_, yellow if k_ == st.get("mdrag") else red)
        if st['box']:
            bx0, by0, bx1, by1, shb = st['box']
            cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 140, 255) if shb else yellow, 1)
        if st['mode'] == 'ruler' and st['ruler']:            # линейка
            a_ = to_view(st['ruler'][0])
            b_ = to_view(st['ruler'][1]) if len(st['ruler']) == 2 else st.get('mouse')
            q2 = st['ruler'][1] if len(st['ruler']) == 2 else (from_view(b_) if b_ is not None else None)
            UI.ruler(img, a_, b_, ruler_text(st['ruler'][0], q2) if q2 is not None else "")
        mode = {'look': 'Look', 'cross': 'Mark crosses', 'select': 'Select',
                'paint': 'Brush', 'label': 'Number squares', 'ruler': 'Ruler'}[st['mode']]
        mcol = {'look': (170, 170, 178), 'cross': (90, 90, 255), 'select': (90, 210, 90),
                'paint': (90, 210, 90), 'label': UI.ACCENT, 'ruler': (0, 220, 255)}[st['mode']]
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
        if not os.path.exists(os.path.join(folder, 'join', 'positions_all.csv')):
            it += ["not stitched: S"]
        if g.ready():
            it += [f"pitch {np.hypot(*g.cr.vectors()[0]) * px_um:.1f} um", f"tilt {g.cr.angle():+.2f} deg"]
        else:
            it += ["no grid yet: C"]
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
                                                        car['scroll'], title="Selections", go=False, hover=car.get('hxy'))
        return pan_

    # история (Ctrl+Z / Ctrl+Y): после каждого действия разметка (кресты, сетка, квадраты, номера, подсказки)
    # сравнивается с прежней; изменилась — пара (было, стало) в историю
    hist = car['hist']

    def g_sig():
        return (tuple(tuple(np.round(np.asarray(q, float), 3)) for q in g.cr.marks), tuple(g.cr.ij),
                hash(frozenset(g.selected)), tuple(sorted(g.labels.items())), g.refined is None,
                hash(frozenset(g.useful)), g.sensitivity)

    def g_snap():
        return {k_: (v_ if k_ in ('hint_model', 'folder') else copy.deepcopy(v_)) for k_, v_ in g.__dict__.items()}

    def g_label():
        t_ = (st['msg'] or "change").split(":")[0].split("(")[0].strip()
        return t_[:60]

    def h_apply(e_, back):
        """Откатить (back) или повторить действие e_. Разметка другого выделения — открыть его."""
        if e_.get('kind') == 'remove':
            import shutil as _sh
            src, dst = (e_['dst'], e_['src']) if back else (e_['src'], e_['dst'])
            if os.path.exists(src) and not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not back and os.path.abspath(src) == os.path.abspath(folder):
                    sc.st.close()
                _sh.move(src, dst)
            st['switch'] = e_['src'] if back else (folder if os.path.abspath(e_['src']) != os.path.abspath(folder) else False)
            return
        t_ = e_['before'] if back else e_['after']
        if os.path.abspath(e_['folder']) != os.path.abspath(folder):     # разметка другого выделения
            gg = Grid(e_['folder'])
            gg.__dict__.update(copy.deepcopy({k_: v_ for k_, v_ in t_.items() if k_ != 'folder'}))
            gg.save(e_.get('px', px_um))
            st['switch'] = e_['folder']
            return
        g.__dict__.update({k_: (v_ if k_ in ('hint_model', 'folder') else copy.deepcopy(v_)) for k_, v_ in t_.items()})
        g.save(px_um)
        if st['straight'] and g.ready():
            set_straight(True)
        elif not g.ready():
            set_straight(False)

    def regions_now():
        """Выделения образца (sample.json) и запросы подсеток этой съёмки (grid.json 'refine')."""
        if smp is None:
            return [], {}
        try:
            regs_ = json.load(open(smp.file, encoding='utf-8')).get('selections', [])
        except (OSError, ValueError):
            regs_ = []
        try:
            rf_ = json.load(open(os.path.join(folder, 'grid.json'), encoding='utf-8')).get('refine') or {}
        except (OSError, ValueError):
            rf_ = {}
        return regs_ + pending_figures(regs_, rf_), rf_

    def pending_figures(regs_, rf_):
        """Выбранные здесь квадраты, из которых окно микроскопа ещё не сделало черновик (или оно закрыто):
        подсетку можно задать и им — запрос 'cell:X,Y' потом перейдёт к черновику с этим квадратом."""
        L_ = lat_now()
        if L_ is None or not g.ready() or not g.selected:
            return []
        from .snap import mosaic_to_stage
        sq_ = sorted(g.selected)
        uv_ = LT.to_uv(L_, mosaic_to_stage(folder, np.array([g.cr.point(i + 0.5, j + 0.5) for i, j in sq_])))
        cells_ = {(int(np.floor(u)), int(np.floor(v))) for u, v in uv_}
        have_ = {tuple(c) for r_ in regs_ if not r_.get('factors') and r_.get('cells') for c in r_['cells']}
        out_ = []
        for fg_ in LT.figures(cells_ - have_):
            key_ = next((k_ for k_ in rf_ if k_.startswith('cell:') and
                         tuple(map(int, k_[5:].split(','))) in fg_), f"cell:{fg_[0][0]},{fg_[0][1]}")
            out_.append(dict(uid=key_, num=0, kind='draft', cells=[list(c) for c in fg_], factors=[], pending=True))
        return out_

    def lat_now():
        try:
            return json.load(open(smp.file, encoding='utf-8')).get('lattice') if smp is not None else None
        except (OSError, ValueError):
            return None

    def refine_target(q):
        """Выделение с подсеткой, в клетке которого точка q (px мозаики): самое глубокое. -> dict или None."""
        L_ = lat_now()
        if L_ is None:
            return None
        from .snap import mosaic_to_stage
        u_, v_ = LT.to_uv(L_, mosaic_to_stage(folder, [q]))[0]
        regs_, rf_ = regions_now()
        best_ = None
        for r_ in regs_:
            n_ = int((rf_.get(r_['uid']) or {}).get('n') or r_.get('sub') or 1)
            if n_ < 2 or not r_.get('cells'):
                continue
            N_ = LT.level(r_)
            if [int(np.floor(u_ * N_)), int(np.floor(v_ * N_))] in r_['cells'] and (best_ is None or N_ > best_['N']):
                best_ = dict(uid=r_['uid'], n=n_, N=N_, cells={tuple(c) for c in r_['cells']}, num=r_['num'],
                             kind='the new Draft' if r_.get('pending') else ('Draft' if r_.get('kind') == 'draft' else 'Selection'),
                             factors=(r_.get('factors') or []) + [n_])
        return best_

    def write_refine(uid, n, cells):
        gp_ = os.path.join(folder, 'grid.json')
        g.save(px_um)
        d_ = json.load(open(gp_, encoding='utf-8'))
        rf_ = d_.get('refine') or {}
        rf_[uid] = dict(n=int(n), cells=[list(c) for c in sorted(cells)])
        d_['refine'] = rf_
        with open(gp_, 'w', encoding='utf-8') as f_:
            json.dump(d_, f_, indent=1)

    def set_refine(n):
        """Клавиши 2–5 над выделением: его подсетка в n раз плотнее (1 — убрать)."""
        L_ = lat_now()
        if L_ is None or st.get('hov') is None:
            st['msg'] = "The finer grid needs the lattice of the sample: mark the crosses and number two squares (C, N)"
            return
        from .snap import mosaic_to_stage
        q_ = from_view(st['hov'])
        u_, v_ = LT.to_uv(L_, mosaic_to_stage(folder, [q_]))[0]
        regs_, rf_ = regions_now()
        hit_ = [r_ for r_ in regs_ if r_.get('cells') and
                [int(np.floor(u_ * LT.level(r_))), int(np.floor(v_ * LT.level(r_)))] in r_['cells']]
        if not hit_:
            st['msg'] = ("Finer grid: first select lithography squares (M, they turn green), then put the mouse over "
                         "them and press 2-5. Scanned rectangles have no cells: select squares inside them")
            return
        r_ = max(hit_, key=LT.level)
        old_ = rf_.get(r_['uid']) or {}
        write_refine(r_['uid'], n, old_.get('cells', []) if int(old_.get('n') or 0) == n else [])
        kind_ = 'Draft' if r_.get('kind') == 'draft' else 'Selection'
        if r_.get('pending'):
            kind_ = 'The new draft'
        st['msg'] = (f"{kind_} {r_['num'] or ''}: grid 1/{n} inside it (cells {LT.pitch_text(lat_now(), LT.level(r_) * n)}); "
                     "M selects its cells" if n > 1 else f"{kind_} {r_['num']}: no finer grid")

    def paint_refine(tg_, pts, on, single=False):
        """Кисть / щелчок по подсетке выделения: клетки уровня N·n внутри него."""
        L_ = lat_now()
        from .snap import mosaic_to_stage
        M_ = tg_['N'] * tg_['n']
        uv_ = LT.to_uv(L_, mosaic_to_stage(folder, np.asarray(pts, float)))
        if single:
            cs_ = {(int(np.floor(uv_[0][0] * M_)), int(np.floor(uv_[0][1] * M_)))}
        else:
            from shapely.geometry import Polygon as _Poly, Point as _Pt
            poly_ = _Poly(uv_ * M_)
            a0, b0 = np.floor((uv_ * M_).min(0)).astype(int)
            a1, b1 = np.ceil((uv_ * M_).max(0)).astype(int)
            cs_ = {(a, b) for a in range(a0, a1 + 1) for b in range(b0, b1 + 1)
                   if poly_.intersects(_Pt(a + 0.5, b + 0.5).buffer(0.45, 4))}
        n_ = tg_['n']
        cs_ = {c for c in cs_ if (c[0] // n_, c[1] // n_) in tg_['cells']}   # строго внутри выделения
        rf_ = regions_now()[1].get(tg_['uid']) or {}
        have_ = {tuple(c) for c in rf_.get('cells', [])}
        have_ = (have_ | cs_) if on else (have_ - cs_)
        write_refine(tg_['uid'], n_, have_)
        st['msg'] = (f"Grid 1/{M_} of {tg_['kind']} {tg_['num']}: {len(have_)} cells painted "
                     f"(they become Drafts in the microscope window)")

    def unpaint_refine(q, whole):
        """Правый щелчок по закрашенной клетке подсетки: убрать её (whole — всю связную область). -> bool."""
        L_ = lat_now()
        if L_ is None:
            return False
        from .snap import mosaic_to_stage
        regs_, rf_ = regions_now()
        for uid_, rq_ in rf_.items():
            r_ = next((r2 for r2 in regs_ if r2['uid'] == uid_), None)
            if r_ is None or not rq_.get('cells'):
                continue
            M_ = LT.level(r_) * int(rq_.get('n') or 1)
            u_, v_ = LT.to_uv(L_, mosaic_to_stage(folder, [q]))[0]
            c_ = (int(np.floor(u_ * M_)), int(np.floor(v_ * M_)))
            have_ = {tuple(c2) for c2 in rq_['cells']}
            if c_ in have_:
                gone_ = _connected(have_, c_) if whole else {c_}
                write_refine(uid_, rq_.get('n'), have_ - gone_)
                st['msg'] = (f"{len(gone_)} cells of grid 1/{M_} removed" if whole else
                             "Cell removed (Shift + right click: the whole area)")
                return True
        return False

    def finer_layers():
        """Съёмки образца крупнее этой (50x, 100x над 10x), каждая — [картинка, preview px -> мозаика этой (2x3),
        контур в мозаике этой]. Кэш по времени sample.json и превью."""
        if smp is None:
            return []
        from . import project as P2_
        from .snap import mosaic_to_stage, stage_to_mosaic, geometry
        from .layers import from_json
        my_obj = sc.st.meta.get('objective') or 0
        scans_ = [f_ for f_ in P2_.latest_scans(smp.folder) if os.path.abspath(f_) != os.path.abspath(folder)]
        key_ = (os.path.getmtime(smp.file) if os.path.exists(smp.file) else 0,
                tuple((f_, os.path.getmtime(os.path.join(f_, 'join', 'preview_all.jpg')))
                      for f_ in scans_ if os.path.exists(os.path.join(f_, 'join', 'preview_all.jpg'))))
        if st.get('_fl_key') == key_:
            return st['_fl']
        regs_ = {r_['uid']: r_ for r_ in json.load(open(smp.file, encoding='utf-8')).get('selections', [])}
        out_ = []
        for f_ in scans_:
            jp_ = os.path.join(f_, 'join', 'preview_all.jpg')
            try:
                meta_ = json.load(open(os.path.join(f_, 'meta.json'), encoding='utf-8'))
                if not os.path.exists(jp_) or (meta_.get('objective') or 0) <= my_obj:
                    continue
                pic_ = cv2.imread(jp_)
                g2_ = geometry(f_)
                size_ = np.asarray(g2_['pos'], float).max(0) + (g2_['W'], g2_['H'])
                k_ = min(1.0, 4000 / size_.max())
                src_ = np.float32([[0, 0], [pic_.shape[1], 0], [0, pic_.shape[0]]])
                dst_ = stage_to_mosaic(folder, mosaic_to_stage(f_, src_ / k_)).astype(np.float32)
                A_ = cv2.getAffineTransform(src_, dst_)
                r_ = regs_.get(meta_.get('sample_uid'))
                ring_ = None
                if r_ is not None and r_.get('shape'):
                    gm_ = from_json(r_['shape'])
                    ps_ = [gm_] if gm_.geom_type == 'Polygon' else list(gm_.geoms)
                    ring_ = [stage_to_mosaic(folder, np.array(p_.exterior.coords)[:-1]) for p_ in ps_]
                out_.append((meta_.get('objective') or 0, pic_, A_, ring_))
            except Exception:
                continue
        out_.sort(key=lambda t_: t_[0])                     # крупнее — выше
        st['_fl_key'], st['_fl'] = key_, out_
        return out_

    def draw_finer(img):
        Hh, Ww = img.shape[:2]
        Rm = R() * st['s']                                  # мозаика этой -> окно: mid + s R (M - c)
        T_ = np.c_[Rm, mid() - Rm @ np.array([st['cx'], st['cy']])]
        for o_, pic_, A_, ring_ in finer_layers():
            M_ = T_[:, :2] @ A_[:, :2]
            t_ = T_[:, :2] @ A_[:, 2] + T_[:, 2]
            corners_ = np.array([[0, 0], [pic_.shape[1], 0], [0, pic_.shape[0]], [pic_.shape[1], pic_.shape[0]]], float)
            cv_ = corners_ @ M_.T + t_
            if cv_[:, 0].max() < 0 or cv_[:, 1].max() < 0 or cv_[:, 0].min() > Ww or cv_[:, 1].min() > Hh:
                continue
            if np.hypot(*M_[:, 0]) * pic_.shape[1] < 12:        # совсем мелко — не видно, не тратим время
                continue
            w_ = cv2.warpAffine(pic_, np.float32(np.c_[M_, t_]), (Ww, Hh), flags=cv2.INTER_LINEAR, borderValue=0)
            mask_ = cv2.inRange(w_, (7, 7, 7), (255, 255, 255))  # вне кадров (чёрное) — не кладём
            if ring_:
                cm_ = np.zeros_like(mask_)
                cv2.fillPoly(cm_, [np.int32(np.round(to_view_arr(r2_) * 16)) for r2_ in ring_], 255, cv2.LINE_8, 4)
                mask_ = cv2.bitwise_and(mask_, cm_)
            img = cv2.copyTo(w_, mask_, img)
        return img

    def draw_subgrids(img):
        """Подсетки выделений (тонкие линии внутри их клеток) и клетки кисти в них (зелёные)."""
        L_ = lat_now()
        if L_ is None:
            return
        from .snap import stage_to_mosaic
        regs_, rf_ = regions_now()
        Hh, Ww = img.shape[:2]
        for r_ in regs_:
            n_ = int((rf_.get(r_['uid']) or {}).get('n') or r_.get('sub') or 1)
            if n_ < 2 or not r_.get('cells'):
                continue
            N_ = LT.level(r_)
            col_ = SM.color(r_['num'] - 1)
            for a, b in r_['cells']:
                for t in range(1, n_):
                    for p0, p1 in ((((a + t / n_) / N_, b / N_), ((a + t / n_) / N_, (b + 1) / N_)),
                                   ((a / N_, (b + t / n_) / N_), ((a + 1) / N_, (b + t / n_) / N_))):
                        q_ = stage_to_mosaic(folder, LT.to_stage(L_, [p0, p1]))
                        v0, v1 = to_view(q_[0]), to_view(q_[1])
                        if max(v0[0], v1[0]) < 0 or min(v0[0], v1[0]) > Ww or max(v0[1], v1[1]) < 0 or min(v0[1], v1[1]) > Hh:
                            continue
                        cv2.line(img, (int(v0[0] * 16), int(v0[1] * 16)), (int(v1[0] * 16), int(v1[1] * 16)),
                                 SM.dim(col_, 0.8), 1, cv2.LINE_AA, 4)
            M_ = N_ * n_
            pc_ = (rf_.get(r_['uid']) or {}).get('cells', [])
            if pc_:                                         # выбранные клетки подсетки — зелёной рамкой
                clear_ = [to_view_arr(stage_to_mosaic(folder, LT.cell_stage(L_, a, b, M_))) for a, b in pc_]
                cv2.polylines(img, [np.int32(np.round(q_ * 16)) for q_ in clear_], True, (90, 200, 90), 1, cv2.LINE_AA, 4)

    def draw_new_drafts(img):
        """Выбранные квадраты, из которых черновик ещё не сделан (окно микроскопа сделает): пунктир «Draft (new)» —
        видно сразу, каким он будет."""
        regs_, rf_ = regions_now()
        L_ = lat_now()
        if L_ is None:
            return
        from .snap import stage_to_mosaic
        tv_ = lambda A_: to_view_arr(stage_to_mosaic(folder, A_))
        for r_ in [r2 for r2 in regs_ if r2.get('pending')]:
            for ext_, hs_ in UI.geom_shapes(LT.cells_geom(L_, [tuple(c) for c in r_['cells']], 1), tv_):
                UI.dashed_poly(img, ext_, (235, 235, 235), 2)
                x_l, y_l = ext_[:, 0].min(), ext_[:, 1].min()
                if 0 <= x_l < img.shape[1] - 90 and 30 <= y_l < img.shape[0]:
                    UI.panel(img, x_l, y_l - 20, x_l + UI.size("Draft (new)", 0.42)[0] + 10, y_l - 2, 0.8, (235, 235, 235))
                    UI.put(img, "Draft (new)", x_l + 5, y_l - 6, 0.42, (235, 235, 235))

    def pyramid_tint(img):
        """Пирамидка: внутри каждого выделения, в котором выбраны участки меньше (дети, клетки подсетки), всё,
        кроме них, — под цветом. Чем глубже — тем чище: видно, куда смотреть дальше."""
        if smp is None:
            return
        from .snap import stage_to_mosaic
        from .layers import from_json
        regs_, rf_ = regions_now()
        L_ = lat_now()
        kids_ = {}
        for r_ in regs_:
            if r_.get('parent') and r_.get('kind') != 'group':
                kids_.setdefault(r_['parent'], []).append(r_)
        for r_ in regs_:
            if r_.get('kind') == 'group' or not r_.get('shape'):
                continue
            if os.path.abspath(folder).startswith(os.path.abspath(P0_.sel_folder(smp.folder, r_)) + os.sep):
                continue                                    # своё выделение этой съёмки — по квадратам (выше)
            holes_ = [from_json(k_['shape']) for k_ in kids_.get(r_['uid'], []) if k_.get('shape')]
            pc_ = (rf_.get(r_['uid']) or {}).get('cells', [])
            if pc_ and L_ is not None:
                M_ = LT.level(r_) * int((rf_.get(r_['uid']) or {}).get('n') or 1)
                holes_.append(LT.cells_geom(L_, [tuple(c) for c in pc_], M_))
            if not holes_:
                continue
            tv_ = lambda A_: to_view_arr(stage_to_mosaic(folder, A_))
            mask_ = np.zeros(img.shape[:2], np.uint8)
            for ext_, hs_ in UI.geom_shapes(from_json(r_['shape']), tv_):
                cv2.fillPoly(mask_, [np.int32(np.round(ext_ * 16))], 255, cv2.LINE_8, 4)
            for h_ in holes_:
                for ext_, hs_ in UI.geom_shapes(h_, tv_):
                    cv2.fillPoly(mask_, [np.int32(np.round(ext_ * 16))], 0, cv2.LINE_8, 4)
            tint_ = cv2.addWeighted(img, 0.72, np.full_like(img, SM.color(r_['num'] - 1)), 0.28, 0)
            cv2.copyTo(tint_, mask_, img)

    def sample_outlines():
        """Выделения образца на этой мозаике: [(кольцо в пикселях мозаики, цвет, подпись, пунктир, ждёт)]."""
        if smp is None:
            return []
        sp_ = smp.file
        key_ = (os.path.getmtime(sp_) if os.path.exists(sp_) else 0,
                os.path.getmtime(os.path.join(folder, 'grid.json')) if os.path.exists(os.path.join(folder, 'grid.json')) else 0)
        if st.get('_so_key') == key_:
            return st['_so']
        out_ = []
        try:
            from .snap import stage_to_mosaic
            from .layers import from_json
            from shapely.geometry import box as _box
            for r_ in json.load(open(sp_, encoding='utf-8')).get('selections', []):
                if r_.get('kind') == 'group':
                    continue
                if r_.get('shape'):
                    gm_ = from_json(r_['shape'])
                else:
                    (ax, ay), (bx, by) = r_['A'], r_['B']
                    fw_, fh_ = r_.get('fw', 0), r_.get('fh', 0)
                    gm_ = _box(min(ax, bx) - fw_ / 2, min(ay, by) - fh_ / 2, max(ax, bx) + fw_ / 2, max(ay, by) + fh_ / 2)
                ps_ = [gm_] if gm_.geom_type == 'Polygon' else list(getattr(gm_, 'geoms', []))
                kind_ = {'draft': 'Draft'}.get(r_.get('kind'), 'Sel.')
                for p_ in ps_:
                    q_ = stage_to_mosaic(folder, np.array(p_.exterior.coords)[:-1])
                    out_.append((q_, SM.color(r_['num'] - 1), f"{kind_} {r_['num']} {r_.get('obj', '')}x"
                                 if r_.get('kind') != 'draft' else f"Draft {r_['num']}", r_.get('kind') == 'draft', False))
        except Exception:
            out_ = []
        st['_so_key'], st['_so'] = key_, out_
        return out_

    h_last = [g_sig(), g_snap()]
    def mouse_cb(ev, x, y, flags, _=None):                 # окно: сверху меню, ниже — картинка и список
        if y < UI.MENU_H and st['down'] is None:
            menu['hover'] = x if ev == cv2.EVENT_MOUSEMOVE else menu['hover']
            if ev == cv2.EVENT_LBUTTONDOWN:
                st['menu_click'] = x
            if ev == cv2.EVENT_MOUSEMOVE:
                st['car_dirty'] = True
            return
        if menu['hover'] is not None:
            menu['hover'], st['car_dirty'] = None, True
        mouse(ev, x, y - UI.MENU_H, flags)

    cv2.setMouseCallback(WIN, mouse_cb)
    while True:
        if st['down'] is None and st.get('lab') is None:
            sg_ = g_sig()
            if sg_ != h_last[0]:                            # что-то сделано: в историю
                new_ = g_snap()
                hist.push(g_label(), h_last[1], new_, folder=folder, px=px_um)
                h_last[:] = [sg_, new_]
        if _t.monotonic() - st.get('_sync_t', 0) > 0.5:  # окно микроскопа поменяло образец или кисть —
            st['_sync_t'] = _t.monotonic()                 # показать то же, что и там
            mt_ = tuple(os.path.getmtime(p_) if os.path.exists(p_) else 0 for p_ in
                        ((smp.file if smp is not None else ''), os.path.join(folder, 'grid.json')))
            if st.get('_mt') is None:
                st['_mt'] = mt_
            elif mt_ != st['_mt'] and st['down'] is None:
                st['_mt'] = mt_
                subs2 = SM.session_of(folder)[1]
                if subs2 != subs:                            # выделение удалили / вернули / сняли новое
                    subs[:] = subs2
                    st['car_dirty'] = st['dirty'] = True
                    if smp is not None and not any(os.path.abspath(folder).startswith(os.path.abspath(
                            P0_.sel_folder(smp.folder, r_)) + os.sep) for r_ in json.load(
                            open(smp.file, encoding='utf-8')).get('selections', [])):
                        st['msg'] = ("This selection was deleted in the microscope window "
                                     "(Ctrl+Z there brings it back)")
                try:
                    g2_ = Grid(folder)
                    if g2_.selected != g.selected:           # кисть поменяли там (удалили черновик и т. п.)
                        g.selected = set(g2_.selected)
                        h_last[:] = [g_sig(), g_snap()]      # это не своё действие: не в историю
                        st['dirty'] = True
                except Exception:
                    pass
                st['dirty'] = True
        if st['switch'] is not None:                      # другой образец из карусели
            break
        if not fullscreen:
            W_, H_ = window_size(WIN, (st['Wv'] + SM.CAR_W, st['Hv'] + UI.MENU_H))
            H_ -= UI.MENU_H
            if W_ >= 100 + SM.CAR_W and H_ >= 100 and (W_ - SM.CAR_W, H_) != (st['Wv'], st['Hv']):
                st['Wv'], st['Hv'], st['dirty'] = W_ - SM.CAR_W, H_, True
                car['size'] = (W_, H_ + UI.MENU_H)
        if st['s'] is None:
            fit()
        if st.get('menu_click') is not None:                # строка меню: Open, Scans, Info, Help
            mx_, st['menu_click'] = st['menu_click'], None
            it_ = next((t_ for x0_, x1_, t_ in menu['hits'] if x0_ <= mx_ <= x1_), None)
            if it_:
                menu_do(it_)
                st['dirty'] = True
                if st['switch'] is not None:
                    break
        if st['dirty']:
            show(render())
            st['dirty'] = st['car_dirty'] = False
            if not st.get('icon'):                          # окно уже на экране: значок программы (Windows)
                st['icon'] = True
                set_icon(WIN)
        elif st.get('car_dirty') and st.get('scene') is not None:   # только карусель (наведение мыши)
            show(np.hstack([st['scene'], carousel_img()]))
            st['car_dirty'] = False
        key = next(keys, EXIT) if keys is not None else cv2.waitKeyEx(8 if st['down'] else 20)
        if keys is not None:
            cv2.waitKey(1)
            if isinstance(key, tuple):                      # тест: ('click'|'rclick', x, y[, shift]) или ('drag', ...)
                kind, *a = key
                fl = cv2.EVENT_FLAG_SHIFTKEY if (len(a) > 2 and a[-1] == 'shift') else 0
                if kind == 'call':                          # тест: сделать что-то «из другого окна»
                    a[0]()
                    key = -1
                    kind = ''
                elif kind == 'sleep':
                    _t.sleep(a[0])
                    key = -1
                    kind = ''
                if kind[:1] == 'q':                         # ('qclick', ...) — точки в пикселях мозаики
                    kind = kind[1:]
                    nums_ = [v_ for v_ in a if not isinstance(v_, str)]
                    a = [int(round(v_)) for xy_ in zip(nums_[0::2], nums_[1::2]) for v_ in to_view(xy_)] + \
                        (['shift'] if a[-1] == 'shift' else [])
                if kind == 'move':
                    mouse(cv2.EVENT_MOUSEMOVE, a[0], a[1], 0)
                elif kind in ('click', 'rclick'):
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
        ud_ = undo_key(key)
        if ud_:                                             # Ctrl+Z — отменить, Ctrl+Y / Ctrl+Shift+Z — вернуть
            e_ = hist.undo() if ud_ == 'undo' else hist.redo()
            if e_ is None:
                st['msg'] = "Nothing to undo" if ud_ == 'undo' else "Nothing to redo"
            else:
                h_apply(e_, ud_ == 'undo')
                h_last[:] = [g_sig(), g_snap()]
                st['msg'] = (f"Undone: {e_['label']}  (Ctrl+Y: redo, {len(hist)} more)" if ud_ == 'undo' else
                             f"Redone: {e_['label']}")
            st['dirty'] = True
            if st['switch'] is not None:
                break
            continue
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
                unseed()                                    # свои номера — вместо подставленных
                if not buf:
                    g.labels[c] = g.name(*c)                # Enter без цифр — подтвердить показанный номер
                elif len(parts) == 2 and all(p_.lstrip('-').isdigit() for p_ in parts):
                    g.labels[c] = (int(parts[0]), int(parts[1]))
                else:
                    st['msg'] = "Two numbers: X Y"; st['dirty'] = True; continue
                g.save(px_um)
                st['lab'] = None
                X_, Y_ = g.labels[c]
                warn_ = ("   [!] Lithography numbers go from 1,1 to 58,58: check this one"
                         if not (1 <= X_ <= 58 and 1 <= Y_ <= 58) else "")
                st['msg'] = f"Square {c} = {X_},{Y_}.   {g.numbering()[2]}{warn_}"
            st['dirty'] = True
            continue
        ch = chr(key & 0xFF) if 0 <= key < 256 else ''
        raw = ch
        step = 0.25 * min(st['Wv'], st['Hv']) / st['s']    # стрелки — четверть окна
        ch = ch.lower()
        if key == 27:
            if st.get('help'):
                st['help'] = False
            elif st['mode'] != 'look':
                st['mode'], st['ruler'] = 'look', []
            else:
                st['msg'] = "Exit: Shift+Esc or close the window"
            st['dirty'] = True
        elif raw in (',', '.') and len(subs) > 1:           # предыдущий / следующий образец
            st['switch'] = subs[(subs.index(folder) + (1 if raw == '.' else -1)) % len(subs)]
        elif raw == 'C':                                     # Shift+C: кресты заново (Ctrl+Z вернёт)
            unseed(); g.cr.clear(); g.refined = None; set_straight(False); g.save(px_um)
            st['mode'] = 'cross'
            st['msg'] = "Crosses cleared (Ctrl+Z brings them back). Click a cross, then its neighbour in the same row"
        elif ch in 'sj' and ch:                              # S: сшить прямо здесь; окно потом откроется заново
            stitch_here()
            break
        elif ch == 'r':                                      # R: линейка, две точки -> расстояние
            if st['mode'] == 'ruler':
                st['mode'], st['ruler'] = 'look', []
                st['msg'] = "Ruler off"
            else:
                st['mode'], st['ruler'] = 'ruler', []
                st['msg'] = "Ruler: click two points (a third click starts again). R or Esc: off"
        elif ch == 'c':                                      # C: отмечать кресты (ещё раз C — выйти)
            if st['mode'] == 'cross':
                st['mode'] = 'look'
                st['msg'] = ""
            else:
                st['mode'] = 'cross'
                st['msg'] = ("Click a cross, then its neighbour in the same row: the picture turns level. "
                             "Shift+C: start again")
        elif ch in 'mb' and ch:                              # M (и B по привычке): выбирать клетки
            st['mode'] = 'select' if g.ready() else st['mode']
            st['msg'] = ("Select: click = square (again = off), drag = box (Shift = unselect); right click = remove "
                         "one, Shift + right click = the whole area; they become Drafts") if g.ready() \
                else "Mark at least 2 crosses first (C)"
        elif key == 25 or raw == 'Y':                       # Ctrl+Y / Shift+Y: жёлтое -> зелёное
            if g.ready():
                if not g.useful and not getattr(g, 'scores', None):
                    hints()
                n0 = len(g.selected)
                g.selected |= set(g.useful)
                g.save(px_um)
                st['mode'] = 'select'                       # сразу выбор: поправить выделение руками
                st['msg'] = f"Hints selected: +{len(g.selected) - n0} squares, total {len(g.selected)}. " \
                            "Select: drag boxes to add, Shift+drag to unselect"
        elif ch == 'y' and g.ready():                       # Y: подсказка вкл. / выкл. (первый раз — посчитать)
            if not st.get('hints', _cfg.AUTO_HINTS) and not getattr(g, 'scores', None):
                st['msg'] = "Looking for squares with something in them..."
                show(render()); cv2.waitKey(1)
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
                else "Mark at least 2 crosses first (C)"
        elif key == 8:
            if st['mode'] == 'cross':
                unseed(); g.cr.undo(); g.save(px_um); st['msg'] = "Last cross mark removed"
        elif ch == 'f' and g.ready():
            n_, rms = g.refine(lambda x, y, w, h: sc.render(x, y, 1.0, w, h), sc.size, px_um)
            if _cfg.AUTO_HINTS or (getattr(g, 'scores', None) and st.get('hints')):
                hints()
            g.save(px_um)
            st['msg'] = f"Grid fitted to all {n_} crosses" + (f": {rms:.2f} px off" if rms is not None else ": too few") + \
                f"   hints: {len(g.useful)} squares (Y)"
        elif ch == 'h':
            st['help'] = not st.get('help')
        elif ch == 't':                                      # T: ровно (наклон снят) / как снято
            if not g.ready():
                st['msg'] = "The tilt is taken from the crosses: mark two neighbouring crosses in a row (C), then T"
            else:
                set_straight(not st['straight'])
                st['msg'] = (f"Level: the picture is turned by {st['rot']:+.2f} deg, the squares stand straight (T: as shot)"
                             if st['straight'] else "Picture as shot, with its tilt (T: level)")
        elif ch in '12345' and ch and st.get('lab') is None:   # подсетка выделения под мышью: в n раз плотнее
            set_refine(int(ch))
        elif ch == 'a' and st['mode'] in ('select', 'paint') and g.ready():
            W_, H_ = sc.size
            g.selected |= g.cells_in([(0, 0), (W_, 0), (W_, H_), (0, H_)], sc.size)
            g.save(px_um); st['msg'] = f"All {len(g.selected)} squares selected"
        elif ch == 'x' and st['mode'] in ('select', 'paint'):
            g.selected.clear(); g.save(px_um); st['msg'] = "Selection cleared"
        elif ch == 'e':                                      # вывод: картинки квадратов + чертёж, прогресс в окне
            g.save(px_um)
            r, e = busy("Export", lambda: export(folder))
            if e is not None:
                st['msg'] = f"Export: {e}"
            else:
                dw = r.get('DWG', '')
                st['msg'] = (f"Exported {r['Squares']} squares in {st['busy_s']:.0f} s: "
                             + (f"drawing {dw}" if str(dw).endswith('.dwg') else f"drawing {r.get('Drawing', '-')}"
                                + ("  (DWG: install ODA File Converter)" if 'ODA' in str(dw) else "")))
        elif ch and ch in '+=' or key in {2162688, 65365}:
            zoom(1.25)
        elif ch and ch in '-_' or key in {2228224, 65366}:
            zoom(0.8)
        elif ch == '0':
            fit()
        elif key in {2424832, 65361}:                       # только стрелки двигают картинку
            st['cx'] -= step
        elif key in {2555904, 65363}:
            st['cx'] += step
        elif key in {2490368, 65362}:
            st['cy'] -= step
        elif key in {2621440, 65364}:
            st['cy'] += step
        st['dirty'] = True
    if os.path.isdir(folder) and not st.get('stitched'):   # после сшивки разметку уже записала она (кресты — за кадрами)
        g.save(px_um)
    if os.path.isdir(folder):
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
        raise ValueError("No grid yet: mark the crosses in mc.view(folder) (C).")
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
        raise ValueError("No grid yet: mark the crosses in mc.view(folder) (C).")
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
