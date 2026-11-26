"""Просмотр скана в окне OpenCV (лёгкий, для слабого ПК).

  мышь (тянуть любой кнопкой) / стрелки / WASD   перемещение (Shift — быстрее)
  двойной щелчок            этот участок — в центр
  колесо / + - / PgUp PgDn  масштаб
  0                         показать весь скан
  Esc                       выход

Мелкий масштаб собирается из превью (1/16), которые пишутся при скане;
крупный — из нужных кусков кадров. Поэтому окно открывается сразу и на тысячах кадров.
"""
import copy, csv, json, math, os
import time as _t
from collections import OrderedDict
from .text import cv2                                     # cv2 со шрифтом JetBrains Mono
import numpy as np
from .storage import ScanData, THUMB, meta_A
from .util import wheel_delta, wheel_gesture, open_window, window_size, exit_key, EXIT, set_icon, shift_held
from .undo import History, undo_key
from . import ui as UI
from . import log as L
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
        self.org = P.min(0)                                 # где начинается мозаика (кадры добавляются — меняется)
        self.P = P - self.org
        self.size = self.P.max(0) + (self.W, self.H)
        self.thumbs = self.st.thumbs()
        self.cache, self.cache_b = OrderedDict(), 0         # (i, уровень) -> уменьшенный кадр; байт в кэше
        import threading
        self.lock = threading.RLock()                       # кэш — общий с фоновой подгрузкой
        self.want, self.fresh = [], False                   # что подгрузить фоном; подгрузилось — перерисовать
        self.wake = threading.Event()
        self.alive = True
        threading.Thread(target=self._loader, daemon=True).start()
        self.over, self.ok_ = None, 0.0                     # обзорная мозаика (join/preview_all.jpg)
        pv = os.path.join(folder, 'join', 'preview_all.jpg')
        if os.path.exists(pv) and (not os.path.exists(jp) or _mt(pv) >= _mt(jp)):
            try:                                            # очередь как раз пишет его — тогда без обзора
                img = cv2.imdecode(np.fromfile(pv, np.uint8), cv2.IMREAD_COLOR)
            except (OSError, ValueError):
                img = None
            if img is not None:
                self.over, self.ok_ = img, (img.shape[1] - 2) / self.size[0]

    def level_img(self, i, lv):
        key = (i, lv)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        f = 2 ** lv
        p = os.path.join(self.st.folder, 'viewer', 'levels', f"{i:05d}_{lv}.jpg")
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) if lv and os.path.exists(p) else None
        if img is None:                                     # нет готового уровня — из кадра
            with self.lock:
                fr_ = self.st.frame(i)
            img = fr_ if lv == 0 else cv2.resize(fr_, (max(1, self.W // f), max(1, self.H // f)),
                                                 interpolation=cv2.INTER_AREA)
        with self.lock:
            self.cache[key] = img
            self.cache_b += img.nbytes
            while self.cache_b > 400e6 and len(self.cache) > 1:  # не больше ~400 МБ на старом ПК
                self.cache_b -= self.cache.popitem(last=False)[1].nbytes
        return img

    def cached(self, i, lv):
        """Готовая копия кадра i уровня lv или ближайшая грубее из кэша (None — только миниатюра)."""
        with self.lock:
            for l_ in range(lv, 4):
                a = self.cache.get((i, l_))
                if a is not None:
                    if l_ == lv:
                        self.cache.move_to_end((i, l_))
                    return a, l_
        return None, None

    def _loader(self):
        """Фоном: раскрыть кадры, которых просил вид (окно не ждёт диска и JPEG — тянется плавно)."""
        while self.alive:
            self.wake.wait(0.5)
            self.wake.clear()
            while self.alive:
                with self.lock:
                    if not self.want:
                        break
                    key = self.want.pop()                   # последние просьбы — первыми (что видно сейчас)
                try:
                    self.level_img(*key)
                    self.fresh = True
                except Exception:
                    pass

    def close(self):
        self.alive = False
        self.wake.set()
        try:
            self.st.close()
        except Exception:
            pass

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
                    pc_ = part[v0:v1, u0:u1]
                    if pc_.ndim == 2:
                        pc_ = cv2.cvtColor(pc_, cv2.COLOR_GRAY2BGR)
                    roi_ = out[oy + v0:oy + v1, ox + u0:ox + u1]
                    # в обзоре «не снято» — чёрное: остаётся фоном (как при крупном масштабе), иначе оно
                    # закрывало слои других выделений, и при зуме они то появлялись, то пропадали
                    np.copyto(roi_, pc_, where=(pc_.max(2) > 6)[..., None])
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
                src = self.thumbs[i]
            else:
                src, got_ = self.cached(i, lv)              # нет нужной копии: пока — что есть, её — фоном
                if got_ != lv:
                    with self.lock:
                        if (i, lv) not in self.want:
                            self.want.append((i, lv))
                            if len(self.want) > 64:         # ушли дальше — старые просьбы не нужны
                                del self.want[0]
                    self.wake.set()
                if src is None:
                    src = self.thumbs[i]
            k = src.shape[1] / W
            sx0, sy0 = int((ax0 - px) * k), int((ay0 - py) * k)
            sx1, sy1 = max(sx0 + 1, int(math.ceil((ax1 - px) * k))), max(sy0 + 1, int(math.ceil((ay1 - py) * k)))
            crop = src[sy0:sy1, sx0:sx1]
            dx0, dy0 = int(round((ax0 - x0) * s)), int(round((ay0 - y0) * s))
            dx1, dy1 = int(round((ax1 - x0) * s)), int(round((ay1 - y0) * s))
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            crop = cv2.resize(np.ascontiguousarray(crop), (dx1 - dx0, dy1 - dy0),
                              interpolation=cv2.INTER_AREA if dx1 - dx0 < crop.shape[1] else
                              (cv2.INTER_NEAREST if k >= 1 else cv2.INTER_LINEAR))
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            out[dy0:dy1, dx0:dx1] = crop[:dy1 - dy0, :dx1 - dx0]
        return out


# одна буква — один смысл в обоих окнах: клавиша -> внутреннее действие просмотра
VIEW_KEYS = {'s': 'q', 'S': 'q', 'J': 's', 'x': 'e', 'X': 'e', 'o': 'g', 'O': 'G', '[': 'o', ']': 'O', 'a': 'l',
             'A': 'l', 'g': 'k', 'G': 'k', 'C': 'f', 'b': '', 'B': '', 'e': '', 'q': '', 'l': '', 'f': '', 'k': '',
             'i': '', 'w': '', 'v': ''}
MICRO_ONLY = {'e': "stage up", 'q': "stage down", 'f': "autofocus", 'k': "focus point here", 'l': "light",
              'i': "laser", 'w': "white balance", 'v': "open the view", ' ': "picture", 'E': "stage up",
              'Q': "stage down", 'F': "autofocus", 'K': "focus point", 'L': "light", 'I': "laser", 'W': "white balance",
              'V': "open the view"}
HELP_V2 = "Arrows / drag: move  Wheel: zoom  0: whole  , / .: selection  Ctrl+click: mark  H: all keys  Esc: close"
HELP_MODE = {
    'look': "C: crosses  M: select cells (Enter: a selection)  U: outline (a selection by corners)  N: numbers  T: level  R: ruler  [ ]: transparency  A: align  O: objective  S: scan (queue)  X: export  Y: others",
    'cross': "Click: cross exactly there (2 in a row = tilt)  Shift+click: snap to centre  Drag a mark: move  Right click: remove  Del: start again  Shift+C: frames onto the crosses  C / Esc: done",
    'select': "Click: square  Drag: box  Shift: unselect  Right click: remove a square (Shift: the whole area)  Ctrl+A: all / none  Enter: make a selection  Esc: done",
    'stitch': "Click a seam dot: seam off/on  Click a frame: its row  Shift+click: its column  Arrows: 1 px  Shift+arrows: 10 px  Backspace: automatic  J: save",
    'paint': "Drag a rectangle: select all squares in it  Shift+drag: unselect  Click: one square  Right click: remove one (Shift: whole area)  Ctrl+A: all / none  Enter: make a selection  Esc: done",
    'label': "Click a square, type its number X Y  Enter: set  Esc: done",
    'ruler': "R: point-point / line-line / point-line / off. Click the points: the distance (and the angle)  Drag: move  Esc: off",
    'align': "Drag / arrows (Shift: x10): move the other pictures onto this scan  Space: by itself  Enter or A: save  Backspace: no shift  Esc: cancel",
    'poly': "Click the corners (sides only across or up / down, snapping to the crosses and the points: grey lines)  Click the first point or Enter: close  Backspace / right click: back  Esc: cancel",
}
HELP_V = ("C: crosses   Shift+C: frames onto the crosses   M: select cells, Enter: a selection   N: number squares (X Y)   "
          "R: ruler   S: scan (queue)   X: export   Esc: leave the mode")
HELP_ALL = [
    ("Same keys", "One letter means one thing in both windows; keys of the microscope only (E Q F K L I W V Space) say so here"),
    ("Look", "Arrows or drag (right / middle in any mode): move   Wheel, + / -: zoom   0: whole"),
    ("", "T: level (tilt removed, from the crosses) / as shot   , / .: previous / next selection   H: this help   Esc: leave the mode"),
    ("Undo", "Ctrl+Z: undo any change (crosses, grid, squares, numbers, removed selection)   Ctrl+Y: redo"),
    ("Exit", "Esc (when nothing else is open: a mode, the help, a list), or close the window"),
    ("Timing", "` (or F12): ms per frame for each part of the window (also in the log every 5 s): what is slow on this PC"),
    ("Stitch", "J: edit the stitch by hand - seam dots (green < 0.5 px, yellow < 1.5, red; grey = off): click to switch a seam; "
               "click a frame: its row (Shift: its column), arrows: 1 px (Shift: 10), Backspace: automatic, J: save"),
    ("", "Shift+J: exact (quality) stitch of this scan - into the queue of the microscope window (it re-shoots blurry frames)   G: straight edges (crop the steps)"),
    ("Crosses", "C: mark crosses: click two neighbouring crosses in one row = tilt; more crosses far apart = more exact"),
    ("", "The mark goes exactly where you click   Shift+click: snap to the cross centre   Drag a mark: move it"),
    ("", "Right click: remove a cross   Backspace: the last one   Del: start again   Shift+C: frames onto the crosses (exact)"),
    ("Grid", "One grid for the whole sample: from the crosses marked last (on any selection); the others take it as is"),
    ("Outline", "U: a selection by its corners: every side only across or up / down, snapping to the crosses and to the points put (grey lines); click the first point or Enter - it closes itself (the last corner is added)"),
    ("Numbers", "N: click a square, type the number written next to it (X Y), Enter. Two squares fix all numbers"),
    ("Select", "M: click a square (again: off) / drag a box (Shift: unselect)   Ctrl+A: all / none   Enter: they become a selection"),
    ("", "Right click on green: remove that square (or cell)   Shift + right click: the whole green area"),
    ("Ruler", "R: point-point -> line-line -> point-line -> off; click the points"),
    ("Layers", "All pictures in the order of the list (top = on top); the chosen card - on top for now   Y: the others on / off"),
    ("", "[ / ]: the picture of the chosen selection (or the marked ones) more / less transparent (100 / 75 / 50 / 25 / 0 % = outline)"),
    ("", "A: align the chosen selection (this scan) with the others: drag it (or arrows) - only it moves; Space - by the picture itself, Enter - save (its outline moves with it)"),
    ("Many", "Ctrl+click selections (picture or cards): mark several; Esc: unmark. For the marked (or the chosen one): "
             "O - objective (draft, 4x ... 100x), S - queue their scans, [ ] - transparency, Del - delete (not scanned)"),
    ("", "Shift+O: the objective of new selections made from cells (M, Enter): draft or 4x ... 100x"),
    ("Export", "X: everything ticked on the cards -> <sample>/export/: all ticked must be quality: pictures, mixed.png (pixel of the finest scan), the drawing - when the crosses are marked"),
    ("Scans", "... on a card: all scans of the selection; the star = the main one (maps, view, export); View - open it"),
    ("Cards", "Click: open that selection   Drag up / down: the layer order   Box: in the export   Chip: objective   x: delete"),
    ("Queue", "The status bar shows the queue of the microscope window; S puts scans there, Shift+J a quality stitch"),
    ("Profiles", "P: two profile lines (cyan, pink): 4 clicks, then R, G, B along them, averaged over a band: edges, width, contrast"),
]


_PREV = {}


def _preview(folder, w):
    """Картинка образца для карусели: обзорная мозаика (после сшивки) или первый кадр."""
    pv = os.path.join(folder, 'join', 'preview_all.jpg')
    from . import samples as _SM
    gp_ = os.path.join(folder, 'grid.json')
    key = (folder, _SM.sample_uid(folder), _mt(pv), w,   # uid: не путать
           _mt(gp_))
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
                        cv2.fillPoly(ov, [np.int32(np.round(q * kk))], _SM.dim(_SM.sample_color(folder), 0.6),
                                     cv2.LINE_8)
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
                lines.append(f"selected {len(d.get('selected', []))}")
            except Exception:
                pass
        else:
            try:                                            # сетка образца одна: уже есть — она будет и здесь
                lat_ = json.load(open(os.path.join(os.path.dirname(os.path.dirname(f)), 'sample.json'),
                                      encoding='utf-8')).get('lattice') or {}
            except (OSError, ValueError):
                lat_ = {}
            lines.append("grid of the sample" if lat_ and not lat_.get('stage') else "no grid yet (C)")
        items.append(dict(title=f"Selection {SM.sample_index(f)}", color=SM.sample_color(f),
                          lines=lines, thumbs=[_preview(f, SM.CAR_W - 36)], confirm=car.get('confirm') == f, go=False))
    return items


def view(folder, fullscreen=False, _keys=None):
    """Окно; если что-то упало — полный текст ошибки пишется в журнал (logs/MC_*.log), окно закрывается."""
    try:
        return _view(folder, fullscreen=fullscreen, _keys=_keys)
    except Exception:
        import traceback
        L.event('VIEW-999', "The window stopped with an error:\n" + traceback.format_exc(), 40)
        try:
            cv2.destroyWindow(WIN)                          # только своё окно: окно микроскопа остаётся
        except Exception:
            pass
        raise


def _mt(p):
    """Время изменения файла или 0 (нет его — или его как раз переписывает очередь в другом процессе)."""
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0


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
    log_ = os.path.join(smp.folder if smp else folder, '.view_error.log')   # упало — окно микроскопа скажет почему
    pr_ = subprocess.Popen([sys.executable, '-c', f"import microcontrol as mc; mc.view({folder!r})"], cwd=base,
                           env=env, stderr=open(log_, 'w', encoding='utf-8'), **kw)
    pr_.error_log = log_
    return pr_


def _view(folder, fullscreen=False, _keys=None):
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
        if not fullscreen:                                  # окно могли растянуть: заставка — его размера, не растянутая
            Wv, Hv = window_size(WIN, (Wv, Hv))
            car['size'] = (Wv, Hv)
        car['open_label'] = f"Selection {SM.sample_index(cur)}" if ses else os.path.basename(cur)
        car['open_t'] = _t.monotonic()
        _open_frame(car)                                    # пока грузится — видно, что идёт работа
        cur = _view_one(cur, fullscreen, keys, car)
    if not fullscreen:
        window_size(WIN, remember=True)                                    # запомнить размер для следующего окна
    try:                                                   # окно уже закрыли крестиком — ничего
        cv2.destroyWindow(WIN)
    except cv2.error:
        pass
    cv2.waitKey(1)


_ARROWS = {2424832: (-1, 0), 65361: (-1, 0), 2555904: (1, 0), 65363: (1, 0),
           2490368: (0, -1), 65362: (0, -1), 2621440: (0, 1), 65364: (0, 1)}


def _arrow(key):
    """Стрелка -> (dx, dy) или None (с Shift / Ctrl на Linux в старших битах — тоже)."""
    if key is None or key < 0:
        return None
    return _ARROWS.get(key) or _ARROWS.get(key & 0xFFFF)


def _seed(folder, g, LAT, smp0, sc, px_um, need_crosses=False, snap=True):
    """Сетка без своих крестов — из решётки LAT (выделения, большего вокруг, образца или по столику):
    подставлена (seeded), кисть прежних съёмок перенесена, затем притянута к крестам этих кадров, если они есть.
    need_crosses (сама, при открытии): в этих кадрах не нашлось ни одного креста — сетки нет, grid.json как был.
    -> (Grid, сообщение)."""
    from . import lattice as LT
    from .mask import Grid
    import time as _t
    msg_ = ""
    gp0_ = os.path.join(folder, 'grid.json')
    old_ = open(gp0_, encoding='utf-8').read() if os.path.exists(gp0_) else None
    LT.seed_grid(folder, g, LAT)
    if not LAT.get('numbered'):                             # номеров нет — подписи не выдумывать
        g.labels = {}
    g.save(px_um)
    d_ = json.load(open(os.path.join(folder, 'grid.json'), encoding='utf-8'))
    d_['seeded'] = True
    d_['brush_t'] = _t.time()                               # новая съёмка — теперь главная для кисти
    d_['seeded_labels'] = [list(c_) for c_ in g.labels]
    try:                                                    # кисть прежних съёмок — в новую сетку
        from .layers import brush_into_grid
        g2_ = Grid(folder)
        rf0_ = brush_into_grid(smp0.data.get('selections', []), LAT, folder, g2_) if smp0 is not None else None
        d_['selected'] = [list(c) for c in sorted(g2_.selected)]
        if rf0_:
            d_['refine'] = rf0_
    except Exception:
        pass
    with open(os.path.join(folder, 'grid.json'), 'w', encoding='utf-8') as f_:
        json.dump(d_, f_, indent=1)
    g = Grid(folder)
    nfx_ = 0
    if not LAT.get('stage') and snap:
        try:                                                # по координатам столика — на кресты этих кадров
            nfx_, dfx_ = g.snap_to_crosses(lambda x, y, w, h: sc.render(x, y, 1.0, w, h), sc.size, px_um,
                                           render_s=lambda x, y, s_, w, h: sc.render(x, y, s_, w, h))
            if nfx_:
                g.save(px_um)
                msg_ = (f"   The grid snapped to {nfx_} crosses of this scan "
                        f"(moved by {np.hypot(*dfx_) * px_um:.2f} um); Shift+C: frames onto the crosses")
        except Exception:
            nfx_ = 0
        if need_crosses and not g.crosses_seen():               # крестов здесь не видно — сетку не рисовать
            if old_ is None:
                os.remove(gp0_)
            else:
                with open(gp0_, 'w', encoding='utf-8') as f_:
                    f_.write(old_)
            return Grid(folder), ("   No crosses in this scan: no grid drawn (C: mark crosses; M: squares by the "
                                  "stage coordinates)")
    return g, msg_


def _seed_mark(folder, LAT):
    """Подставленной сетке — отпечаток решётки и совмещения (поменяются — подставить заново). -> Grid."""
    from .layers import lattice_key
    from .mask import Grid
    gp_ = os.path.join(folder, 'grid.json')
    d_ = json.load(open(gp_, encoding='utf-8'))
    d_['seed_key'] = lattice_key(LAT, folder)
    with open(gp_, 'w', encoding='utf-8') as f_:
        json.dump(d_, f_, indent=1)
    return Grid(folder)


def stage_lattice(smp0, pitch):
    """Решётка по координатам столика (крестов нет): квадраты pitch мкм от точки 0, 0. Запоминается в образце,
    пока не отметят настоящие кресты (тогда решётка — по ним)."""
    L = dict(O=[0.0, 0.0], e1=[float(pitch), 0.0], e2=[0.0, float(pitch)], pitch=float(pitch), numbered=False,
             stage=True, src=None)
    if smp0 is not None and smp0.data.get('lattice') is None:
        smp0.data['lattice'] = L
        smp0.save()
    return L


def _open_frame(car):
    """Кадр заставки «Opening»: бегущая полоса (сколько ждать — неизвестно), сколько прошло."""
    try:
        Wv, Hv = car['size']
        ld_ = np.full((Hv, Wv, 3), 24, np.uint8)
        UI.progress_panel(ld_, "Opening", car.get('open_label', ''), None, "loading the stitched picture and its grid",
                          _t.monotonic() - car.get('open_t', _t.monotonic()), None, "")
        cv2.imshow(WIN, ld_)
        cv2.waitKey(1)
    except Exception:
        pass


def _opening(fn, car):
    """fn() в своём потоке; окно тем временем рисует бегущую полосу (~25 кадров/с). -> результат fn."""
    import threading
    box = {}

    def run():
        try:
            box['r'] = fn()
        except BaseException as e:
            box['e'] = e
    th = threading.Thread(target=run, daemon=True)
    th.start()
    while th.is_alive():
        _open_frame(car)
        th.join(0.04)
    if 'e' in box:
        raise box['e']
    return box['r']


def _view_one(folder, fullscreen, keys, car):
    """Открыть сшитый скан. Перемещение — левая кнопка (в режимах выделения — правая или средняя), колесо — масштаб.
    G — отметить кресты (щелчок у креста: центр находится сам по симметрии креста); соседний крест по ряду,
    по столбцу и дальние — решётка уточняется. R — повернуть картинку так, чтобы квадраты стали ровными.
    S — выделять квадраты: щелчок — квадрат, рамка — много сразу (Shift — снять); B — кисть: прямоугольники.
    A — все, X — ни одного.
    E — вывести выделенное (mc.export). Разметка хранится в <скан>/grid.json."""
    from .mask import Grid, snap_cross, export
    from . import samples as SM
    sc = _opening(lambda: _Scene(folder), car)             # грузится — бегущая полоса, окно живое
    g = Grid(folder)
    px_um = sc.st.meta.get('pixel_size_um') or 1.0
    moved_ = g.sync(sc.P, sc.W, sc.H) if g.cr.marks else 0   # кадры встали иначе (сшивка): кресты — за ними
    if moved_:
        g.refined = None
    if g.cr.marks:
        g.save(px_um)
    from . import project as P0_
    from . import lattice as LT
    smp0 = P0_.sample_obj(folder)
    from .layers import update_lattice_of, lattice_key
    if smp0 is not None:                                    # сетка образца ОДНА: по крестам, отмеченным руками
        update_lattice_of(smp0)                             # последними (на любом выделении)
    LAT = smp0.data.get('lattice') if smp0 is not None else None
    seeded_, st0_msg = False, ""
    hand_ = LAT is not None and not LAT.get('stage') and LAT.get('src') and os.path.isdir(LAT['src'])
    if not hand_:                                           # сетки образца нет (кресты убрали): подставленная — тоже
        try:
            gp0_ = os.path.join(folder, 'grid.json')
            d0_ = json.load(open(gp0_, encoding='utf-8'))
            if d0_.get('seeded'):
                for k0_ in ('marks', 'index', 'labels', 'selected', 'seeded', 'seed_key', 'seeded_labels'):
                    d0_.pop(k0_, None)
                with open(gp0_, 'w', encoding='utf-8') as f0_:
                    json.dump(d0_, f0_, indent=1)
                g = Grid(folder)
        except (OSError, ValueError):
            pass
    if hand_ and os.path.abspath(LAT['src']) != os.path.abspath(folder):   # кресты — на другой съёмке: её сетка здесь
        try:
            d0_ = json.load(open(os.path.join(folder, 'grid.json'), encoding='utf-8'))
        except (OSError, ValueError):
            d0_ = {}
        key_ = lattice_key(LAT, folder)
        if not (d0_.get('seeded') and d0_.get('seed_key') == key_):     # нет её здесь или устарела (кресты, совмещение)
            try:
                g, st0_msg = _seed(folder, g, LAT, smp0, sc, px_um, snap=False)
                g = _seed_mark(folder, LAT)
                seeded_ = g.ready()
            except Exception:
                pass
    ses, subs = SM.session_of(folder)
    Wv, Hv = car['size']
    Hv -= UI.MENU_H                                         # сверху — строка меню
    try:                                                    # сетка подставлена (из чужой разметки): крестов своих нет
        seeded_ = seeded_ or bool(json.load(open(os.path.join(folder, 'grid.json'), encoding='utf-8')).get('seeded'))
    except (OSError, ValueError):
        pass
    st = dict(cx=sc.size[0] / 2, cy=sc.size[1] / 2, s=None, rot=0.0, mode='look', down=None, box=None,
              poly=None, dirty=True, Wv=Wv - SM.CAR_W, Hv=Hv, straight=False, switch=None, ruler=[], mouse=None,
              msg=car.pop('msg', None) or (f"Selection {SM.sample_index(folder)} ({subs.index(folder) + 1} of {len(subs)})" if len(subs) > 1 else "")
              + (f"   Crosses moved with the stitched frames ({moved_}); Shift+C: frames onto the crosses" if moved_ else "")
              + ("   The grid of the sample (crosses marked on another selection); the picture is off: A aligns it; "
                 "C here: these crosses make the grid for all" if seeded_ else "")
              + st0_msg)
    st['seeded'] = seeded_
    from .util import Prof, PROF_KEYS
    st['timing'] = car.setdefault('timing', Prof('view'))     # `: замер по этапам (живёт, пока открыт просмотр)

    def car_click(x, y, ctrl=False):
        h_ = SM.hit(car.get('hits', []), x - st['Wv'], y)
        if not h_:
            return
        act, k = h_
        mp_ = car.get('map')
        if ctrl and act == 'sel' and mp_ is not None and 0 <= k < len(mp_):   # Ctrl+щелчок: отметить для дела разом
            u_ = mp_[k][2] if mp_[k][0] == 'sub' and len(mp_[k]) > 2 else (mp_[k][1] if mp_[k][0] == 'draft' else None)
            if u_:
                mark_toggle(u_)
            return
        u_k_ = (mp_[k][2] if mp_[k][0] == 'sub' and len(mp_[k]) > 2 else mp_[k][1]) if mp_ is not None and 0 <= k < len(mp_) else None
        if act == 'more' and u_k_:                          # «...»: все съёмки выделения, звезда — основная
            st['msg'] = pick_open(u_k_)
            st['car_dirty'] = st['dirty'] = True
            return
        if act == 'alpha' and u_k_:                         # прозрачность на карточке: 100 -> 75 -> ... -> 0 -> 100
            lv_ = [1.0, 0.75, 0.5, 0.25, 0.0]
            a0_ = alphas().get(u_k_, 1.0)
            i_ = min(range(len(lv_)), key=lambda i2_: abs(lv_[i2_] - a0_))
            a1_ = lv_[(i_ + 1) % len(lv_)]
            alphas({u_k_: a1_})
            st['msg'] = f"Picture {100 * a1_:.0f} %" + (" (outline only)" if a1_ == 0 else "") + "   (click: less, [ ]: more / less)"
            st['_fl_key'] = None
            st['car_dirty'] = st['dirty'] = True
            return
        if act == 'chk' and u_k_:                           # галочка: в экспорт или нет
            st['msg'] = set_export(u_k_)
            st['car_dirty'] = st['dirty'] = True
            return
        if act == 'obj' and u_k_:                           # плашка объектива: следующий
            from .config import OBJECTIVES
            cyc_ = ['draft'] + sorted(OBJECTIVES)
            r0_ = next((r_ for r_ in regions_now()[0] if r_.get('uid') == u_k_), None)
            if r0_ is not None:
                cur_ = 'draft' if r0_.get('kind') == 'draft' else r0_.get('obj')
                st['msg'] = set_objective([u_k_], cyc_[(cyc_.index(cur_) + 1) % len(cyc_)] if cur_ in cyc_ else cyc_[1])
            st['car_dirty'] = st['dirty'] = True
            return
        if mp_ is not None and 0 <= k < len(mp_):           # в меню и черновики: карточка -> съёмка
            if mp_[k][0] != 'sub':
                if act == 'del':                            # крестик: убрать (Ctrl+Z вернёт квадраты)
                    st['msg'] = delete_draft(mp_[k][1], mp_[k][2])
                    car.pop('focus', None)
                    st['car_dirty'] = True
                elif mp_[k][1] and not str(mp_[k][1]).startswith('cell:'):
                    car['focus'] = mp_[k][1]                # не снятое: выбрано (рамка на картинке, дела — ему)
                    st['msg'] = ("Not scanned yet: O - its objective, S - queue its scan, Del - delete; "
                                 "Ctrl+click - mark several")
                    st['car_dirty'] = True
                st['dirty'] = True
                return
            k = mp_[k][1]
        if not 0 <= k < len(subs):                          # список сменился (удалили в окне микроскопа)
            st['car_dirty'] = st['dirty'] = True
            return
        f = subs[k]
        if act == 'sel':
            car['confirm'] = None
            car.pop('focus', None)
            if f != folder:
                st['switch'] = f
        elif act == 'del':                                  # сразу, без подтверждения (Ctrl+Z вернёт)
            car['confirm'] = None
            g.save(px_um) if f == folder else None
            if f == folder:
                sc.close()
            dst = SM.remove_sample(f)
            car['hist'].push(f"remove Selection {SM.sample_index(dst)}", None, None, kind='remove', src=f, dst=dst)
            st['acted'] = False                             # это действие уже в истории
            rest = [s_ for s_ in subs if s_ != f]
            st['msg'] = f"Selection {SM.sample_index(dst)} removed to {dst} (Ctrl+Z: bring it back)"
            st['switch'] = (folder if f != folder else (rest[min(k, len(rest) - 1)] if rest else False))
        st['dirty'] = True

    def queue_quality():
        """J на quick: точная сшивка (quality) этой съёмки — в очередь окна микроскопа (там прибор: доснять
        размытые кадры); своей очереди нет — запросом через файл образца."""
        from .jobs import _ALL
        q_ = next((q2_ for q2_ in _ALL if getattr(q2_.m, '_sample', None) is not None and smp is not None and
                   os.path.abspath(q2_.m._sample.folder) == os.path.abspath(smp.folder)), None)
        if q_ is not None:
            same_ = (sc.st.meta.get('objective') or 0) == getattr(q_.m, 'objective', None)
            q_.quality(folder, title="Quality stitch " + os.path.basename(folder), reshoot=same_)
            return "Quality stitch queued: when it is done, J edits it"
        if smp is None:
            return "The quality stitch needs the sample of this scan (and the microscope window)"
        with open(os.path.join(smp.folder, '.queue_requests.jsonl'), 'a', encoding='utf-8') as fo_:
            fo_.write(json.dumps(dict(op='quality', folder=os.path.abspath(folder))) + "\n")
        return ("Quality stitch sent to the queue of the microscope window (it re-shoots blurry frames at the "
                "instrument); when it is done, J edits it")

    def pick_open(uid):
        """Список съёмок выделения uid посреди окна (как в окне микроскопа)."""
        from .project import scan_table
        r_ = next((x_ for x_ in regions_now()[0] if x_.get('uid') == uid), None)
        if r_ is None or smp is None:
            return "This selection is gone"
        rows_, fs_, i_ = scan_table(smp.folder, r_)
        if not rows_:
            return f"Selection {r_.get('num', '')}: no scans yet"
        st['pick'] = dict(uid=uid, rows=rows_, folders=fs_, star=i_, scroll=max(0, len(rows_) - 14),
                          title=f"Scans of Selection {r_.get('num', '')} (oldest first)")
        return f"Selection {r_.get('num', '')}: {len(rows_)} scan{'s' if len(rows_) != 1 else ''} - the star is the main one"

    def pick_do(c_):
        """Щелчок в списке съёмок: звезда / строка — основная (sample.json: окно микроскопа берёт сам), View —
        открыть её здесь, x / мимо / Esc — закрыть."""
        pk_ = st.get('pick')
        if pk_ is None or c_ == 'panel' or c_ is None:
            return None
        if c_ == 'close' or not isinstance(c_, tuple):
            st.pop('pick', None)
            return None
        what_, i_ = c_
        fs_ = pk_['folders']
        if not 0 <= i_ < len(fs_):
            return None
        if what_ == 'view':
            st.pop('pick', None)
            if os.path.abspath(fs_[i_]) != os.path.abspath(folder):
                st['switch'] = fs_[i_]
            return f"View: {os.path.basename(fs_[i_])}"
        from . import project as PJ_
        s_ = PJ_.sample_obj(folder)
        for r_ in s_.data.get('selections', []):
            if r_.get('uid') == pk_['uid']:
                if i_ == len(fs_) - 1:
                    r_.pop('main', None)
                else:
                    r_['main'] = os.path.basename(fs_[i_])
        s_.save()
        pk_['star'] = i_
        st['_fl_key'] = None
        return f"Main scan: {os.path.basename(fs_[i_])} (maps, view, export)"

    def delete_draft(uid, k_sub):
        """Черновик из списка: его квадраты (или клетки подсетки) больше не выбраны в съёмке, где их выбрали,
        запись образца (и его черновики внутри) — прочь сразу. -> сообщение."""
        from . import lattice as LT
        from .snap import mosaic_to_stage
        regs_, rf_ = regions_now()
        r_ = next((x_ for x_ in regs_ if x_.get('uid') == uid), None)
        if r_ is None:
            return "This draft is gone already"
        cells_ = {tuple(c) for c in r_.get('cells') or []}
        name_ = "The new selection" if r_.get('pending') else f"Selection {r_.get('num', '')}"
        f_ = subs[k_sub] if 0 <= k_sub < len(subs) else folder
        gone_ = 0
        if r_.get('factors'):                               # клетки подсетки выделения-родителя
            for f2_ in [f_] + [x_ for x_ in subs if x_ != f_]:
                gp_ = os.path.join(f2_, 'grid.json')
                try:
                    d_ = json.load(open(gp_, encoding='utf-8'))
                except (OSError, ValueError):
                    continue
                rq_ = (d_.get('refine') or {}).get(r_.get('parent'))
                if not rq_:
                    continue
                have_ = {tuple(c) for c in rq_.get('cells', [])}
                if f2_ == folder:
                    write_refine(r_['parent'], rq_.get('n'), have_ - cells_)
                else:
                    rq_['cells'] = [list(c) for c in sorted(have_ - cells_)]
                    d_['brush_t'] = _t.time()
                    with open(gp_, 'w', encoding='utf-8') as fo_:
                        json.dump(d_, fo_, indent=1)
                gone_ += len(have_ & cells_)
        else:                                               # квадраты литографии этой съёмки
            if f_ == folder:
                g_, L_ = g, lat_now()
            else:
                g_ = Grid(f_)
                L_ = None
                try:
                    from .layers import lattice_in, _sel_dir
                    regs0_ = smp0.data.get('selections', []) if smp0 is not None else []
                    me2_ = next((x_ for x_ in regs0_ if x_.get('kind') not in ('group', 'draft') and
                                 os.path.abspath(f_).startswith(os.path.abspath(_sel_dir(f_, x_)) + os.sep)), None)
                    L_ = (lattice_in(smp0, regs0_, me2_, self_too=True) if me2_ is not None else None) or LAT
                except Exception:
                    L_ = LAT
            if L_ is not None and g_.ready() and g_.selected:
                sq_ = sorted(g_.selected)
                uv_ = LT.to_uv(L_, mosaic_to_stage(f_, np.array([g_.cr.point(i + 0.5, j + 0.5) for i, j in sq_])))
                off_ = {c for c, (u_, v_) in zip(sq_, uv_) if (int(np.floor(u_)), int(np.floor(v_))) in cells_}
                if off_:
                    g_.selected -= off_
                    try:
                        from .storage import ScanData as _SD
                        px2_ = px_um if f_ == folder else (_SD(f_).meta.get('pixel_size_um') or 1.0)
                    except Exception:
                        px2_ = px_um
                    g_.save(px2_)
                    gp_ = os.path.join(f_, 'grid.json')
                    d_ = json.load(open(gp_, encoding='utf-8'))
                    d_['brush_t'] = _t.time()               # окно микроскопа: кисть этой съёмки — свежая
                    with open(gp_, 'w', encoding='utf-8') as fo_:
                        json.dump(d_, fo_, indent=1)
                    gone_ = len(off_)
        if not r_.get('pending') and smp is not None:       # запись образца — сразу (и черновики внутри неё)
            try:
                d_ = json.load(open(smp.file, encoding='utf-8'))
                sel_ = d_.get('selections', [])
                dead_, grew_ = {uid}, True
                while grew_:
                    grew_ = False
                    for x_ in sel_:
                        if x_.get('kind') == 'draft' and x_.get('parent') in dead_ and x_['uid'] not in dead_:
                            dead_.add(x_['uid'])
                            grew_ = True
                d_['selections'] = [x_ for x_ in sel_ if x_.get('uid') not in dead_]
                smp.data = d_
                smp.save()
            except (OSError, ValueError) as e_:
                return f"{name_}: its squares are unselected, but the sample file could not be written: {e_}"
        return f"{name_} deleted ({gone_} square{'s' if gone_ != 1 else ''} unselected; Ctrl+Z: select them again)"

    st['_t'] = dict(delete_draft=lambda *a_: delete_draft(*a_), regions=lambda: regions_now()[0],
                    seam_pt=lambda k_: seam_pt(k_))   # для тестов

    def VIEW_M():
        """Мозаика -> вид, одинаковый для всех съёмок и такой же, как в камере и на карте: знаковая перестановка
        осей, при которой сдвиг столика идёт по экрану как в камере сейчас (Q = util.orient_of(K); окно микроскопа
        пишет её в настройки). Без неё — +X столика вправо, +Y вниз."""
        if st.get('_V') is None:
            V_ = np.eye(2)
            try:
                from .snap import geometry
                from .util import ui_pref as _uq
                q_ = _uq('orient')
                st['_Vq'] = q_
                Q_ = np.array(q_, float).reshape(2, 2) if q_ else np.eye(2)
                Mo_ = np.asarray(geometry(folder)['Mo'], float)        # px мозаики на шаг: столбцы — оси столика
                best_ = None
                for P_ in (np.eye(2), np.array([[0, 1], [1, 0]])):
                    for sx_ in (1, -1):
                        for sy_ in (1, -1):
                            Vc_ = np.diag([sx_, sy_]) @ P_
                            sc_ = np.trace(Q_.T @ Vc_ @ Mo_)       # после V: столик идёт по экрану как Q
                            if best_ is None or sc_ > best_[0]:
                                best_ = (sc_, Vc_)
                V_ = best_[1].astype(float)
            except Exception:
                pass
            st['_V'] = V_
        return st['_V']

    def R():
        t = np.radians(st['rot'])
        return VIEW_M() @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    mid = lambda: np.array([st['Wv'] / 2, st['Hv'] / 2])
    to_view = lambda M: mid() + st['s'] * (R() @ (np.asarray(M, float) - (st['cx'], st['cy'])))
    from_view = lambda v: np.array([st['cx'], st['cy']]) + R().T @ ((np.asarray(v, float) - mid()) / st['s'])
    st['tv'] = to_view                                      # для проверок: мозаика -> окно

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
        st['s'] = min(32.0, max(1e-4, st['s'] * f))           # до 32 точек экрана на пиксель: клетки 1/10 видно
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
              'fit grid': "finding every cross and fitting the square grid",
              'crosses: finding': "finding the crosses in every frame", 'crosses': "putting the frames onto the crosses",
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
        from .util import ui_pref as _uip
        if _uip('list') == 'left':                          # раскладка: список слева (как в окне микроскопа)
            img = np.ascontiguousarray(np.hstack([img[:, img.shape[1] - SM.CAR_W:], img[:, :img.shape[1] - SM.CAR_W]]))
        st['_list_left'], st['_fullW'] = _uip('list') == 'left', img.shape[1]
        pk_ = st.get('pick')
        if pk_:                                             # список съёмок выделения — поверх всего
            img = img.copy()
            from .project import SCAN_HEAD
            pk_['rects'] = UI.pick_panel(img, pk_['title'], SCAN_HEAD, pk_['rows'], pk_['star'], pk_.get('hover'),
                                         pk_.get('scroll', 0), note="Click a star (or a row): the main scan - maps, "
                                         "view, export. View: open it here. Esc: close")
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

    PLANS = {                                              # этапы одного дела и их доли: общая полоса, не 0 -> 100 % на этапе
        'stitch': [("Stitch: light", 0.08), ("Stitch: seams", 0.62), ("Stitch: brightness", 0.08), ("Stitch: saving", 0.22)],
        'cross': [("Crosses: finding", 0.45), ("Crosses: solving", 0.2), ("Stitch: saving", 0.35)],
    }

    def busy(title, fn, close_scene=False, footer="", plan=None):
        """Долгая работа прямо в окне: fn() в фоновом потоке, в окне — этап, полоса 0-100 % (общая по этапам plan),
        время, кнопка Stop (Esc) и строка состояния внизу. Возвращает (результат, ошибка или None); Stop — ошибка
        Stopped."""
        from .util import Progress
        base = st.get('scene')
        if base is None:
            base = render()[:, :st['Wv']]
        base = (base.astype(np.float32) * 0.35).astype(np.uint8)
        pan = carousel_img()
        if close_scene:                                     # файлы скана перепишутся (Windows не даст при открытых)
            sc.close()
            sc.thumbs, sc.over = None, None
            sc.cache.clear()
        fs_ = 0.5
        sbh_ = UI.size("Xg", fs_)[1] + 16

        def frame(stt):
            img = base.copy()
            nm_ = stt['stage'].replace("Stitch: ", "").lower()
            stage = STAGES.get(nm_, stt['stage'] if stt['stage'] != title else "")
            r_ = UI.progress_panel(img, title, stage, stt['frac'], stt['text'], _t.monotonic() - stt['t0'], stt['left'],
                                   footer, stopping=stt['stopping'])
            r2_ = UI.status_bar(img, 0, img.shape[0] - sbh_, img.shape[1], img.shape[0], stt, scale=fs_)
            out = np.hstack([img, pan]) if pan.shape[0] == img.shape[0] else img
            ttl = (smp.title + f"  [{smp.project.uid}]") if smp else os.path.basename(folder)
            mb, _h = UI.menu_bar(out.shape[1], ttl, ["Open", "Scans", "Info", "Help"], None)
            hit_ = r_ if (r_ and not stt['stopping']) else r2_
            hit_ = (hit_[0], hit_[1] + mb.shape[0], hit_[2], hit_[3] + mb.shape[0]) if hit_ else None
            return np.vstack([mb, out]), hit_
        try:
            res = UI.run_busy(WIN, title, fn, frame, plan=plan, wait_ms=60)
        finally:
            cv2.setMouseCallback(WIN, mouse_cb)
        st['busy_s'] = res['seconds']
        return res.get('r'), res.get('e')

    def fit_here():
        """F: кадры этого выделения — на кресты (anchor.fit_crosses): решётка, положения, дисторсия; окно
        потом открывается заново."""
        from .anchor import fit_crosses
        g.save(px_um)
        r, e = busy("Frames onto the crosses", lambda: fit_crosses(folder, verbose=True), close_scene=True,
                    footer="(the view reopens by itself)", plan=PLANS['cross'])
        st['stitched'] = True
        if e is not None:
            car['msg'] = f"Frames onto the crosses failed: {type(e).__name__}: {e}"
        else:
            on_ = next((k_ for k_ in r if k_.startswith('On the grid')), None)
            car['msg'] = (f"Frames onto the crosses in {st['busy_s']:.0f} s: {r[on_].split('-> ')[-1]} of the crosses "
                          f"within 0.5 um of the grid; distortion {r.get('Distortion at the edge, px', '-').split('-> ')[-1]} "
                          f"px at the edge" if on_ else "Frames onto the crosses: done")
        st['switch'] = folder

    def stitch_here():
        """S: сшить это выделение (mc.stitch) прямо из просмотра; затем окно открывается заново уже со сшитой
        картинкой (кресты переезжают вместе с кадрами)."""
        from .stitch import stitch as _stitch
        g.save(px_um)
        r, e = busy(f"Stitching Selection {SM.sample_index(folder)}" if len(subs) > 1 else "Stitching",
                    lambda: _stitch(folder, verbose=True), close_scene=True, footer="(the view reopens by itself)",
                    plan=PLANS['stitch'])
        st['stitched'] = True
        if e is not None:
            car['msg'] = f"Stitching failed: {type(e).__name__}: {e}"
        else:
            car['msg'] = (f"Stitched in {st['busy_s']:.0f} s: seams {r.get('Seams accepted', '-')}, "
                          f"error {r.get('Seam error, px', '-')} px; " +
                          ("Shift+C: frames onto the crosses (exact)" if g.ready() else "C: mark 2-3 crosses, then Shift+C: frames onto the crosses (exact)"))
        st['switch'] = folder

    RK_N = {'pp': 2, 'll': 4, 'pl': 3}                     # линейка: сколько точек у вида измерения
    RK_NAME = {'pp': "point - point", 'll': "line - line", 'pl': "point - line"}

    def ruler_geom(P):
        """Измерение по точкам P (px мозаики) для вида st['rk'] -> dict(text, segs: отрезки для рисования) или None."""
        rk = st.get('rk', 'pp')
        pc_ = sc.st.meta.get('pixel_size_cal')                # пиксель, откалиброванный по крестам (F)
        k_ = pc_ or px_um
        tail_ = "" if pc_ else "  (pixel not calibrated: F)"
        P = [np.asarray(p_, float) for p_ in P]
        if rk == 'pp' and len(P) >= 2:
            d = (P[1] - P[0]) * k_
            return dict(text=f"{np.hypot(*d):.3f} um   (dx {d[0]:+.3f}, dy {d[1]:+.3f})" + tail_, segs=[(P[0], P[1])])

        def foot(p, a, b):                                    # основание перпендикуляра из p на прямую ab
            u = (b - a) / max(1e-9, np.hypot(*(b - a)))
            return a + u * float((p - a) @ u), u
        if rk == 'pl' and len(P) >= 3:
            f_, u_ = foot(P[2], P[0], P[1])
            d = np.hypot(*(P[2] - f_)) * k_
            along = float((f_ - P[0]) @ u_) * k_
            return dict(text=f"{d:.3f} um from the line   (foot {along:+.3f} um from its first point)" + tail_,
                        segs=[(P[0], P[1]), (P[2], f_)], lines=[(P[0], P[1])])
        if rk == 'll' and len(P) >= 4:
            a1, b1, a2, b2 = P[:4]
            u1 = (b1 - a1) / max(1e-9, np.hypot(*(b1 - a1)))
            u2 = (b2 - a2) / max(1e-9, np.hypot(*(b2 - a2)))
            ang = float(np.degrees(np.arccos(np.clip(abs(u1 @ u2), 0, 1))))
            m2, m1 = (a2 + b2) / 2, (a1 + b1) / 2
            f2, _ = foot(m2, a1, b1)                          # середина второго отрезка -> первая прямая
            f1, _ = foot(m1, a2, b2)
            d2, d1 = np.hypot(*(m2 - f2)) * k_, np.hypot(*(m1 - f1)) * k_
            return dict(text=f"{(d1 + d2) / 2:.3f} um between the lines (at the middles: {d2:.3f} / {d1:.3f}), "
                             f"angle {ang:.2f} deg" + tail_,
                        segs=[(a1, b1), (a2, b2), (m2, f2), (m1, f1)], lines=[(a1, b1), (a2, b2)])
        return None

    def ruler_text(a=None, b=None):
        """Подпись линейки (вид st['rk']): расстояние, мкм."""
        P = st['ruler'] if a is None else [a, b]
        r_ = ruler_geom(P)
        return r_['text'] if r_ else ""

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
        st['seeded'] = False
        for c_ in d_.get('seeded_labels') or [tuple(k_) for k_ in g.labels]:
            g.labels.pop(tuple(c_), None)
        g.save(px_um)
        d_ = json.load(open(gp_, encoding='utf-8'))
        d_.pop('seeded', None)
        d_.pop('seeded_labels', None)
        with open(gp_, 'w', encoding='utf-8') as f_:
            json.dump(d_, f_, indent=1)

    def add_cross(v, snap=False):
        """Крест — ровно там, где щёлкнули (середина пикселя экрана); snap (Shift+щелчок) — к центру креста
        на снимке (поиск по шаблону «+»)."""
        if st.get('seeded'):                                # подставленные кресты — не его: свои с нуля
            g.cr.clear()
            g.refined = None
        unseed()
        Mc = from_view((v[0] + 0.5, v[1] + 0.5))
        d = np.zeros(2)
        if snap:                                            # окно поиска — ±30 мкм при любом объективе: на 50x-100x
            k_ = min(1.0, px_um / 0.25)                     # крест в сотни пикселей — окно в уменьшенной копии
            r = int(min(150, max(8, 15.0 * k_ / px_um)))    # (~0.25 мкм/px), иначе шаблон «+» его не накрывает
            reg = sc.render(Mc[0], Mc[1], k_, 4 * r + 1, 4 * r + 1)
            from .mask import meas_gray
            d, resp = snap_cross(meas_gray(reg), r, arm=(g.arm * k_ if g.arm else None), angle=g.angle_deg())
            d = np.asarray(d, float) / k_
            # (в уменьшенной копии пиксель ~0.1 мкм; центр по краям плеч — доли пикселя)
        try:
            was_ = g.ready()
            if was_:
                v1_, v2_ = g.cr.vectors()                   # последняя готовая сетка (её индексы — у выбора, подписей)
                st['regrid'] = (np.array(g.cr.O, float), np.array(v1_, float), np.array(v2_, float))
            ij = g.cr.add(Mc + d)
            g.refined = None                                # решётка — по отметкам (квадратная)
            if not was_ and g.ready() and st.get('regrid') is not None:
                O0_, a0_, b0_ = st['regrid']                # сетка заново: индексы — как у прежней в этом месте,
                ij_ = [tuple(int(v_) for v_ in np.round(np.linalg.solve(np.c_[a0_, b0_], np.asarray(p_) - O0_)))
                       for p_ in g.cr.marks]                # чтобы выбор, подписи и номера остались на своих квадратах
                if len(set(ij_)) == len(ij_):
                    g.cr.ij = ij_
                    g.cr._fit()
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
            if g.ready() and len(g.cr.marks) >= 2:          # подгонка по всем крестам — только по F, не сама
                st['msg'] += "   F: fit the grid to all crosses"
            g.save(px_um)
            if st['straight']:
                set_straight(True)
            if smp0 is not None and g.ready():              # эти кресты — теперь сетка образца для всех съёмок
                update_lattice_of(P0_.sample_obj(folder))
        except ValueError as e:
            st['msg'] = str(e)

    def auto_fit():
        """Сетка по отметкам (2–3 щелчка, ошибка щелчка копится вдоль решётки) -> сразу по ВСЕМ крестам мозаики
        (как F), если их не слишком много для мгновенного ответа."""
        if not g.ready() or len(g.cr.marks) < 2:
            return
        n_nodes = sum(1 for _ in g.cells_range((0, 0), sc.size))
        if n_nodes > 600:
            st['msg'] += "   Shift+C: frames onto the crosses (exact)"
            return
        n_, rms_ = fit_grid()
        if rms_ is not None and n_ >= 4:
            st['msg'] += (f"   Grid fitted to all {n_} crosses: {rms_ * px_um:.2f} um off on average. "
                          "Shift+C: frames onto the crosses (exact)")

    def fit_grid():
        """F: сетка по всем крестам мозаики (mask.Grid.refine) — с полосой прогресса в окне."""
        from .util import Progress

        def run():
            pr = Progress(100, "Fit grid", unit='%')

            def cb(f, txt):
                pr.done = int(100 * f)
                pr.update(0, txt, force=True)
            try:
                return g.refine(lambda x, y, w, h: sc.render(x, y, 1.0, w, h), sc.size, px_um,
                                render_s=lambda x, y, s_, w, h: sc.render(x, y, s_, w, h), progress=cb)
            finally:
                pr.close()
        r_, e_ = busy("Fitting the grid to all crosses", run)
        if e_ is not None:
            st['msg'] = f"Fit grid failed: {type(e_).__name__}: {e_}"
            return 0, None
        return r_

    def cut_outline(img):
        """Снятое за контуром выделения (кадры шире контура — до шага мотора, по краям) не показывается:
        картинка — ровно то, что выделено."""
        mt_ = (_mt(smp.file) if smp is not None else 0, _mt(os.path.join(folder, 'align.json')),
               _mt(os.path.join(folder, 'join', 'positions_all.csv')))
        pm_ = st.get('_outline_px')                         # контур двигали (A, объектив) — заново
        if pm_ is not None and st.get('_outline_mt') != mt_:
            pm_ = None
        st['_outline_mt'] = mt_
        if pm_ is None:
            pm_ = False
            try:
                pg_ = scan_outline()
                if pg_ is not None and not pg_.is_empty:
                    from .snap import stage_to_mosaic
                    ps_ = [pg_] if pg_.geom_type == 'Polygon' else list(getattr(pg_, 'geoms', []))
                    pm_ = [(stage_to_mosaic(folder, np.array(p_.exterior.coords)),
                            [stage_to_mosaic(folder, np.array(h_.coords)) for h_ in p_.interiors]) for p_ in ps_]
            except Exception:
                pm_ = False
            st['_outline_px'] = pm_
        if not pm_:
            return img
        return UI.tint_outside(img, [(to_view_arr(e_), [to_view_arr(h_) for h_ in hs_]) for e_, hs_ in pm_],
                               (24, 24, 24), 1.0)

    def scan_outline():
        """Контур выделения этой съёмки (мкм столика) или None — по нему окно микроскопа принимает квадраты."""
        if smp is None:
            return None
        try:
            regs_ = json.load(open(smp.file, encoding='utf-8')).get('selections', [])
        except (OSError, ValueError):
            return None
        me_ = next((r_ for r_ in regs_ if r_.get('kind') not in ('group', 'draft') and 'A' in r_ and
                    P0_.scan_uid(folder) == r_.get('uid')), None)
        if me_ is None:
            return None
        from shapely.geometry import box as _bx
        from .layers import from_json
        if me_.get('shape'):
            return from_json(me_['shape'])
        (ax_, ay_), (bx_, by_) = me_['A'], me_['B']
        return _bx(min(ax_, bx_) - me_.get('fw', 0) / 2, min(ay_, by_) - me_.get('fh', 0) / 2,
                   max(ax_, bx_) + me_.get('fw', 0) / 2, max(ay_, by_) + me_.get('fh', 0) / 2)

    def inside_only(cells):
        """Квадраты для выделения: любые, и там, где ещё не снимали (новое выделение — чтобы его снять, S).
        -> (взятые, сколько отброшено = 0)."""
        return set(cells), 0
        pg_ = scan_outline()
        if pg_ is None or not cells:
            return set(cells), 0
        from .snap import mosaic_to_stage
        from shapely.geometry import Point as _Pt
        cl_ = sorted(cells)
        S_ = mosaic_to_stage(folder, np.array([g.cr.point(i + 0.5, j + 0.5) for i, j in cl_]))
        ok_ = {c for c, S1 in zip(cl_, S_) if pg_.buffer(0.01).contains(_Pt(*S1))}
        return ok_, len(cl_) - len(ok_)

    def apply_poly(poly_view, on, touch=False):
        pts = [from_view(v) for v in poly_view]
        cells = g.cells_touching(pts) if touch else g.cells_in(pts, sc.size)
        out_ = 0
        if on:
            cells, out_ = inside_only(cells)
            g.selected |= cells
        else:
            g.selected -= cells
        g.save(px_um)
        st['msg'] = f"{'Selected' if on else 'Unselected'} {len(cells)} squares, total {len(g.selected)}" + \
            (f"; {out_} outside this scan's selection - not taken" if out_ else "")

    FAST_OK = keys is None or bool(os.environ.get('MC_FAST'))   # проверки — всегда в полном качестве (кроме своей)

    def fast_on():
        """Вид едет (тянут, колесо): кадры — вдвое грубее, пока едет; через 0.2 с покоя — в полном качестве."""
        if FAST_OK:
            st['fast_until'] = _t.monotonic() + 0.2

    def mouse(ev, x, y, flags, _=None):
        if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONUP,
                  cv2.EVENT_LBUTTONDBLCLK):
            st['acted'] = True                              # своё действие: что поменялось на диске — в историю
        if st.get('pick') and st['pick'].get('rects'):     # список съёмок: пока открыт — только он
            pk_ = st['pick']
            in_ = lambda r_: r_[0] <= x <= r_[2] and r_[1] <= y <= r_[3]
            hit_ = next((h_ for r_, h_ in pk_['rects'][1:] if in_(r_)), None)
            if ev == cv2.EVENT_MOUSEMOVE:
                pk_['hover'] = (x, y)
                st['car_dirty'] = True
            elif ev == cv2.EVENT_LBUTTONDOWN:
                r_ = pick_do(hit_ if hit_ is not None else ('panel' if in_(pk_['rects'][0][0]) else 'close'))
                if r_:
                    st['msg'] = r_
                st['dirty'] = st['car_dirty'] = True
            elif ev == cv2.EVENT_MOUSEWHEEL:
                pk_['scroll'] = max(0, pk_.get('scroll', 0) - int(np.sign(wheel_delta(flags))))
                st['car_dirty'] = True
            return
        if st.get('_list_left'):                            # список слева: x — как при списке справа
            x = x + (st['_fullW'] - SM.CAR_W) if x < SM.CAR_W else x - SM.CAR_W
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        dg_ = car.get('drag')
        if dg_ is not None and ev in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP) and not st['down']:
            if dg_['what'] == 'bar':                        # полоса прокрутки: тянуть
                if ev == cv2.EVENT_MOUSEMOVE:
                    ct_ = max(car.get('content', 0), st['Hv'] + 1)
                    car['scroll'] = int(max(0, min(ct_ - st['Hv'] + 20, dg_['s0'] + (y - dg_['y']) * ct_ / st['Hv'])))
                    st['car_dirty'] = st['dirty'] = True
                    return
                car['drag'] = None
                return
            if ev == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:   # карточку тянут: порядок слоёв
                if abs(y - dg_['y0']) > 8:
                    dg_.update(on=True, y=y)
                    st['car_dirty'] = st['dirty'] = True
                return
            if ev == cv2.EVENT_LBUTTONUP:
                car['drag'] = None
                if dg_.get('on'):
                    cards_ = [h_ for h_ in car.get('hits', []) if h_[4] == 'sel']
                    mp_ = car.get('map') or []
                    if cards_:
                        h_ = min(cards_, key=lambda h: 0 if h[1] <= y <= h[3] else min(abs(y - h[1]), abs(y - h[3])))
                        uk_ = lambda k2_: (mp_[k2_][2] if mp_[k2_][0] == 'sub' and len(mp_[k2_]) > 2 else mp_[k2_][1]) \
                            if 0 <= k2_ < len(mp_) else None
                        r_ = reorder_to(dg_['uid'], uk_(h_[5]), y > (h_[1] + h_[3]) / 2)
                        st['msg'] = (r_ or "Not moved") + "  (the list top is the top layer)"
                    st['car_dirty'] = st['dirty'] = True
                    return
        if x >= st['Wv'] and not st['down']:              # карусель образцов справа
            if ev == cv2.EVENT_MOUSEMOVE:                   # наведение: подсветка кнопок и карточек
                h_ = SM.hit(car['hits'], x - st['Wv'], y)
                if h_ != car.get('hov'):
                    car['hov'], car['hxy'], st['car_dirty'] = h_, (x - st['Wv'], y), True
                return
            if ev == cv2.EVENT_LBUTTONDOWN:                 # нажали: полоса прокрутки или карточку (тянуть)
                h_ = SM.hit(car['hits'], x - st['Wv'], y)
                if h_ and h_[0] == 'scroll':
                    car['drag'] = dict(what='bar', y=y, s0=car['scroll'])
                    return
                mp_ = car.get('map') or []
                if h_ and h_[0] == 'sel' and 0 <= h_[1] < len(mp_):
                    u_ = mp_[h_[1]][2] if mp_[h_[1]][0] == 'sub' and len(mp_[h_[1]]) > 2 else mp_[h_[1]][1]
                    if u_ and not str(u_).startswith('cell:'):
                        car['drag'] = dict(what='card', uid=u_, y0=y, y=y, on=False)
                return
            if ev == cv2.EVENT_LBUTTONUP:
                car['drag'] = None
                car_click(x, y, bool(flags & cv2.EVENT_FLAG_CTRLKEY))
            elif ev == cv2.EVENT_MOUSEWHEEL:
                car['scroll'] = int(max(0, min(car.get('content', 0) - st['Hv'] + 20,
                                               car['scroll'] - 60 * np.sign(wheel_delta(flags)))))
                st['dirty'] = True
            return
        if prof_mouse(ev, x, y, flags):                     # окошко профилей
            return
        if car.get('hov') is not None:                      # мышь ушла с карусели
            car['hov'], car['hxy'], st['car_dirty'] = None, None, True
        if ev == cv2.EVENT_MOUSEMOVE:
            st['hov'] = (x, y)
            if st.get('prof'):
                st['prof_dirty'] = True
        if ev == cv2.EVENT_MOUSEMOVE and st['mode'] == 'poly' and st['down'] is None:
            if not st['dirty']:                            # следующая сторона тянется за мышью: картинка и слои —
                st['_reuse'] = 'top'                       # готовые, заново только контур
            st['dirty'] = True
        if ev == cv2.EVENT_MOUSEMOVE and st['mode'] == 'ruler' and 0 < len(st['ruler']) < RK_N[st.get('rk', 'pp')]:
            st['mouse'] = (x + 0.5, y + 0.5)               # линейка тянется за мышью до следующей точки:
            if not st['dirty']:                            # картинку и слои не пересчитывать — только линейку
                st['_reuse'] = 'top'
            st['dirty'] = True
        if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            st['down'] = (x, y, st['cx'], st['cy'], ev, shift, False)
            st['ctrl'] = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
            if st['mode'] == 'align':
                st['al']['dq0'] = list(st['al']['dq'])
            st['mdrag'] = None                                 # левой по отметке креста — тащить её
            if ev == cv2.EVENT_LBUTTONDOWN and st['mode'] == 'cross' and g.cr.marks and not st.get('seeded'):
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
                    if not st['dirty']:                        # картинки те же — заново сетка и кресты
                        st['_reuse'] = 'grid'
                    st['dirty'] = True
            elif st['mode'] == 'align' and ev0 == cv2.EVENT_LBUTTONDOWN:   # L: картинки едут за мышью
                if abs(x - x0) + abs(y - y0) > 1 or moved:
                    st['down'] = (x0, y0, cx, cy, ev0, sh0, True)
                    al_set(np.asarray(st['al']['dq0'], float) - R().T @ (np.array([x - x0, y - y0], float) / st['s']))
                    st['dirty'] = True
            elif abs(x - x0) + abs(y - y0) > 4:
                st['down'] = (x0, y0, cx, cy, ev0, sh0, True)
                # сдвиг: правой / средней кнопкой — в любом режиме; левой — везде, кроме рамки (S) и кисти (B),
                # а там — с Ctrl
                pan = ev0 != cv2.EVENT_LBUTTONDOWN or st['mode'] in ('look', 'cross', 'label', 'ruler', 'poly') or st.get('ctrl')
                if pan:
                    c = np.array([cx, cy]) - R().T @ (np.array([x - x0, y - y0], float) / st['s'])
                    st['cx'], st['cy'] = c
                    fast_on()
                elif st['mode'] in ('select', 'paint'):        # рамка: всё, чего она касается
                    st['box'] = (x0, y0, x, y, sh0)
                    if not st['dirty']:                        # рамка — поверх готового
                        st['_reuse'] = 'top'
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
                    g.save(px_um)
                else:
                    st['msg'] = "Drag the mark to move it; right click removes it"
                st['dirty'] = True
            elif not moved and ev0 == cv2.EVENT_LBUTTONDOWN and st.get('ctrl') and st['mode'] == 'look':
                u_ = sel_at((x, y))                         # Ctrl+щелчок по выделению: отметить (несколько — разом)
                if u_:
                    mark_toggle(u_)
                else:
                    st['msg'] = "Ctrl+click a selection on the picture (or its card) to mark it"
            elif st['box'] is not None:
                bx0, by0, bx1, by1, shb = st['box']
                st['box'] = None
                if g.ready():                               # рамка берёт всё, чего касается
                    apply_poly([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)], not shb, touch=True)
            elif st['mode'] == 'poly' and ev0 == cv2.EVENT_LBUTTONDOWN and not moved:   # U: вершина многоугольника
                poly_click((x, y))
            elif st['mode'] == 'poly' and ev0 == cv2.EVENT_RBUTTONDOWN and not moved:   # правый: на шаг назад
                st['poly']['v'] = st['poly']['v'][:-1]
                st['msg'] = f"{len(st['poly']['v'])} points"
            elif st['mode'] == 'paint' and ev0 == cv2.EVENT_LBUTTONDOWN and not moved:
                if g.ready():                               # щелчок кистью: один квадрат (Shift — снять)
                    c_ = g.cell_at(from_view((x, y)))
                    if not sh0 and not inside_only({c_})[0]:
                        st['msg'] = "This square lies outside the scanned selection: not taken"
                        st['dirty'] = True
                        return
                    (g.selected.discard if sh0 else g.selected.add)(c_)
                    g.save(px_um)
                st['msg'] = f"Selected {len(g.selected)} squares"
            elif not moved and ev0 == cv2.EVENT_LBUTTONDOWN:
                if st['mode'] == 'stitch' and seam_at((x, y)) is not None:   # J: точка шва — выключить / включить
                    seam_toggle(seam_at((x, y)))
                elif st['mode'] == 'stitch':                  # J: ряд (Shift — столбец) кадра под мышью
                    sel_, ids_ = stitch_frames_at(from_view((x + 0.5, y + 0.5)), column=sh0)
                    if ids_:
                        st['edit'].update(sel=sel_, ids=ids_)
                        st['msg'] = (f"{sel_[0].capitalize()} {sel_[1]} ({len(ids_)} frames) chosen: arrows move it "
                                     "by 1 px, Shift+arrows by 10 px. Shift+click: a column")
                elif st.get('prof') and st['mode'] == 'look':
                    prof_click(x, y)
                elif st['mode'] == 'cross':
                    add_cross((x, y), snap=sh0)
                elif st['mode'] == 'select' and g.ready():
                    c = g.cell_at(from_view((x, y)))
                    if c not in g.selected and not inside_only({c})[0]:
                        st['msg'] = "This square lies outside the scanned selection: not taken"
                        st['dirty'] = True
                        return
                    g.selected.symmetric_difference_update({c})
                    g.save(px_um)
                    st['msg'] = f"Square {c} {'selected' if c in g.selected else 'unselected'}, total {len(g.selected)}"
                elif st['mode'] == 'ruler':                 # линейка: точка ровно там, где щёлкнули
                    q_ = from_view((x + 0.5, y + 0.5))
                    n_ = RK_N[st.get('rk', 'pp')]
                    st['ruler'] = [q_] if len(st['ruler']) >= n_ else st['ruler'] + [q_]
                    left_ = n_ - len(st['ruler'])
                    st['msg'] = ruler_text() if not left_ else \
                        (f"{RK_NAME[st.get('rk', 'pp')].capitalize()}: " +
                         {('ll', 3): "the first line - its second point", ('ll', 2): "the second line - its first point",
                          ('ll', 1): "the second line - its second point", ('pl', 2): "the line - its second point",
                          ('pl', 1): "the point"}.get((st.get('rk', 'pp'), left_), "the next point"))
                elif st['mode'] == 'label' and g.ready():
                    c = g.cell_at(from_view((x, y)))
                    X, Y = g.labels.get(c, g.name(*c))
                    st['lab'] = [c, ""]
                    st['msg'] = f"Square {c}: type its number X Y (now {X} {Y}), Enter = keep, Esc = cancel"
            elif not moved and ev0 == cv2.EVENT_RBUTTONDOWN:
                if st['mode'] == 'cross' and g.cr.marks and not st.get('seeded'):
                    dists = [np.hypot(*(to_view(q) - (x, y))) for q in g.cr.marks]
                    k = int(np.argmin(dists))
                    if dists[k] < 25:
                        unseed()
                        g.cr.remove(k)
                        g.save(px_um)
                        st['msg'] = "Cross mark removed"
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
        elif ev in (cv2.EVENT_MOUSEWHEEL, getattr(cv2, 'EVENT_MOUSEHWHEEL', -99)):   # колесо / тачпад
            gs_ = wheel_gesture(ev, flags)
            if gs_[0] == 'zoom':
                zoom(gs_[1], x, y)
            else:                                           # два пальца: картинка едет за пальцами
                c = np.array([st['cx'], st['cy']]) - R().T @ (np.array(gs_[1:], float) / st['s'])
                st['cx'], st['cy'] = c
            fast_on()
            st['dirty'] = True

    prof_cache = {}
    PL_COL = ((255, 230, 0), (255, 80, 255))                # линия A — голубая, B — сиреневая (BGR)
    EDGE_COL = (0, 220, 255)

    def pl_seg_dirs(sg):
        P0, P1 = np.asarray(sg[0], float), np.asarray(sg[1], float)
        L_ = max(float(np.hypot(*(P1 - P0))), 1e-6)
        u = (P1 - P0) / L_
        return P0, P1, L_, u, np.array([-u[1], u[0]])

    def pl_new(Mc):
        """Две линии профиля у места Mc (мозаика). Есть крест — каждая линия поперёк своего плеча, на 0.6 плеча
        от центра, линии перекрещиваются над четвертью креста; нет креста — «+» через Mc по наклону сетки."""
        key_ = (int(round(Mc[0] * 2)), int(round(Mc[1] * 2)))
        if key_ in prof_cache:
            return prof_cache[key_]
        from .mask import meas_gray
        r = int(min(150, max(8, 15.0 / px_um)))
        reg = sc.render(Mc[0], Mc[1], 1.0, 4 * r + 1, 4 * r + 1)
        info = {}
        try:
            d, resp = snap_cross(meas_gray(reg), r, arm=g.arm, info=info, angle=g.angle_deg())
        except Exception:
            d, resp = np.zeros(2), 0.0
        L, t = info.get('arm'), info.get('thick')
        if resp > 0.2 and L and t and np.any(d) and L * px_um >= 2.0:      # крест, а не пылинка
            ang = np.radians(info['angle'] if info.get('edges') else (g.angle_deg() or 0.0))
            c = Mc + d
            eh, ev = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
            H = 0.85 * L
            segs = [[c + 0.6 * L * ev - H * eh, c + 0.6 * L * ev + H * eh],     # A — через вертикальное плечо
                    [c + 0.6 * L * eh - H * ev, c + 0.6 * L * eh + H * ev]]     # B — через горизонтальное
            res = dict(cross=True, c=c, L=L, t=t, ang=ang, segs=segs, w=max(3.0, 0.5 * L), edges_by=info.get('edges'))
        else:
            ang = np.radians(g.angle_deg() or 0.0)
            eh, ev = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
            H = float(min(r, 60))
            res = dict(cross=False, c=np.asarray(Mc, float), L=H, t=4, ang=ang,
                       segs=[[Mc - H * eh, Mc + H * eh], [Mc - H * ev, Mc + H * ev]], w=3.0)
        res['user'] = [None, None]
        res['reg'], res['org'] = reg, np.asarray(Mc, float) - 2 * r
        pl_fill(res)
        if len(prof_cache) > 30:
            prof_cache.clear()
        prof_cache[key_] = res
        return res

    def pl_cross_point(res):
        """Пересечение двух линий (мозаика) и где оно на каждой (расстояние от середины), или None."""
        P0, _, LA, u, _ = pl_seg_dirs(res['segs'][0])
        Q0, _, LB, u2, _ = pl_seg_dirs(res['segs'][1])
        M_ = np.c_[u, -u2]
        if abs(np.linalg.det(M_)) < 0.05:
            return None
        sA, sB = np.linalg.solve(M_, Q0 - P0)
        return P0 + sA * u, (sA - LA / 2, sB - LB / 2)

    def pl_fill(res):
        """Профили R, G, B вдоль обеих линий, усреднённые поперёк по полосе ширины w (как «thickness»
        в Gwyddion, «line width» в ImageJ). Картинка дорисовывается, если линии ушли за неё."""
        from .mask import edges_any
        pts = np.vstack([np.asarray(p_, float) for sg in res['segs'] for p_ in sg])
        lo, hi = pts.min(0) - res['w'] - 6, pts.max(0) + res['w'] + 6
        org, reg = res['org'], res['reg']
        if np.any(lo < org) or np.any(hi > org + (reg.shape[1] - 1, reg.shape[0] - 1)):
            cen = (lo + hi) / 2
            a_ = np.minimum(np.ceil((hi - lo) / 2 * 1.3 + 20), 900).astype(int)
            res['reg'] = reg = sc.render(cen[0], cen[1], 1.0, 2 * a_[0] + 1, 2 * a_[1] + 1)
            res['org'] = org = cen - a_
        res['prof'] = []
        img_ = reg.astype(np.float32)
        for k, sg in enumerate(res['segs']):
            P0, P1, L_, u, v = pl_seg_dirs(sg)
            s_ = np.arange(0.0, L_ + 1e-6, 1.0)
            o_ = np.linspace(-res['w'] / 2, res['w'] / 2, max(1, int(round(res['w'])) + 1))
            X = P0[0] + s_[None, :] * u[0] + o_[:, None] * v[0] - org[0]
            Y = P0[1] + s_[None, :] * u[1] + o_[:, None] * v[1] - org[1]
            prof = cv2.remap(img_, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE).mean(0)
            ed = edges_any(prof[:, 1])
            x_ = s_ - L_ / 2                                 # ноль — середина линии
            res['prof'].append(dict(x=x_, prof=prof, edges=(ed[0] - L_ / 2, ed[1] - L_ / 2) if ed else None,
                                    P0=P0, u=u, v=v, L=L_))

    def prof_edges(res, k):
        u_ = res['user'][k]
        return u_ if u_ is not None else res['prof'][k]['edges']

    def pl_edge_pts(res, k):
        """Края на линии k — точки мозаики (середина линии + смещение вдоль)."""
        d_ = res['prof'][k]
        ed = prof_edges(res, k)
        if not ed:
            return []
        mid_ = d_['P0'] + d_['L'] / 2 * d_['u']
        return [mid_ + e_ * d_['u'] for e_ in ed]

    def pl_draw(img, res, f_map, scale, thick=1, handles=False):
        """Линии, полосы усреднения, засечки краёв (поперёк полосы) — на карте или в лупе.
        f_map — мозаика -> картинка, scale — пикселей картинки на пиксель мозаики."""
        ip = lambda q: (int(round(q[0])), int(round(q[1])))
        for k, d_ in enumerate(res['prof']):
            P0, P1 = res['segs'][k]
            col = PL_COL[k]
            w2 = res['w'] / 2 * d_['v']
            if res['w'] * scale >= 4:                        # полоса усреднения — тонкие края
                for sg_ in (+1, -1):
                    cv2.line(img, ip(f_map(P0 + sg_ * w2)), ip(f_map(P1 + sg_ * w2)), tuple(c_ // 2 for c_ in col), 1, cv2.LINE_8)
            cv2.line(img, ip(f_map(P0)), ip(f_map(P1)), col, thick, cv2.LINE_8)
            hw = max(res['w'] / 2, 4.0 / max(scale, 1e-6))
            for q in pl_edge_pts(res, k):                    # граница — засечка поперёк линии
                cv2.line(img, ip(f_map(q - hw * d_['v'])), ip(f_map(q + hw * d_['v'])), EDGE_COL, thick, cv2.LINE_8)
            if handles:
                for q in (P0, P1):
                    cv2.drawMarker(img, ip(f_map(q)), col, cv2.MARKER_SQUARE, 9, 1)
                cv2.drawMarker(img, ip(f_map((np.asarray(P0) + P1) / 2)), col, cv2.MARKER_TILTED_CROSS, 7, 1)
        cp = pl_cross_point(res)
        if handles and cp is not None:
            cv2.drawMarker(img, ip(f_map(cp[0])), (255, 255, 255), cv2.MARKER_DIAMOND, 11, 1)

    def prof_ylim(d_):
        P = np.asarray(d_['prof'], float)
        lo, hi = float(P.min()), float(P.max())
        span = max(hi - lo, 1.0)
        return lo - 0.08 * span, hi + 0.08 * span

    def prof_plot(img, x0, y0, w, h, d_, title, edges, col, mark=None, ylim=None):
        """Один график: B, G, R вдоль линии (мкм от её середины); края — жёлтые линии (тянутся мышью).
        ylim — общий масштаб по Y (колесо над графиком), None — свой по данным."""
        UI.put(img, title, x0, y0 - 6, 0.45, col)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (90, 90, 90), 1)
        x, P = d_['x'], np.asarray(d_['prof'], float)
        lo, hi = ylim if ylim is not None else prof_ylim(d_)
        fx = lambda v_: x0 + 4 + (v_ - x[0]) / max(x[-1] - x[0], 1) * (w - 8)
        Y = lambda v_: np.clip(y0 + h - 4 - (v_ - lo) / (hi - lo) * (h - 8), y0 + 1, y0 + h - 1)
        if mark is not None and x[0] <= mark <= x[-1]:      # где перекрещиваются линии
            xm = int(round(fx(mark)))
            for yy_ in range(y0 + 2, y0 + h - 2, 6):
                cv2.line(img, (xm, yy_), (xm, yy_ + 2), (170, 170, 170), 1)
        if edges:
            for e_ in edges:
                xe = int(round(fx(e_)))
                if x0 <= xe <= x0 + w:
                    cv2.line(img, (xe, y0 + 1), (xe, y0 + h - 1), EDGE_COL, 1, cv2.LINE_8)
                    cv2.drawMarker(img, (xe, y0 + h - 2), EDGE_COL, cv2.MARKER_TRIANGLE_UP, 7, 1)
        for ch_, c_ in ((0, (255, 140, 60)), (1, (80, 220, 80)), (2, (70, 70, 255))):
            pts = np.c_[fx(x), Y(P[:, ch_])].round().astype(np.int32)
            cv2.polylines(img, [pts], False, c_, 1, cv2.LINE_AA)
        UI.put(img, f"{hi:.0f}", x0 + w + 4, y0 + 12, 0.38, UI.DIM)
        UI.put(img, f"{lo:.0f}", x0 + w + 4, y0 + h - 2, 0.38, UI.DIM)
        UI.put(img, f"{x[0] * px_um:+.1f}", x0, y0 + h + 14, 0.38, UI.DIM)
        UI.put(img, f"{x[-1] * px_um:+.1f} um", x0 + w - 52, y0 + h + 14, 0.38, UI.DIM)
        if edges:
            a_, b_ = sorted(edges)
            G_ = P[:, 1]
            n_ = len(G_)
            k_ = np.clip(np.round(np.array([a_, b_]) - x[0]).astype(int), 0, n_ - 1)
            out_ = np.r_[G_[:max(1, k_[0] - 2)], G_[min(n_ - 1, k_[1] + 3):]]
            bg = float(np.median(out_)) if len(out_) else float(np.median(G_))
            arm_ = float(np.median(G_[k_[0]:k_[1] + 1])) if k_[1] > k_[0] else bg
            return (b_ - a_), (arm_ - bg) / max(bg, 1.0)
        return None

    def prof_res():
        """Что показывать: линии по 4 точкам (закреплены)."""
        if st.get('prof_pin') is not None:
            return st['prof_pin']
        hx, hy = st['hov']
        x0_, y0_, x1_, y1_ = st.get('prof_rect') or (0, 0, -1, -1)
        if x0_ <= hx <= x1_ and y0_ <= hy <= y1_ and st.get('prof_last') is not None:
            return st['prof_last']
        st['prof_last'] = pl_new(from_view((hx + 0.5, hy + 0.5)))
        return st['prof_last']

    def with_prof(img):
        """Окошко профилей (P): две линии, которые можно двигать и наклонять как угодно (концы — на карте),
        профили R, G, B вдоль них с усреднением по полосе; края — на графике, на карте и в лупе окошка."""
        if not st.get('prof') or st.get('hov') is None:
            return img
        Wv, Hv = st['Wv'], st['Hv']
        hx, hy = st['hov']
        if not (0 <= hx < Wv and 0 <= hy < Hv) and st.get('prof_pin') is None:
            return img
        img = img.copy()
        if st.get('prof_pin') is None:                     # сначала 4 точки — потом графики
            pts_ = list(st.get('prof_pts') or [])
            if st.get('mouse') is not None and len(pts_) < 4:
                pts_v = [to_view(q) for q in pts_] + [tuple(st['mouse'])]
            else:
                pts_v = [to_view(q) for q in pts_]
            for i_, q_ in enumerate(pts_v):
                UI.plus(img, q_[0], q_[1], PL_COL[0 if i_ < 2 else 1])
            for k_, (i0_, i1_) in enumerate(((0, 1), (2, 3))):
                if len(pts_v) > i1_:
                    cv2.line(img, tuple(np.int32(np.round(pts_v[i0_]))), tuple(np.int32(np.round(pts_v[i1_]))),
                             PL_COL[k_], 1, cv2.LINE_AA)
            t_ = PROF_ASK[min(3, len(st.get('prof_pts') or []))] + f"  ({len(st.get('prof_pts') or [])} of 4)"
            UI.toast(img, t_, Wv / 2, 40, 0.5, 'info', Wv - 40)
            return img
        try:
            res = prof_res()
        except Exception as e_:
            UI.toast(img, f"Profiles: {e_}", Wv / 2, 40, 0.5, 'err', Wv - 40)
            return img
        pl_draw(img, res, to_view, st['s'], 1, handles=True)
        if res['cross']:
            q = to_view(res['c'])
            cv2.drawMarker(img, (int(round(q[0])), int(round(q[1]))), EDGE_COL, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        pw, ph = 400, 520
        inside = st.get('prof_rect') is not None and (st['prof_rect'][0] <= hx <= st['prof_rect'][2] and
                                                      st['prof_rect'][1] <= hy <= st['prof_rect'][3])
        if not inside and st.get('pl_drag') is None:
            st['prof_x0'] = Wv - pw - 12 if hx < Wv - pw - 40 else 12   # окошко — с другой стороны от мыши
        x0, y0 = st.get('prof_x0', Wv - pw - 12), 44
        st['prof_rect'] = (x0, y0, x0 + pw, y0 + ph)
        UI.panel(img, x0, y0, x0 + pw, y0 + ph, 0.92, UI.LINE)
        ttl = "Profiles" + ("  (pinned)" if st.get('prof_pin') is not None else "  (follows the mouse)")
        UI.put(img, ttl, x0 + 12, y0 + 22, 0.5, UI.ACCENT, 1)
        # лупа: то же место крупно, с линиями, полосами и краями
        lz = 150
        pts = np.vstack([np.asarray(p_, float) for sg in res['segs'] for p_ in sg])
        cen = (pts.min(0) + pts.max(0)) / 2
        half = max(float((pts.max(0) - pts.min(0)).max()) / 2 + res['w'] / 2 + 6, 10.0)
        a0 = np.round(cen - half - res['org']).astype(int)
        n_ = int(np.ceil(2 * half))
        reg = res['reg']
        crop = np.full((n_, n_, 3), 20, np.uint8)
        ys0, xs0 = max(0, a0[1]), max(0, a0[0])
        ys1, xs1 = min(reg.shape[0], a0[1] + n_), min(reg.shape[1], a0[0] + n_)
        if ys1 > ys0 and xs1 > xs0:
            crop[ys0 - a0[1]:ys1 - a0[1], xs0 - a0[0]:xs1 - a0[0]] = reg[ys0:ys1, xs0:xs1]
        kz = lz / n_
        lup = cv2.resize(crop, (lz, lz), interpolation=cv2.INTER_LINEAR if kz < 3 else cv2.INTER_NEAREST)
        f_l = lambda q: (np.asarray(q, float) - res['org'] - a0 + 0.5) * kz - 0.5
        pl_draw(lup, res, f_l, kz, 1)
        lx, ly = x0 + 12, y0 + 34
        img[ly:ly + lz, lx:lx + lz] = lup
        cv2.rectangle(img, (lx - 1, ly - 1), (lx + lz, ly + lz), (90, 90, 90), 1)
        UI.put(img, f"{2 * half * px_um:.0f} um", lx + 4, ly + lz - 6, 0.38, UI.TEXT)
        # цифры справа от лупы
        cp = pl_cross_point(res)
        info_, plots, rows = [], {}, []
        for k, nm in enumerate(("A", "B")):
            d_ = res['prof'][k]
            r_ = prof_plot(img, x0 + 12, y0 + 222 + k * 124, pw - 60, 80, d_,
                           f"{nm}: along the {'cyan' if k == 0 else 'pink'} line", prof_edges(res, k), PL_COL[k],
                           mark=cp[1][k] if cp is not None else None, ylim=st.get('prof_y'))
            plots[k] = (x0 + 12, y0 + 222 + k * 124, pw - 60, 80, float(d_['x'][0]), float(d_['x'][-1]))
            angd = (np.degrees(np.arctan2(-d_['u'][1], d_['u'][0])) + 180) % 180
            if r_:
                info_.append((f"{nm}: width {r_[0] * px_um:.2f} um" + (" *" if res['user'][k] is not None else ""), PL_COL[k]))
                info_.append((f"   {r_[0]:.1f} px, contrast {100 * r_[1]:+.0f} %", UI.TEXT))
            else:
                info_.append((f"{nm}: no edges", PL_COL[k]))
            info_.append((f"   angle {angd:.1f} deg, {d_['L'] * px_um:.1f} um", UI.DIM))
        st['prof_plots'] = plots
        yl = st.get('prof_y')                              # масштаб по Y: общий для обоих графиков, щелчок — авто
        ytxt = "Y: auto" if yl is None else f"Y: {yl[0]:.0f}-{yl[1]:.0f}  [auto]"
        tw = UI.size(ytxt, 0.42)[0]
        UI.put(img, ytxt, x0 + pw - 14 - tw, y0 + 22, 0.42, UI.TEXT if yl is None else UI.ACCENT)
        st['prof_yrect'] = (x0 + pw - 18 - tw, y0 + 6, x0 + pw - 8, y0 + 28)
        info_.append((f"Band (averaging) {res['w'] * px_um:.2f} um", UI.TEXT))
        if cp is not None:
            dang = np.degrees(np.arccos(min(1.0, abs(float(res['prof'][0]['u'] @ res['prof'][1]['u'])))))
            info_.append((f"Lines cross at {dang:.1f} deg", UI.DIM))
        yy = ly + 12
        for t_, c_ in info_:
            UI.put(img, t_, lx + lz + 12, yy, 0.42, c_)
            yy += 17
        if max(float(np.max(d_['prof'])) for d_ in res['prof']) >= 250:
            rows.append(("Overexposed (255): lower the exposure for exact edges", UI.ERR))
        rows += [("Map: drag the squares (ends) - tilt and length, Shift - 15 deg", UI.DIM),
                 ("x - move a line, diamond - both. Click elsewhere: pin / unpin", UI.DIM),
                 ("Here: drag yellow edges; wheel: on a plot - Y scale, else - band", UI.DIM),
                 ("Right click on a plot: auto edges. Click 'Y': auto scale", UI.DIM)]
        yy = y0 + ph - 10 - 16 * (len(rows) - 1)
        for t_, c_ in rows:
            UI.put(img, t_, x0 + 12, yy, 0.4, c_)
            yy += 16
        return img

    def pl_handle(res, x, y):
        """Ручка линий под мышью (окно просмотра): конец ('end', k, i), середина ('mid', k), пересечение ('x',)."""
        best, dmin = None, 9.0
        for k, sg in enumerate(res['segs']):
            for i, q in enumerate(sg):
                dd = float(np.hypot(*(to_view(q) - (x, y))))
                if dd < dmin:
                    best, dmin = ('end', k, i), dd
        if best:
            return best
        cp = pl_cross_point(res)
        if cp is not None and np.hypot(*(to_view(cp[0]) - (x, y))) < 9:
            return ('x',)
        for k, sg in enumerate(res['segs']):
            if np.hypot(*(to_view((np.asarray(sg[0]) + sg[1]) / 2) - (x, y))) < 9:
                return ('mid', k)
        return None

    def pl_save():
        """Линии и края — в файл образца (profile_lines.json, мкм столика): их рисует и карта ручного режима."""
        try:
            from . import project as P_
            from .snap import mosaic_to_stage
            smp = P_.sample_of(folder)
            pth = os.path.join(smp, 'profile_lines.json') if smp else None
            if not pth:
                return
            res = st.get('prof_pin')
            if not st.get('prof') or res is None:
                if os.path.exists(pth):
                    os.remove(pth)
                return
            out = []
            for k in range(2):
                S_ = mosaic_to_stage(folder, np.vstack([np.asarray(q, float) for q in res['segs'][k]]))
                E_ = pl_edge_pts(res, k)
                out.append(dict(ends=S_.round(3).tolist(), w_um=round(res['w'] * px_um, 3),
                                edges=mosaic_to_stage(folder, np.vstack(E_)).round(3).tolist() if E_ else []))
            with open(pth, 'w', encoding='utf-8') as fh:
                json.dump(dict(scan=os.path.abspath(folder), time=_t.strftime('%Y-%m-%d %H:%M'), lines=out), fh)
        except Exception as e_:
            L.event('VIEW-PROF', f"profile lines not saved: {e_}", 30)

    def prof_mouse(ev, x, y, flags):
        """Мышь для профилей: в окошке — края (тянуть), колесо — полоса, правый щелчок — края заново;
        на карте — ручки линий (концы, середины, пересечение). -> True, если событие съедено."""
        if not st.get('prof'):
            return False
        drag = st.get('prof_drag')
        pdrag = st.get('pl_drag')
        rect = st.get('prof_rect')
        in_panel = rect is not None and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]
        res = st.get('prof_pin') or st.get('prof_last')
        if pdrag is not None:                                # тянем ручку линии на карте
            if ev == cv2.EVENT_MOUSEMOVE:
                M_ = from_view((x + 0.5, y + 0.5))
                d0 = M_ - pdrag['m0']
                segs0 = pdrag['segs0']
                if pdrag['h'][0] == 'end':
                    _, k, i = pdrag['h']
                    other = np.asarray(segs0[k][1 - i], float)
                    q = np.asarray(segs0[k][i], float) + d0
                    if flags & cv2.EVENT_FLAG_SHIFTKEY:      # Shift: наклон шагами 15° от наклона креста
                        vv = q - other
                        a_ = np.arctan2(vv[1], vv[0]) - res['ang']
                        a_ = np.round(a_ / np.radians(15)) * np.radians(15) + res['ang']
                        q = other + np.hypot(*vv) * np.array([np.cos(a_), np.sin(a_)])
                    if np.hypot(*(q - other)) >= 4:
                        res['segs'][k][i] = q
                elif pdrag['h'][0] == 'mid':
                    k = pdrag['h'][1]
                    res['segs'][k] = [np.asarray(q, float) + d0 for q in segs0[k]]
                else:
                    res['segs'] = [[np.asarray(q, float) + d0 for q in sg] for sg in segs0]
                pl_fill(res)
                st['prof_dirty'] = True
            elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                st['pl_drag'] = None
                pl_save()
            return True
        if drag is not None or in_panel:
            if res is None:
                return True

            def off_at(k):
                px0, py0, w_, h_, xa, xb = st['prof_plots'][k]
                return xa + (x - px0 - 4) / max(w_ - 8, 1) * (xb - xa)

            def plot_at():
                for k, (px0, py0, w_, h_, xa, xb) in st.get('prof_plots', {}).items():
                    if px0 - 6 <= x <= px0 + w_ + 6 and py0 - 4 <= y <= py0 + h_ + 4:
                        return k
                return None
            yr = st.get('prof_yrect')
            if ev == cv2.EVENT_LBUTTONDOWN and yr and yr[0] <= x <= yr[2] and yr[1] <= y <= yr[3]:
                st['prof_y'] = None                         # масштаб по Y — снова по данным
                st['prof_dirty'] = True
                return True
            if ev == cv2.EVENT_LBUTTONDOWN:
                k = plot_at()
                if k is not None:
                    st['prof_pin'] = res                    # тянут край — линии закреплены
                    ed = list(prof_edges(res, k) or (off_at(k) - 2, off_at(k) + 2))
                    i_ = int(np.argmin([abs(off_at(k) - e_) for e_ in ed]))
                    ed[i_] = off_at(k)
                    res['user'][k] = tuple(ed)
                    st['prof_drag'] = (k, i_)
                    st['prof_dirty'] = True
            elif ev == cv2.EVENT_MOUSEMOVE and drag is not None:
                k, i_ = drag
                ed = list(res['user'][k])
                ed[i_] = off_at(k)
                res['user'][k] = tuple(ed)
                st['prof_dirty'] = True
            elif ev == cv2.EVENT_LBUTTONUP:
                if drag is not None:
                    st['prof_drag'] = None
                    pl_save()
            elif ev == cv2.EVENT_RBUTTONDOWN:
                k = plot_at()
                for k_ in ([k] if k is not None else [0, 1]):
                    res['user'][k_] = None
                st['prof_dirty'] = True
                pl_save()
            elif ev == cv2.EVENT_MOUSEWHEEL and plot_at() is not None:   # масштаб по Y у мыши, общий для обоих
                k = plot_at()
                px0, py0, w_, h_, xa, xb = st['prof_plots'][k]
                if st.get('prof_y') is None:
                    l0, h0 = zip(*[prof_ylim(d_) for d_ in res['prof']])
                    st['prof_y'] = (min(l0), max(h0))
                lo, hi = st['prof_y']
                yv = lo + (py0 + h_ - 4 - y) / max(h_ - 8, 1) * (hi - lo)       # значение под мышью остаётся на месте
                f_ = 0.8 if wheel_delta(flags) > 0 else 1.25
                lo, hi = yv - (yv - lo) * f_, yv + (hi - yv) * f_
                if hi - lo >= 2 and lo > -60 and hi < 320:
                    st['prof_y'] = (lo, hi)
                st['prof_dirty'] = True
            elif ev == cv2.EVENT_MOUSEWHEEL:                # полоса усреднения: шире / уже
                res['w'] = float(np.clip(res['w'] * (1.25 if wheel_delta(flags) > 0 else 0.8), 1.0, 4 * res['L']))
                pl_fill(res)
                st['prof_dirty'] = True
                pl_save()
            return True
        if ev == cv2.EVENT_LBUTTONDOWN and res is not None:  # ручка линии на карте
            h_ = pl_handle(res, x, y)
            if h_ is not None:
                st['prof_pin'] = res
                st['pl_drag'] = dict(h=h_, m0=from_view((x + 0.5, y + 0.5)),
                                     segs0=[[np.asarray(q, float).copy() for q in sg] for sg in res['segs']])
                return True
        return False

    PROF_ASK = ("Profiles: click the first end of line A", "Profiles: click the second end of line A",
                "Profiles: click the first end of line B", "Profiles: click the second end of line B")

    def pl_from_pts(pts):
        """Две линии профиля по 4 точкам (мозаика): A — точки 1-2, B — 3-4."""
        P = np.vstack([np.asarray(q, float) for q in pts])
        c = (P.min(0) + P.max(0)) / 2
        r = int(max(8, float((P.max(0) - P.min(0)).max()) / 2 + 20))
        reg = sc.render(c[0], c[1], 1.0, 2 * r + 1, 2 * r + 1)
        d0 = P[1] - P[0]
        res = dict(cross=False, c=c, L=float(np.hypot(*d0)) / 2, t=4, ang=float(np.arctan2(d0[1], d0[0])),
                   segs=[[P[0].copy(), P[1].copy()], [P[2].copy(), P[3].copy()]], w=3.0, user=[None, None],
                   reg=reg, org=c - r)
        pl_fill(res)
        return res

    def prof_click(x, y):
        """Щелчок по карте (не по ручке): точки линий профиля — сначала 4 (концы A, концы B), потом графики;
        линии уже есть — щелчок мимо ручек начинает новые."""
        q_ = from_view((x + 0.5, y + 0.5))
        if st.get('prof_pin') is not None:
            st['prof_pin'], st['prof_pts'] = None, [q_]
        else:
            st.setdefault('prof_pts', []).append(q_)
        pts_ = st['prof_pts']
        if len(pts_) >= 4:
            st['prof_pin'], st['prof_pts'] = pl_from_pts(pts_[:4]), []
            st['msg'] = ("Profiles built. Drag the line ends to tilt them, the yellow edges in the window; the same "
                         "lines are on the map of the manual mode. A click elsewhere: 4 new points")
        else:
            st['msg'] = PROF_ASK[len(pts_)] + f"  ({len(pts_)} of 4)"
        pl_save()
        st['dirty'] = True

    def inner_rect():
        """Прямоугольник, целиком покрытый кадрами (без ступенек по краям) — мозаика, px."""
        P_, W_, H_ = sc.P, sc.W, sc.H
        try:
            rows_ = [int(r_['row']) for r_ in sc.st.index]
            cols_ = [int(r_['col']) for r_ in sc.st.index]
        except Exception:
            return None
        R_, C_ = np.array(rows_), np.array(cols_)
        xl = max(P_[C_ == C_.min(), 0].max(), 0.0)
        xr = min(P_[C_ == C_.max(), 0].min() + W_, sc.size[0])
        yt = max(P_[R_ == R_.min(), 1].max(), 0.0)
        yb = min(P_[R_ == R_.max(), 1].min() + H_, sc.size[1])
        return (xl, yt, xr, yb) if xr - xl > W_ / 4 and yb - yt > H_ / 4 else None

    def crop_edges(img):
        """K: всё вне прямоугольника без ступенек — пустым (как вне мозаики): края ровные."""
        if not st.get('crop', True) or st['mode'] == 'stitch':
            return img
        r_ = st.get('inner') or inner_rect()
        st['inner'] = r_
        if r_ is None:
            return img
        q_ = np.array([to_view(p_) for p_ in ((r_[0], r_[1]), (r_[2], r_[1]), (r_[2], r_[3]), (r_[0], r_[3]))])
        mk_ = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mk_, [np.int32(np.round(q_))], 255)
        img[mk_ == 0] = 20
        return img

    def seams_load():
        """J: швы последней сшивки — на картинке, цвет по невязке; щелчок по точке шва — выключить / включить."""
        from .stitch import seams as _seams, seams_edit, resolve
        ed_ = st['edit']
        z_ = _seams(folder)
        if z_ is None or len(z_['I']) == 0 or int(z_['I'].max(initial=0)) >= len(sc.P) \
                or int(z_['J'].max(initial=0)) >= len(sc.P):
            ed_['z'] = None
            return
        ovr_ = seams_edit(folder)
        pz_, res_, w_ = resolve(z_, ovr_)
        sh_ = np.median(sc.P - pz_, 0)                          # от координат сшивки — к картинке
        ed_.update(z=z_, ovr=ovr_, res=res_, w=w_, sh=sh_, man=sc.P - (pz_ + sh_))   # man: руками и по крестам

    def seam_pt(k_):
        """Точка шва k на мозаике (px): середина; у диагонального шва — на 35 % от первого кадра, иначе у двух
        диагоналей крест-накрест (змейка) точки совпали бы и щелчок попадал бы не в тот шов."""
        z_ = st['edit']['z']
        a_, b_ = sc.P[z_['I'][k_]], sc.P[z_['J'][k_]]
        d_ = np.abs(b_ - a_)
        t_ = 0.35 if d_[0] > 0.25 * sc.W and d_[1] > 0.25 * sc.H else 0.5
        return a_ + t_ * (b_ - a_) + (sc.W / 2, sc.H / 2)

    def seam_at(xy):
        """Шов, чья точка под мышью (в пикселях окна), или None."""
        ed_ = st.get('edit') or {}
        z_ = ed_.get('z')
        if z_ is None:
            return None
        best_ = None
        for k_ in range(len(z_['I'])):
            m_ = to_view(seam_pt(k_))
            d_ = np.hypot(m_[0] - xy[0], m_[1] - xy[1])
            if d_ <= 7 and (best_ is None or d_ < best_[0]):
                best_ = (d_, k_)
        return None if best_ is None else best_[1]

    def seam_toggle(k_):
        """Шов k: выключить (включён) / включить (выключен) — и сразу положения кадров заново по всем швам."""
        from .stitch import resolve
        ed_ = st['edit']
        z_ = ed_['z']
        key_ = (int(z_['I'][k_]), int(z_['J'][k_]))
        on_ = ed_['w'][k_] > 0
        ed_['ovr'][key_] = not on_
        pz_, res_, w_ = resolve(z_, ed_['ovr'])
        sc.P[:] = pz_ + ed_['sh'] + ed_['man']
        ed_.update(res=res_, w=w_, moved=True, seams_changed=True)
        sc.over = None
        a_, b_ = key_
        r_ = res_[w_ > 0]
        st['msg'] = (f"Seam {a_ + 1} + {b_ + 1} {'off' if on_ else 'on'}: frames placed again by "
                     f"{int((w_ > 0).sum())} seams, error {np.median(r_) if len(r_) else 0:.2f} px median, "
                     f"{r_.max() if len(r_) else 0:.2f} max. J: save, Backspace: undo all")

    def stitch_frames_at(Mc, column=False):
        """Кадры ряда (или столбца) того кадра, что под точкой Mc."""
        P_, W_, H_ = sc.P, sc.W, sc.H
        hit_ = [i for i in range(len(P_)) if P_[i, 0] <= Mc[0] < P_[i, 0] + W_ and P_[i, 1] <= Mc[1] < P_[i, 1] + H_]
        if not hit_:
            return None, []
        i0 = hit_[-1]
        key_ = 'col' if column else 'row'
        v_ = sc.st.index[i0][key_]
        return (key_, v_), [i for i, r_ in enumerate(sc.st.index) if r_[key_] == v_]

    def stitch_move(dx, dy):
        ed_ = st.get('edit')
        if not ed_ or not ed_.get('ids'):
            st['msg'] = "Click a frame first: its row moves (Shift+click: its column)"
            return
        sc.P[ed_['ids']] += (dx, dy)
        sc.over = None                                      # обзор — уже не тот: кадры рисуются сами
        sc.cache.clear() if False else None
        ed_['moved'] = True
        sh_ = sc.P[ed_['ids'][0]] - ed_['orig'][ed_['ids'][0]]
        st['msg'] = (f"{ed_['sel'][0].capitalize()} {ed_['sel'][1]}: moved by {sh_[0]:+.0f}, {sh_[1]:+.0f} px "
                     f"({sh_[0] * px_um:+.2f}, {sh_[1] * px_um:+.2f} um). J: save, Backspace: undo all")

    def stitch_save():
        """Позиции кадров после правки — в join/positions_all.csv (прежние автоматические — positions_auto.csv),
        обзор пересобирается; кресты и сетка идут за кадрами при следующем открытии."""
        ed_ = st.get('edit') or {}
        if not ed_.get('moved'):
            return "Stitch not changed"
        jd_ = os.path.join(folder, 'join')
        if ed_.get('seams_changed') and ed_.get('z') is not None:
            from .stitch import seams_edit
            seams_edit(folder, ed_['ovr'])
        jp_ = os.path.join(jd_, 'positions_all.csv')
        bk_ = os.path.join(jd_, 'positions_auto.csv')
        if os.path.exists(jp_) and not os.path.exists(bk_):
            import shutil as _sh
            _sh.copy2(jp_, bk_)
        P_ = sc.P - sc.P.min(0)
        with open(jp_, 'w', newline='', encoding='utf-8') as f_:
            wr_ = csv.writer(f_)
            wr_.writerow(['id', 'x_px', 'y_px', 'scale'])
            for i in range(len(P_)):
                wr_.writerow([i, f"{P_[i, 0]:.2f}", f"{P_[i, 1]:.2f}", "1.0000"])
        try:
            from .stitch import make_preview
            busy("Stitch edited: rebuilding the overview", lambda: make_preview(folder, pos=P_, levels=False, verbose=True))
        except Exception:
            pass
        st['restart'] = True
        return "Stitch saved (the automatic one: join/positions_auto.csv)"

    def pictures(fast_):
        """Нижний слой: мозаика, ровные края, контур, снимки других выделений (в текущих st: s, Wv, Hv)."""
        Wv, Hv = st['Wv'], st['Hv']
        if abs(st['rot']) < 1e-6 and np.allclose(VIEW_M(), np.eye(2)):
            img = sc.render(st['cx'], st['cy'], st['s'], Wv, Hv, fast=fast_)
        else:                                               # повёрнутый (или по осям столика) вид: с запасом, потом поворот
            D = int(math.hypot(Wv, Hv)) + 2
            big = sc.render(st['cx'], st['cy'], st['s'], D, D, fast=fast_)
            Rm = R()
            t = mid() - Rm @ np.array([D / 2, D / 2])
            img = cv2.warpAffine(big, np.float32(np.c_[Rm, t]), (Wv, Hv), flags=cv2.INTER_LINEAR)
        img = crop_edges(img)                               # G: ровные края, без ступенек
        img = cut_outline(img)
        return draw_finer(img)                              # снимки всех выделений — слоями по списку

    def render():
        """Кадр окна слоями: картинки (дорого) -> сетка, контуры, подписи -> то, что за мышью (линейка, рамка,
        U), полосы. Двигается только мышь — нижние слои берутся готовыми (st['_reuse']: 'top' | 'grid')."""
        Wv, Hv = st['Wv'], st['Hv']
        pf_ = st['timing']
        pf_.start()
        fast_ = _t.monotonic() < st.get('fast_until', 0)  # тянут / крутят колесо: вдвое грубее, потом — чисто
        lk_ = (st['cx'], st['cy'], st['s'], st['rot'], Wv, Hv, tuple(np.ravel(VIEW_M())), fast_)
        reuse_ = st.pop('_reuse', None)
        lc_, tc_ = st.get('_layers'), st.get('_top')
        top_ = reuse_ == 'top' and tc_ is not None and tc_[0] == lk_
        if top_:
            img = tc_[1].copy()                             # за мышью — только инструмент: всё под ним готово
        elif reuse_ in ('top', 'grid') and lc_ is not None and lc_[0] == lk_:
            img = lc_[1].copy()                             # картинки — готовые, сетку и контуры — заново
        else:
            if fast_ and Wv * Hv > 250000:                  # вдвое меньше по каждой оси — вчетверо быстрее
                s0_ = st['s']
                st['s'], st['Wv'], st['Hv'] = s0_ / 2, max(1, Wv // 2), max(1, Hv // 2)
                try:
                    img = pictures(True)
                finally:
                    st['s'], st['Wv'], st['Hv'] = s0_, Wv, Hv
                img = cv2.resize(img, (Wv, Hv), interpolation=cv2.INTER_LINEAR)
            else:
                img = pictures(fast_)
            st['_layers'] = (lk_, img.copy())
        pf_.mark('pictures')
        if not top_:
            img = overlays(img)
            st['_top'] = (lk_, img.copy())
        return tools(img, pf_)

    def overlays(img):
        """Средний слой: сетка, оттенки выделений, их контуры и подписи, кресты."""
        Wv, Hv = st['Wv'], st['Hv']
        pf_ = st['timing']
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
            if sel_ and step >= 7:                          # край выбранного — зелёной линией
                cv2.polylines(img, list(np.int32(np.round(quads[sel_]))), True, green, 1, cv2.LINE_8)
            if step >= 34 and len(cells) <= 3000 and len(g.labels) >= 2:   # номера — когда человек дал два (N)
                num = g.numbering()
                fs = min(0.5, step / 160)
                for c, pts in zip(cells, quads):
                    u, v = pts[np.argmin(pts.sum(1))] + (4, 4 + 14 * fs / 0.5)
                    txt = "{},{}".format(*g.name(*c, num))
                    col = (0, 230, 255) if c in g.labels else ((255, 255, 255) if g.labels else (170, 170, 170))
                    cv2.putText(img, txt, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, txt, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
            if step >= 34 and len(g.labels) < 2:            # пока номеров нет — только те, что уже введены (N)
                for c, pts in zip(cells, quads):
                    if c in g.labels:
                        u, v = pts[np.argmin(pts.sum(1))] + (4, 4 + 14 * min(0.5, step / 160) / 0.5)
                        txt = "{},{}".format(*g.labels[c])
                        cv2.putText(img, txt, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, min(0.5, step / 160), (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(img, txt, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, min(0.5, step / 160), (0, 230, 255), 1, cv2.LINE_AA)
            if st.get('lab'):                               # вводимый номер — у квадрата
                pts = np.array([to_view(q) for q in g.corners(*st['lab'][0])])
                cv2.polylines(img, [np.int32(pts)], True, (0, 230, 255), 2, cv2.LINE_8)
            if False:
                pass
            elif g.selected:
                st['msg'] = st['msg'] if 'zoom in' in st['msg'] else st['msg']
        img = cv2.copyTo(base_, void_, img)
        if g.ready():                                         # красная решётка — бесконечная: на всё окно,
            v1, v2 = g.cr.vectors()                           # и над мозаикой, и вне её, сквозь выделения
            step = min(np.hypot(*v1), np.hypot(*v2)) * st['s']
            if step >= 1.5:
                try:
                    Mg_ = np.c_[v1, v2]
                    ij_ = np.array([np.linalg.solve(Mg_, from_view(v) - g.cr.O)
                                    for v in ((0, 0), (Wv, 0), (0, Hv), (Wv, Hv))])
                    i0_, j0_ = np.floor(ij_.min(0)).astype(int) - 1
                    i1_, j1_ = np.ceil(ij_.max(0)).astype(int) + 1
                    k_ = max(1, int(math.ceil(6 / step)))     # совсем мелко — каждая k-я линия
                    for i in range(i0_ - i0_ % k_, i1_ + 1, k_):
                        a_, b_ = to_view(g.cr.O + i * v1 + j0_ * v2), to_view(g.cr.O + i * v1 + j1_ * v2)
                        cv2.line(img, tuple(np.int32(np.round(a_))), tuple(np.int32(np.round(b_))), red, 1, cv2.LINE_8)
                    for j in range(j0_ - j0_ % k_, j1_ + 1, k_):
                        a_, b_ = to_view(g.cr.O + i0_ * v1 + j * v2), to_view(g.cr.O + i1_ * v1 + j * v2)
                        cv2.line(img, tuple(np.int32(np.round(a_))), tuple(np.int32(np.round(b_))), red, 1, cv2.LINE_8)
                except (np.linalg.LinAlgError, ValueError, TypeError):
                    pass
        pf_.mark('grid')
        pyramid_tint(img)                                     # внутри выделений — всё, кроме выбранного в них
        pf_.mark('tint')
        H_img, W_img = img.shape[:2]
        mks_ = car.get('marks') or set()
        dqo_ = np.asarray(st['al']['dq'], float) if st.get('al') else np.zeros(2)   # совмещение: контуры других едут
        me_o_ = sc.st.meta.get('sample_uid')
        labs_ = []
        for q_, c_, lab_, dash_, new_, *u_ in sample_outlines():      # выделения образца на этой картинке
            sh_o_ = dqo_ if (u_ and u_[0] != me_o_) else 0
            pv_ = np.array([to_view(p_ + sh_o_) for p_ in q_], float)
            if pv_[:, 0].max() < 0 or pv_[:, 1].max() < 0 or pv_[:, 0].min() > W_img or pv_[:, 1].min() > H_img:
                continue
            if u_ and u_[0] in mks_:                        # отмеченное (Ctrl+щелчок) — белой каймой
                cv2.polylines(img, [np.int32(np.round(pv_))], True, (255, 255, 255), 5, cv2.LINE_8)
            if dash_:
                UI.dashed_poly(img, pv_, c_, 2 if new_ else 1)
            else:
                cv2.polylines(img, [np.int32(np.round(pv_))], True, c_, 2, cv2.LINE_8)
            x_l, y_l = pv_[:, 0].min(), pv_[:, 1].min()
            if 0 <= x_l < W_img - 60 and 30 <= y_l < H_img:
                labs_.append((x_l, y_l, lab_, c_))
        draw_new_drafts(img)                                  # выбранное здесь — черновиком сразу
        draw_subgrids(img)                                    # подсетки выделений и их клетки кисти
        for k_, q in enumerate([] if st.get('seeded') else g.cr.marks):   # подставленные кресты — не показывать
            u_, v_ = to_view(q)
            UI.dot(img, u_, v_, yellow if k_ == st.get("mdrag") else red)
        for x_l, y_l, lab_, c_ in labs_:                      # подписи выделений — поверх точек крестов
            tw_ = UI.size(lab_, 0.42)[0]
            UI.panel(img, x_l, y_l - 20, x_l + tw_ + 10, y_l - 2, 0.8, c_)
            UI.put(img, lab_, x_l + 5, y_l - 6, 0.42, c_)
        pf_.mark('outlines')
        return img

    def tools(img, pf_):
        """Верхний слой: то, что за мышью (рамка, U, линейка, швы), полосы, подсказки; и список справа."""
        Wv, Hv = st['Wv'], st['Hv']
        yellow = (0, 220, 255)
        if st['box']:
            bx0, by0, bx1, by1, shb = st['box']
            cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 140, 255) if shb else yellow, 1)
        if st['mode'] == 'poly' and st.get('poly'):          # U: многоугольник, следующая сторона, направляющие
            P_ = st['poly']
            V_ = [to_view(P_['R'].T @ np.asarray(v, float)) for v in P_['v']]
            q_, gd_ = poly_next(st['hov']) if st.get('hov') is not None else (None, [])
            if q_ is not None:
                qv_ = to_view(P_['R'].T @ q_)
                for ax_, val_ in gd_:                          # направляющая: через всё окно
                    f0_ = np.array(q_, float)
                    f1_ = f0_.copy()
                    k_ = 1 if ax_ == 'x' else 0
                    f0_[k_], f1_[k_] = q_[k_] - 1e5 / st['s'], q_[k_] + 1e5 / st['s']
                    a_, b_ = (to_view(P_['R'].T @ f0_), to_view(P_['R'].T @ f1_))
                    cv2.line(img, tuple(np.round(a_).astype(int)), tuple(np.round(b_).astype(int)), (120, 120, 120), 1,
                             cv2.LINE_AA)
            from . import rectpoly as RP_
            if len(P_['v']) >= 2 and q_ is not None:           # как замкнётся (угол — сам)
                ring_, _w = RP_.close(P_['v'] + [list(q_)])
                if ring_ is not None:
                    Rv_ = np.array([to_view(P_['R'].T @ np.asarray(v, float)) for v in ring_])
                    ov_ = img.copy()
                    cv2.fillPoly(ov_, [np.round(Rv_).astype(np.int32)], (0, 200, 255))
                    cv2.addWeighted(ov_, 0.18, img, 0.82, 0, img)
            for a_, b_ in zip(V_, V_[1:] + ([qv_] if q_ is not None else [])):
                cv2.line(img, tuple(np.round(a_).astype(int)), tuple(np.round(b_).astype(int)), (0, 200, 255), 2,
                         cv2.LINE_AA)
            for i_, v_ in enumerate(V_):
                cv2.circle(img, tuple(np.round(v_).astype(int)), 6 if i_ == 0 else 4, (0, 200, 255), 2 if i_ == 0 else -1,
                           cv2.LINE_AA)
            if q_ is not None:
                cv2.circle(img, tuple(np.round(qv_).astype(int)), 4, (255, 255, 255), 1, cv2.LINE_AA)
        if st['mode'] == 'ruler' and st['ruler']:            # линейка: точки, отрезки, перпендикуляры, подпись
            P_ = list(st['ruler'])
            n_ = RK_N[st.get('rk', 'pp')]
            if len(P_) < n_ and st.get('mouse') is not None:
                P_ = P_ + [from_view(st['mouse'])]
            r_ = ruler_geom(P_) if len(P_) >= n_ else None
            cy_ = (0, 220, 255)
            for p_ in P_:
                u_ = to_view(p_)
                UI.plus(img, u_[0], u_[1], cy_)
            if r_ is None and len(P_) >= 2:                   # недостроено: отрезок по двум первым
                UI.ruler(img, to_view(P_[-2]), to_view(P_[-1]), "")
            if r_ is not None:
                Hh_, Ww_ = img.shape[:2]
                for a0_, b0_ in r_.get('lines', []):            # прямые — тонко через всё окно
                    av_, bv_ = np.array(to_view(a0_)), np.array(to_view(b0_))
                    dv_ = (bv_ - av_) / max(1e-9, np.hypot(*(bv_ - av_)))
                    L2_ = 2 * (Ww_ + Hh_)
                    cv2.line(img, tuple(np.int32(av_ - dv_ * L2_)), tuple(np.int32(av_ + dv_ * L2_)), (0, 150, 180), 1, cv2.LINE_8)
                for k2_, (a0_, b0_) in enumerate(r_['segs']):
                    UI.ruler(img, to_view(a0_), to_view(b0_), r_['text'] if k2_ == len(r_['segs']) - 1 else "")
        if st['mode'] == 'stitch' and (st.get('edit') or {}).get('z') is not None:   # швы: цвет по невязке
            ed_ = st['edit']
            z_, c_ = ed_['z'], np.array([sc.W / 2, sc.H / 2])
            for k_ in range(len(z_['I'])):
                a_ = to_view(sc.P[z_['I'][k_]] + c_)
                b_ = to_view(sc.P[z_['J'][k_]] + c_)
                m_ = to_view(seam_pt(k_))
                if not (-20 < m_[0] < img.shape[1] + 20 and -20 < m_[1] < img.shape[0] + 20):
                    continue
                r_ = ed_['res'][k_]
                col_ = (110, 110, 110) if ed_['w'][k_] <= 0 else (80, 200, 80) if r_ < 0.5 else \
                    (0, 210, 255) if r_ < 1.5 else (60, 60, 255)
                if np.hypot(b_[0] - a_[0], b_[1] - a_[1]) > 24:
                    cv2.line(img, (int(round(a_[0])), int(round(a_[1]))), (int(round(b_[0])), int(round(b_[1]))),
                             col_, 1, cv2.LINE_8)
                cv2.circle(img, (int(round(m_[0])), int(round(m_[1]))), 3, col_, -1, cv2.LINE_8)
                if ed_['w'][k_] <= 0:
                    cv2.circle(img, (int(round(m_[0])), int(round(m_[1]))), 5, col_, 1, cv2.LINE_8)
        if st['mode'] == 'stitch' and (st.get('edit') or {}).get('ids'):   # кадры, которые двигают
            for i_ in st['edit']['ids']:
                p_ = sc.P[i_]
                q_ = np.array([to_view(c_) for c_ in (p_, p_ + (sc.W, 0), p_ + (sc.W, sc.H), p_ + (0, sc.H))])
                cv2.polylines(img, [np.int32(np.round(q_))], True, (255, 220, 0), 1, cv2.LINE_8)
        mode = {'look': 'Look', 'cross': 'Mark crosses', 'select': 'Select', 'stitch': 'Edit stitch',
                'paint': 'Brush', 'label': 'Number squares', 'ruler': 'Ruler', 'align': 'Align', 'poly': 'Outline'}.get(st['mode'], st['mode'])
        mcol = {'look': (170, 170, 178), 'cross': (90, 90, 255), 'select': (90, 210, 90), 'stitch': (255, 220, 0),
                'paint': (90, 210, 90), 'label': UI.ACCENT, 'ruler': (0, 220, 255), 'align': (255, 120, 255)}.get(st['mode'], UI.ACCENT)
        if st.get('lab'):
            st['msg'] = f"Square number X Y: {st['lab'][1]}_   (Enter = set, Esc = cancel)"
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
        if car.get('marks'):
            it += [f"marked {len(car['marks'])}"]
        if car.get('new_obj', 'draft') != 'draft':
            it += [f"new: {car['new_obj']}x"]
        UI.facts(img, it, x_, bh - 11, fs)
        if st['straight']:
            w_ = UI.size("Straight", fs)[0] + 18
            UI.chip(img, "Straight", Wv - w_ - 10, bh - 11, fs, (235, 170, 60))
        hl = [HELP_MODE.get(st['mode'], ''), HELP_V2]
        sbh = UI.size("Xg", 0.5)[1] + 16                    # строка состояния — в самом низу
        hb = len(hl) * (th_ + 12) + 12 + sbh
        UI.panel(img, 0, Hv - hb, Wv, Hv - sbh, 0.82)       # над ней: клавиши этого режима
        for k_, h_ in enumerate(hl):
            UI.keys_line(img, h_, 12, Hv - hb + 6 + (k_ + 1) * (th_ + 12) - 4, fs, Wv - 10)
        from .jobs import summary_all as _qsum
        qs_ = _qsum()                                      # очередь окна микроскопа идёт и здесь: видно, что в фоне
        if not qs_ and smp is not None:                    # окно микроскопа — другой процесс: его состояние очереди
            try:
                sq_ = json.load(open(os.path.join(smp.folder, '.queue_status.json'), encoding='utf-8'))
                if sq_.get('summary') and _t.time() - float(sq_.get('t', 0)) < 5:
                    qs_ = sq_['summary']
            except (OSError, ValueError):
                pass
        UI.status_bar(img, 0, Hv - sbh, Wv, Hv, None, [mode, f"{len(sc.st)} frames"] + ([qs_] if qs_ else []), scale=0.5)
        if st['msg'] != st.get('_lm'):
            st['_lm'], st['_lt'] = st['msg'], _t.monotonic()
        prompt = bool(st.get('lab'))
        if st['msg'] and (prompt or _t.monotonic() - st['_lt'] < 8):
            low = st['msg'].lower()
            kind = 'err' if (st['msg'].startswith('[E') or 'no ' in low[:4] or 'not ' in low or 'disagree' in low) \
                else ('info' if st.get('lab') else 'ok')
            UI.toast(img, st['msg'], Wv / 2, Hv - hb - 18, 0.55, kind, Wv - 40)
        if st.get('help'):                                  # H: все клавиши (переносом, влезает в окно)
            UI.help_panel(img, HELP_ALL)
        pf_.mark('tools, bars')
        if pf_.on:                                          # `: мс на кадр по этапам
            UI.prof_panel(img, pf_, Wv - 10, 40)
        st['scene'] = img
        st['prof_dirty'] = False
        out_ = np.hstack([with_prof(img), carousel_img()])
        pf_.mark('list')
        return out_

    def carousel_img():
        """Меню справа — как в окне микроскопа: все выделения одним списком, в порядке слоёв (верх — верхний слой):
        снятые (с картинкой), не снятые, без объектива (draft) и новое из выбранных клеток. Галочка — в экспорт."""
        base_ = _car_items(subs, folder, car)
        items_, cmap_ = [], []
        try:
            regs_ = regions_now()[0] if smp is not None else []
        except Exception:
            regs_ = []
        sub_of_ = {}
        for k_, f_ in enumerate(subs):
            u_ = P0_.scan_uid(f_)
            if u_:
                sub_of_[u_] = k_
        rk_ = lambda r_: r_.get('rank') if r_.get('rank') is not None else -1e9
        regs_ = sorted(regs_, key=lambda r_: (0 if r_.get('pending') else 1, rk_(r_), -int(r_.get('num', 0) or 0)))
        mk_ = car.setdefault('marks', set())
        sel_k = None
        fo_ = car.get('focus')
        for r_ in regs_:
            if r_.get('kind') == 'group':
                continue
            u_ = r_.get('uid')
            new_ = bool(r_.get('pending'))
            dr_ = r_.get('kind') == 'draft'
            chk_ = None if new_ else r_.get('export', False)
            if u_ in sub_of_:                               # снятое: картинка и кадры
                k_ = sub_of_[u_]
                it_ = dict(base_[k_], chosen=u_ in mk_, check=chk_, title=f"Selection {r_.get('num', '')}", more=True,
                           alpha=alphas().get(u_, 1.0),
                           color=SM.color(int(r_.get('num', 1)) - 1), obj='draft' if dr_ else f"{r_.get('obj')}x")
                if subs[k_] == folder and not fo_:
                    sel_k = len(items_)
                items_.append(it_)
                cmap_.append(('sub', k_, u_))
            else:
                nc_ = len(r_.get('cells') or [])
                items_.append(dict(title=("Selection (new)" if new_ else f"Selection {r_.get('num', '')}"),
                                   color=(235, 235, 235) if new_ else SM.color(int(r_.get('num', 1)) - 1),
                                   lines=[(f"{nc_} cell{'s' if nc_ != 1 else ''}" if nc_ else "")
                                          + ("" if not new_ else "  - Enter: make it a selection")
                                          if new_ else ("not scanned yet" + (f", {nc_} cells" if nc_ else ""))]
                                   + ([] if dr_ or new_ else [f"{r_.get('obj')}x: S queues its scan"]),
                                   go=False, muted=new_, chosen=u_ in mk_, check=chk_,
                                   **({} if new_ else {'obj': 'draft' if dr_ else f"{r_.get('obj')}x"})))
                if fo_ and u_ == fo_:
                    sel_k = len(items_)
                cmap_.append(('draft', u_, subs.index(folder) if folder in subs else 0))
        car['map'] = cmap_
        pan_, car['hits'], car['content'] = SM.carousel(st['Hv'], items_, sel_k, car['scroll'], title="Selections",
                                                        go=False, hover=car.get('hxy'))
        dg_ = car.get('drag')
        if dg_ and dg_.get('on'):                           # тянут карточку: список сразу такой, каким станет
            cards_ = [h_ for h_ in car['hits'] if h_[4] == 'sel']     # место — по раскладке ДО перестановки
            uid_ = lambda c_: c_[2] if c_[0] == 'sub' and len(c_) > 2 else c_[1]
            src_ = next((i_ for i_, c_ in enumerate(cmap_) if uid_(c_) == dg_['uid']), None)
            if cards_ and src_ is not None:
                h_ = min(cards_, key=lambda h: 0 if h[1] <= dg_['y'] <= h[3] else min(abs(dg_['y'] - h[1]), abs(dg_['y'] - h[3])))
                idx_ = list(range(len(items_)))
                idx_.remove(src_)
                if h_[5] != src_:
                    idx_.insert(idx_.index(h_[5]) + (1 if dg_['y'] > (h_[1] + h_[3]) / 2 else 0), src_)
                else:
                    idx_.insert(src_, src_)
                sel2_ = idx_.index(sel_k) if sel_k is not None and sel_k in idx_ else None
                pan_, hits2_, _c = SM.carousel(st['Hv'], [items_[i_] for i_ in idx_], sel2_, car['scroll'],
                                               title="Selections", go=False)
                sh_ = next((h2_ for h2_ in hits2_ if h2_[4] == 'sel' and h2_[5] == idx_.index(src_)), None)
                if sh_ is not None and sh_[3] > 39:          # карточку видно в рамке
                    cv2.rectangle(pan_, (sh_[0], max(39, sh_[1])), (sh_[2] - 1, sh_[3] - 1), (90, 220, 90), 2)
        return pan_

    # история (Ctrl+Z / Ctrl+Y): после каждого действия разметка (кресты, сетка, квадраты, номера)
    # сравнивается с прежней; изменилась — пара (было, стало) в историю
    hist = car['hist']

    def stub():
        """То, что функции выделений (layers) берут у микроскопа: образец с диска, объектив и кадр этой съёмки."""
        class _SampleOnly:
            pass
        sh_ = _SampleOnly()
        from . import project as PJ_
        sh_._sample = sh_.sample = PJ_.sample_obj(folder)     # заново с диска: что там сейчас
        fov_ = sc.st.meta.get('fov_um') or {}
        sh_.frame_size = (float(fov_.get('x', 100.0)), float(fov_.get('y', 100.0)))
        sh_.objective = sc.st.meta.get('objective') or 10
        sh_._off = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        sh_.motor_step = float(sc.st.meta.get('motor_step_um') or 5.56)
        sh_._area_seq = int(sh_._sample.data.get('seq', 0) or 0)
        PJ_.load_regions(sh_)
        return sh_

    def brush_now():
        """Выбранные клетки -> выделения в sample.json сразу (не дожидаясь окна микроскопа): без объектива (draft)
        или с объективом, выбранным для новых (Shift+G)."""
        if smp is None:
            return
        try:
            from .layers import sync_brush, objective_to
            from . import project as PJ_
            g.save(px_um)
            sh_ = stub()
            old_ = {r_['uid'] for r_ in sh_._regions}
            r_ = sync_brush(sh_)
            no_ = car.get('new_obj', 'draft')
            if r_ and no_ != 'draft':                        # новые — сразу с объективом
                got_ = [objective_to(sh_, k_, no_) for k_, x_ in enumerate(sh_._regions)
                        if x_['uid'] not in old_ and x_.get('kind') == 'draft']
                bad_ = [m_ for ok_, m_ in got_ if not ok_]
                r_ += f"; {no_}x" + (f" ({len(bad_)} stay draft: {bad_[0]})" if bad_ else "")
            if r_:
                PJ_.save_regions(sh_)
                if no_ != 'draft':
                    regrid()
                st['msg'] = r_ + " (in the list on the right)"
                st['car_dirty'] = True
                st['_so_key'] = None
        except Exception as e:
            L.debug('VIW-009', f"Brush -> selections skipped: {e}")

    def regrid():
        """Выделение с объективом живёт само: его клетки ушли из кисти (grid.json) — взять кисть с диска,
        иначе следующая запись вернула бы их и сделала из них ещё одно выделение."""
        from .mask import Grid as _G
        g2_ = _G(folder)
        g.selected = set(g2_.selected)
        h_last[:2] = [g_sig(), g_snap()]

    def poly_cands(f_):
        """Привязки многоугольника у мыши (рамка): узлы сетки рядом (кресты) — x / y и точки."""
        P_ = st['poly']
        if not g.ready():
            return [], [], []
        v1_, v2_ = (np.asarray(v, float) for v in g.cr.vectors())
        O_ = np.asarray(g.cr.O, float)
        try:
            ij_ = np.linalg.solve(np.column_stack([v1_, v2_]), P_['R'].T @ np.asarray(f_, float) - O_)
        except np.linalg.LinAlgError:
            return [], [], []
        i0, j0 = np.round(ij_).astype(int)
        nd_ = [P_['R'] @ (O_ + i * v1_ + j * v2_) for i in range(i0 - 6, i0 + 7) for j in range(j0 - 6, j0 + 7)]
        return [n[0] for n in nd_], [n[1] for n in nd_], nd_

    def poly_next(xy):
        """Куда встанет следующая вершина под мышью xy (окно). -> (q рамки, направляющие)."""
        from . import rectpoly as RP
        P_ = st['poly']
        f_ = P_['R'] @ from_view((xy[0] + 0.5, xy[1] + 0.5))
        xs_, ys_, nd_ = poly_cands(f_)
        return RP.next_point(P_['v'], f_, xs_, ys_, 10.0 / st['s'], nd_)

    def poly_click(xy):
        from . import rectpoly as RP
        P_ = st['poly']
        if len(P_['v']) >= 3 and np.hypot(*(to_view(P_['R'].T @ np.asarray(P_['v'][0])) - xy)) <= 10:
            st['msg'] = poly_finish()
            return
        q_, _gd = poly_next(xy)
        P_['v'], why_ = RP.add(P_['v'], q_)
        st['msg'] = why_ or (f"{len(P_['v'])} points. Next: click (it goes only across or up/down, snapping to the "
                             "crosses and the points). Click the first point or Enter: close  Backspace / right "
                             "click: back  Esc: cancel")

    def poly_finish():
        """Замкнуть многоугольник и сделать из него выделение (draft или объектив новых, Shift+O). -> сообщение."""
        from . import rectpoly as RP
        P_ = st['poly']
        ring_, why_ = RP.close(P_['v'])
        if ring_ is None:
            return why_
        if smp is None:
            return "No sample: open the scan from a project"
        from .snap import mosaic_to_stage
        from shapely.geometry import Polygon as _Poly
        from .layers import set_geom, objective_to, new_num
        from .samples import to_top as _top
        from . import project as PJ_
        import uuid as _uu
        S_ = mosaic_to_stage(folder, np.array([P_['R'].T @ np.asarray(v, float) for v in ring_]))
        pg_ = _Poly(S_)
        if not pg_.is_valid or pg_.area <= 0:
            return "This outline cannot be a selection"
        sh_ = stub()
        r_ = dict(uid=_uu.uuid4().hex, num=new_num(sh_), obj=sh_.objective, fw=sh_.frame_size[0],
                  fh=sh_.frame_size[1], pts=[], thumbs=[], kind='draft')
        set_geom(sh_, r_, pg_)
        _top(sh_, r_)
        sh_._regions.append(r_)
        no_ = car.get('new_obj', 'draft')
        bad_ = ''
        if no_ != 'draft':
            ok_, m_ = objective_to(sh_, len(sh_._regions) - 1, no_)
            bad_ = '' if ok_ else f"; {m_}"
        sh_._sample.data['focus'] = r_['uid']
        PJ_.save_regions(sh_)
        sh_._sample.save()
        car['focus'] = r_['uid']
        st['_so_key'] = st['_sg_key'] = None
        st['car_dirty'] = True
        st['poly'], st['mode'] = None, 'look'
        return (f"Selection {r_['num']} made from the outline ({len(ring_)} corners, "
                f"{'draft' if no_ == 'draft' else f'{no_}x'}), chosen on the right{bad_}")

    def brush_commit():
        """Enter: выбранные клетки -> новое выделение (каждая фигура — своё), с объективом для новых (Shift+G) или
        draft; оно живёт само (клетки уходят из кисти), сразу в списке и выбрано. -> сообщение."""
        if smp is None:
            return "No sample: open the scan from a project"
        from .layers import sync_brush, objective_to, detach
        from . import project as PJ_
        g.save(px_um)
        sh_ = stub()
        old_ = {r_['uid'] for r_ in sh_._regions}
        sync_brush(sh_, make=True)
        new_ = [k_ for k_, x_ in enumerate(sh_._regions) if x_['uid'] not in old_]
        if not new_:
            return "Nothing to make: select cells first (M, click / drag), then Enter"
        no_ = car.get('new_obj', 'draft')
        bad_ = []
        for k_ in new_:
            x_ = sh_._regions[k_]
            if no_ != 'draft':
                ok_, m_ = objective_to(sh_, k_, no_)
                if not ok_:
                    bad_.append(m_)
            if x_.get('src'):
                detach(sh_, x_)                             # живёт само: клетки ушли из кисти
        u_ = sh_._regions[new_[-1]]['uid']
        sh_._sample.data['focus'] = u_                      # окно микроскопа выберет его же
        PJ_.save_regions(sh_)
        sh_._sample.save()
        hb_ = list(h_last[:2])
        regrid()
        h_last[:2] = hb_                                    # клетки ушли в выделение — одним шагом Ctrl+Z с ним
        car['focus'] = u_
        st['_so_key'] = st['_sg_key'] = None
        st['car_dirty'] = True
        nm_ = ", ".join(f"Selection {sh_._regions[k_]['num']}" for k_ in new_)
        return (f"{nm_} made ({'draft' if no_ == 'draft' else f'{no_}x'}), chosen on the right"
                + (f"; {bad_[0]}" if bad_ else "") + "   Shift+O: objective of the next new ones")

    def sel_geoms():
        """Выделения образца на столике: [(uid, shapely, площадь)] — для щелчков по ним (кэш как у контуров)."""
        if smp is None:
            return []
        key_ = (_mt(smp.file),
                _mt(os.path.join(folder, 'grid.json')))
        if st.get('_sg_key') == key_:
            return st['_sg']
        from .layers import from_json
        from shapely.geometry import box as _box
        out_ = []
        for r_ in regions_now()[0]:
            if r_.get('kind') == 'group' or r_.get('pending'):
                continue
            try:
                L_c = lat_now() if r_.get('cells') else None
                if L_c is not None:
                    gm_ = LT.cells_geom(L_c, [tuple(c) for c in r_['cells']], LT.level(r_))
                elif r_.get('shape'):
                    gm_ = from_json(r_['shape'])
                else:
                    (ax, ay), (bx, by) = r_['A'], r_['B']
                    fw_, fh_ = r_.get('fw', 0), r_.get('fh', 0)
                    gm_ = _box(min(ax, bx) - fw_ / 2, min(ay, by) - fh_ / 2, max(ax, bx) + fw_ / 2, max(ay, by) + fh_ / 2)
                out_.append((r_['uid'], gm_, gm_.area))
            except Exception:
                continue
        st['_sg_key'], st['_sg'] = key_, out_
        return out_

    def sel_at(xy):
        """Выделение под точкой окна: самое маленькое из тех, в которых она лежит. -> uid или None."""
        from .snap import mosaic_to_stage
        from shapely.geometry import Point
        S_ = mosaic_to_stage(folder, [from_view(xy)])[0]
        P_ = Point(float(S_[0]), float(S_[1]))
        hit_ = [(a_, u_) for u_, gm_, a_ in sel_geoms() if gm_.contains(P_)]
        return min(hit_)[1] if hit_ else None

    def mark_toggle(uid):
        mk_ = car.setdefault('marks', set())
        (mk_.discard if uid in mk_ else mk_.add)(uid)
        regs_ = {r_['uid']: r_ for r_ in regions_now()[0]}
        st['msg'] = (f"Marked {len(mk_)}: " + ", ".join(f"Selection {regs_[u_]['num']}" for u_ in mk_ if u_ in regs_)
                     + "   O: objective, S: queue the scans, [ ]: transparency, Del: delete, Esc: unmark") if mk_ else "Nothing marked"
        st['car_dirty'] = st['dirty'] = True

    def targets():
        """С чем делать дело: отмеченные (Ctrl+щелчок), иначе — ВЫБРАННОЕ в списке (не то, что под мышью). -> [uid]."""
        mk_ = [u_ for u_ in car.get('marks', set()) if any(r_['uid'] == u_ for r_ in regions_now()[0])]
        car['marks'] = set(mk_)
        if mk_:
            return mk_
        u_ = car.get('focus') or sc.st.meta.get('sample_uid')
        return [u_] if u_ and any(r_['uid'] == u_ for r_ in regions_now()[0]) else []

    def scanned(r_):
        from .project import Selection as _Sel
        try:
            return _Sel(smp, r_).latest is not None
        except Exception:
            return False

    def set_objective(uids, o):
        """Объектив выделений (только не снятых): o — 4 … 100 или 'draft'. -> сообщение."""
        from .layers import objective_to
        from . import project as PJ_
        sh_ = stub()
        ok_, bad_ = [], []
        for u_ in uids:
            k_ = next((i_ for i_, r_ in enumerate(sh_._regions) if r_['uid'] == u_), None)
            if k_ is None:
                continue
            r_ = sh_._regions[k_]
            o_ok, m_ = objective_to(sh_, k_, o)
            (ok_ if o_ok else bad_).append(r_['num'] if o_ok else m_)
        if ok_:
            PJ_.save_regions(sh_)
            regrid()
            st['_so_key'] = st['_sg_key'] = None
            st['car_dirty'] = True
        return ((f"Selection{'s' if len(ok_) > 1 else ''} {', '.join(map(str, ok_))}: "
                 + ("draft (no objective)" if o == 'draft' else f"{o}x") if ok_ else "Objective not changed")
                + (f";  {len(bad_)} not: {bad_[0]}" if bad_ else ""))

    def set_export(uid):
        """Галочка карточки: идёт ли выделение в экспорт. -> сообщение."""
        from . import project as PJ_
        sh_ = stub()
        r_ = next((x_ for x_ in sh_._regions if x_['uid'] == uid), None)
        if r_ is None:
            return "Not saved yet"
        r_['export'] = not r_.get('export', False)
        PJ_.save_regions(sh_)
        return f"Selection {r_['num']}: " + ("goes into the export" if r_['export'] else "not in the export")

    def reorder_to(src_uid, dst_uid, after):
        """Перетащили карточку: src — перед dst (after — после). Порядок списка = слои."""
        from . import project as PJ_
        sh_ = stub()
        ix_ = {r_['uid']: i_ for i_, r_ in enumerate(sh_._regions)}
        if src_uid not in ix_ or dst_uid not in ix_ or src_uid == dst_uid:
            return None
        from .samples import init_areas as _ia
        _ia(sh_)
        r2_ = SM.reorder(sh_, ix_[src_uid], ix_[dst_uid], after, save=False)
        PJ_.save_regions(sh_)
        st['_fl_key'] = None
        return r2_

    def queue_scans(uids):
        """Q: съёмки выделений — в очередь окна микроскопа (того же образца). -> сообщение."""
        from .jobs import _ALL, tm as _qtm
        q_ = next((q2_ for q2_ in _ALL if getattr(q2_.m, '_sample', None) is not None and smp is not None and
                   os.path.abspath(q2_.m._sample.folder) == os.path.abspath(smp.folder)), None)
        if q_ is None:                                      # окно микроскопа — другой процесс: запрос ему через файл
            regs_ = {r_['uid']: r_ for r_ in regions_now()[0]}
            ok_ = [u_ for u_ in uids if u_ in regs_ and regs_[u_].get('kind') not in ('draft', 'group')]
            skip_ = len(uids) - len(ok_)
            if not ok_:
                return "Nothing to scan: give the selection an objective (O) first"
            st_q_ = os.path.join(smp.folder, '.queue_status.json')
            alive_ = _t.time() - _mt(st_q_) < 600
            with open(os.path.join(smp.folder, '.queue_requests.jsonl'), 'a', encoding='utf-8') as fo_:
                fo_.write(json.dumps(dict(op='scan', uids=ok_)) + "\n")
            return (f"Sent {len(ok_)} scan{'s' if len(ok_) != 1 else ''} to the queue of the microscope window"
                    + ("" if alive_ else " (it takes them when m.manual() with this sample is open)")
                    + (f"; {skip_} without an objective skipped" if skip_ else ""))
        regs_ = {r_['uid']: r_ for r_ in regions_now()[0]}
        b_ = object()
        inq_ = {j_.uid for j_ in q_.active() if j_.kind == 'scan'}
        n_, skip_ = 0, []
        for u_ in uids:
            r_ = regs_.get(u_)
            if r_ is None or u_ in inq_:
                continue
            if r_.get('kind') == 'draft':
                skip_.append(f"Selection {r_['num']} has no objective (G)")
                continue
            if r_.get('kind') == 'group':
                skip_.append(f"Group {r_['num']}: mark its selections")
                continue
            q_.scan(u_, batch=b_)
            n_ += 1
        wf_ = q_.waiting_for()
        return ((f"Queued {n_} scan{'s' if n_ != 1 else ''}" if n_ else "Nothing queued")
                + (f" ({', '.join(f'{c_} wait for {o_}x' for o_, c_ in sorted(wf_.items()))}: put the objective in, "
                   "then S in the microscope window)" if wf_ else "")
                + (f"; the queue is done in ~{_qtm(q_.total_left())}" if q_.active() else "")
                + (f";  not: {skip_[0]}" + (f" (+{len(skip_) - 1})" if len(skip_) > 1 else "") if skip_ else ""))

    def g_sig():
        return (tuple(tuple(np.round(np.asarray(q, float), 3)) for q in g.cr.marks), tuple(g.cr.ij),
                hash(frozenset(g.selected)), tuple(sorted(g.labels.items())), g.refined is None,
                )

    def g_snap():
        return {k_: (v_ if k_ == 'folder' else copy.deepcopy(v_)) for k_, v_ in g.__dict__.items()}

    def g_label():
        t_ = (st['msg'] or "change").split(":")[0].split("(")[0].strip()
        return t_[:60]

    def d_files():
        """Файлы образца, которые меняет просмотр: sample.json (выделения, порядок, объективы, галочки, решётка,
        основная), view.json (прозрачность), align.json съёмок (совмещение)."""
        if smp is None:
            return []
        import glob as _gl
        return [smp.file, os.path.join(smp.folder, 'view.json'), os.path.join(folder, 'grid.json')] + sorted(
            _gl.glob(os.path.join(smp.folder, 'scans', '*', 'align.json')))

    def d_snap():
        """Содержимое этих файлов: {путь: текст или None}."""
        out_ = {}
        for p_ in d_files():
            try:
                out_[p_] = open(p_, encoding='utf-8').read()
            except OSError:
                out_[p_] = None
        return out_

    def d_mt():
        return tuple((p_, _mt(p_)) for p_ in d_files())

    def d_restore(t_):
        """Вернуть файлы образца как были (нет файла — удалить)."""
        al_ = False
        for p_ in set(t_) | {p2_ for p2_ in d_files() if p2_.endswith('align.json')}:
            want_ = t_.get(p_)
            try:
                now_ = open(p_, encoding='utf-8').read()
            except OSError:
                now_ = None
            if want_ == now_:
                continue
            al_ = al_ or p_.endswith('align.json')
            if want_ is None:
                try:
                    os.remove(p_)
                except OSError:
                    pass
            else:
                tmp_ = p_ + '.tmp'
                with open(tmp_, 'w', encoding='utf-8') as f_:
                    f_.write(want_)
                os.replace(tmp_, p_)
        if smp is not None:
            try:
                smp.data = json.load(open(smp.file, encoding='utf-8'))
            except (OSError, ValueError):
                pass
        st['_so_key'] = st['_sg_key'] = st['_fl_key'] = None
        st.pop('_lat', None)
        st['car_dirty'] = st['dirty'] = True
        return al_

    def h_apply(e_, back):
        """Откатить (back) или повторить действие e_. Разметка другого выделения — открыть его."""
        if e_.get('kind') == 'remove':
            import shutil as _sh
            src, dst = (e_['dst'], e_['src']) if back else (e_['src'], e_['dst'])
            if os.path.exists(src) and not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not back and os.path.abspath(src) == os.path.abspath(folder):
                    sc.close()
                _sh.move(src, dst)
            st['switch'] = e_['src'] if back else (folder if os.path.abspath(e_['src']) != os.path.abspath(folder) else False)
            return
        t_ = e_['before'] if back else e_['after']
        if 'g' in t_:                                       # разметка + файлы образца
            if t_.get('d') is not None and d_restore(t_['d']):   # совмещение поменялось — открыть заново
                st['switch'] = folder if os.path.abspath(e_['folder']) == os.path.abspath(folder) else e_['folder']
            t_ = t_['g']
        if os.path.abspath(e_['folder']) != os.path.abspath(folder):     # разметка другого выделения
            gg = Grid(e_['folder'])
            gg.__dict__.update(copy.deepcopy({k_: v_ for k_, v_ in t_.items() if k_ != 'folder'}))
            gg.save(e_.get('px', px_um))
            st['switch'] = e_['folder']
            return
        g.__dict__.update({k_: (v_ if k_ == 'folder' else copy.deepcopy(v_)) for k_, v_ in t_.items()})
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
        me_ = next((r_ for r_ in regs_ if r_.get('kind') not in ('group', 'draft') and
                    P0_.scan_uid(folder) == r_.get('uid')), None)
        if me_ is not None:                                 # как в окне микроскопа: середина квадрата — в съёмке
            from shapely.geometry import Point as _Pt, box as _bx
            from .layers import from_json
            if me_.get('shape'):
                pg_ = from_json(me_['shape'])
            else:
                (ax_, ay_), (bx_, by_) = me_['A'], me_['B']
                pg_ = _bx(min(ax_, bx_) - me_.get('fw', 0) / 2, min(ay_, by_) - me_.get('fh', 0) / 2,
                          max(ax_, bx_) + me_.get('fw', 0) / 2, max(ay_, by_) + me_.get('fh', 0) / 2)
            S_ = mosaic_to_stage(folder, np.array([g.cr.point(i + 0.5, j + 0.5) for i, j in sq_]))
            uv_ = [uv_[i_] for i_, S1 in enumerate(S_) if pg_.buffer(0.01).contains(_Pt(*S1))]
        cells_ = {(int(np.floor(u)), int(np.floor(v))) for u, v in uv_}
        have_ = {tuple(c) for r_ in regs_ if not r_.get('factors') and r_.get('cells') for c in r_['cells']}
        out_ = []
        for fg_ in LT.figures(cells_ - have_):
            key_ = next((k_ for k_ in rf_ if k_.startswith('cell:') and
                         tuple(map(int, k_[5:].split(','))) in fg_), f"cell:{fg_[0][0]},{fg_[0][1]}")
            out_.append(dict(uid=key_, num=0, kind='draft', cells=[list(c) for c in fg_], factors=[], pending=True))
        return out_

    def lat_now():
        """Решётка образца для черновиков и подсеток здесь: по сетке ЭТОЙ съёмки, если она размечена (кресты
        только что отметили — квадраты встают строго по ним), иначе — записанная у образца."""
        try:
            L_s = json.load(open(smp.file, encoding='utf-8')).get('lattice') if smp is not None else None
        except (OSError, ValueError):
            L_s = None
        if not g.ready():
            return L_s
        v1_, _v2 = g.cr.vectors()
        key_ = (tuple(np.round(g.cr.O, 3)), tuple(np.round(v1_, 5)), tuple(sorted(g.labels.items())))
        c_ = st.get('_lat')
        if c_ is None or c_[0] != key_:
            if not g.labels and L_s and L_s.get('numbered') and st.get('_inh') != key_:
                st['_inh'] = key_                           # номера квадратов — от решётки образца (раз на сетку)
                try:
                    if LT.inherit_labels(folder, g, L_s, sc.size):
                        g.save(px_um)
                        key_ = (key_[0], key_[1], tuple(sorted(g.labels.items())))
                except Exception:
                    pass
            try:
                L_g = LT.from_grid(folder, g)
            except Exception:
                L_g = None
            st['_lat'] = c_ = (key_, L_g)
        L_g = c_[1]
        if L_g is not None and (L_g['numbered'] or not (L_s and L_s.get('numbered'))):
            return L_g
        return L_s

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
                             kind='the new Selection' if r_.get('pending') else 'Selection',
                             factors=(r_.get('factors') or []) + [n_])
        return best_

    def write_refine(uid, n, cells):
        gp_ = os.path.join(folder, 'grid.json')
        g.save(px_um)
        d_ = json.load(open(gp_, encoding='utf-8'))
        rf_ = d_.get('refine') or {}
        rf_[uid] = dict(n=int(n), cells=[list(c) for c in sorted(cells)])
        d_['refine'] = rf_
        d_['brush_t'] = _t.time()
        with open(gp_, 'w', encoding='utf-8') as f_:
            json.dump(d_, f_, indent=1)

    def set_refine(n):
        """Клавиши 2–9 над выделением (1, затем 0 — 10): его подсетка в n раз плотнее (1 — убрать)."""
        L_ = lat_now()
        ch_ = [r_ for r_ in regions_now()[0] if r_.get('uid') in (car.get('focus'),) and r_.get('cells')]
        if L_ is not None and ch_:                          # выбранное в списке (из клеток) — ему, где бы ни была мышь
            r_ = ch_[0]
            old_ = regions_now()[1].get(r_['uid']) or {}
            write_refine(r_['uid'], n, old_.get('cells', []) if int(old_.get('n') or 0) == n else [])
            st['msg'] = (f"Selection {r_['num'] or ''}: grid 1/{n} inside it (cells "
                         f"{LT.pitch_text(L_, LT.level(r_) * n)}); M selects its cells" if n > 1
                         else f"Selection {r_['num']}: no finer grid")
            return
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
        kind_ = 'Selection'
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
                     f"- Enter: make a selection of them")
        st['car_dirty'] = True

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
        """Снимки всех других выделений образца, в порядке списка (первый — верхний слой): каждый — (объектив,
        картинка, preview px -> мозаика этой (2x3), None, uid, папка, номер). Без обрезки по контуру.
        Кэш по времени sample.json, превью и совмещения."""
        if smp is None:
            return []
        from . import project as P2_
        from .snap import mosaic_to_stage, stage_to_mosaic, geometry
        from .layers import from_json
        my_obj = sc.st.meta.get('objective') or 0
        scans_ = [f_ for f_ in P2_.latest_scans(smp.folder, stitched=True) if os.path.abspath(f_) != os.path.abspath(folder)]
        ap_ = os.path.join(folder, 'align.json')
        oth_ = car.get('others', True)
        me_uid_ = sc.st.meta.get('sample_uid')
        key_ = (oth_, _mt(smp.file),
                _mt(ap_),
                tuple((f_, _mt(os.path.join(f_, 'join', 'preview_all.jpg')),
                       _mt(os.path.join(f_, 'align.json')))
                      for f_ in scans_ if os.path.exists(os.path.join(f_, 'join', 'preview_all.jpg'))))
        if st.get('_fl_key') == key_:
            return st['_fl']
        regs_ = {r_['uid']: r_ for r_ in json.load(open(smp.file, encoding='utf-8')).get('selections', [])}
        out_ = []
        for f_ in scans_:
            jp_ = os.path.join(f_, 'join', 'preview_all.jpg')
            try:
                meta_ = json.load(open(os.path.join(f_, 'meta.json'), encoding='utf-8'))
                if not os.path.exists(jp_) or (meta_.get('sample_uid') and meta_.get('sample_uid') == me_uid_):
                    continue
                if not oth_:                               # Y: только эта съёмка
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
                out_.append((meta_.get('objective') or 0, pic_, A_, ring_, meta_.get('sample_uid'), f_,
                             r_['num'] if r_ is not None else meta_.get('selection')))
            except Exception:
                continue
        out_ = [L_ + (True,) for L_ in out_]                # порядок — как в списке (latest_scans: сверху вниз)
        st['_fl_key'], st['_fl'] = key_, out_
        return out_

    def alphas(new=None):
        """Прозрачность картинок выделений в просмотре: {uid: 0..1}, файл view.json образца (общий для съёмок)."""
        if smp is None:
            return {}
        p_ = os.path.join(smp.folder, 'view.json')
        if st.get('_alpha') is None:
            try:
                st['_alpha'] = json.load(open(p_, encoding='utf-8')).get('alpha', {})
            except (OSError, ValueError):
                st['_alpha'] = {}
        if new is not None:
            st['_alpha'].update(new)
            try:
                d_ = json.load(open(p_, encoding='utf-8')) if os.path.exists(p_) else {}
            except (OSError, ValueError):
                d_ = {}
            d_['alpha'] = {k_: v_ for k_, v_ in st['_alpha'].items() if v_ < 0.99}
            with open(p_, 'w', encoding='utf-8') as fo_:
                json.dump(d_, fo_, indent=1)
        return st['_alpha']

    def layer_at(xy):
        """Снимок другого выделения под мышью (верхний по списку) -> запись finer_layers или None."""
        q_ = from_view(xy) - (np.asarray(st['al']['dq'], float) if st.get('al') else 0)
        for L_ in finer_layers():
            o_, pic_, A_, ring_ = L_[:4]
            if ring_:
                if any(cv2.pointPolygonTest(np.float32(r2_), (float(q_[0]), float(q_[1])), False) >= 0 for r2_ in ring_):
                    return L_
                continue
            p_ = np.linalg.solve(A_[:, :2], q_ - A_[:, 2])
            if 0 <= p_[0] < pic_.shape[1] and 0 <= p_[1] < pic_.shape[0] and pic_[int(p_[1]), int(p_[0])].max() > 6:
                return L_
        return None

    def auto_align(L_):
        """Совместить эту съёмку с картинкой L_ (точнее) по содержимому: фазовая корреляция там, где они
        перекрываются, с поправкой на текущий сдвиг. -> (dq, отклик) или (None, причина)."""
        o_, pic_, A_, ring_ = L_[:4]
        dq0 = np.asarray(st['al']['dq'], float)
        cs_ = np.array([[0, 0], [pic_.shape[1], 0], [0, pic_.shape[0]], [pic_.shape[1], pic_.shape[0]]], float)
        Q_ = cs_ @ A_[:, :2].T + A_[:, 2] + dq0
        x0_, y0_ = np.maximum(Q_.min(0), 0)
        x1_, y1_ = np.minimum(Q_.max(0), sc.size)
        if x1_ - x0_ < 32 or y1_ - y0_ < 32:
            return None, "the picture is outside this scan"
        k_ = min(1.0, 1400 / max(x1_ - x0_, y1_ - y0_))
        Wr, Hr = int((x1_ - x0_) * k_), int((y1_ - y0_) * k_)
        base_ = sc.render((x0_ + x1_) / 2, (y0_ + y1_) / 2, k_, Wr, Hr)
        Mw_ = np.c_[k_ * A_[:, :2], k_ * (A_[:, 2] + dq0 - (x0_, y0_))].astype(np.float32)
        ov_ = cv2.warpAffine(pic_, Mw_, (Wr, Hr), flags=cv2.INTER_AREA if k_ < 1 else cv2.INTER_LINEAR, borderValue=0)
        msk_ = (cv2.cvtColor(ov_, cv2.COLOR_BGR2GRAY) > 6) & (cv2.cvtColor(base_, cv2.COLOR_BGR2GRAY) > 24)
        if ring_:
            cm_ = np.zeros((Hr, Wr), np.uint8)
            cv2.fillPoly(cm_, [np.int32(np.round((r2_ + dq0 - (x0_, y0_)) * k_)) for r2_ in ring_], 1)
            msk_ &= cm_ > 0
        if msk_.sum() < 0.05 * Wr * Hr:
            return None, "too little overlap"
        w8_ = cv2.GaussianBlur(cv2.erode(msk_.astype(np.float32), np.ones((9, 9), np.uint8)), (0, 0), 6)

        def hp_(im_):
            g_ = cv2.cvtColor(im_, cv2.COLOR_BGR2GRAY).astype(np.float32)
            g_ = g_ - cv2.GaussianBlur(g_, (0, 0), 8)
            g_ = g_ / (np.sqrt((g_ ** 2 * w8_).sum() / max(1.0, w8_.sum())) + 1e-6)
            return g_ * w8_
        (sx_, sy_), resp_ = cv2.phaseCorrelate(hp_(base_), hp_(ov_))
        if resp_ < 0.03:
            return None, f"no clear match (response {resp_:.3f}): drag it by hand"
        return dq0 - np.array([sx_, sy_]) / k_, resp_

    def other_grid_src():
        """Сетка образца построена по крестам ДРУГОЙ съёмки (руками)? -> её имя («Selection 3») или None."""
        if smp is None:
            return None
        try:
            L_ = json.load(open(smp.file, encoding='utf-8')).get('lattice') or {}
        except (OSError, ValueError):
            return None
        sr_ = L_.get('src')
        if L_.get('stage') or not sr_ or not os.path.isdir(sr_) or os.path.abspath(sr_) == os.path.abspath(folder):
            return None
        try:                                                # кресты там ещё стоят
            d_ = json.load(open(os.path.join(sr_, 'grid.json'), encoding='utf-8'))
            if d_.get('seeded') or len(d_.get('marks') or []) < 2:
                return None
        except (OSError, ValueError):
            return None
        return f"Selection {SM.sample_index(sr_)}"

    def al_set(dq_new):
        """Совмещение: сдвиг dq — остальных картинок относительно этой; вид едет так же, чтобы на экране двигалась
        ТОЛЬКО эта съёмка (выбранное выделение), а остальные стояли."""
        d_ = np.asarray(dq_new, float) - np.asarray(st['al']['dq'], float)
        st['al']['dq'] = [float(v_) for v_ in np.asarray(dq_new, float)]
        st['cx'] += float(d_[0])
        st['cy'] += float(d_[1])

    def align_save():
        """Записать сдвиг этой съёмки (align.json): картинки и контуры совпали с выбранной точной съёмкой."""
        from .snap import geometry, align as _align
        G_ = geometry(folder)
        d0_ = np.asarray(G_.get('d_um', (0.0, 0.0)), float)
        d_ = d0_ - G_['ms'] * np.linalg.solve(G_['Mo'], np.asarray(st['al']['dq'], float))
        _align(folder, d_ if np.hypot(*d_) > 1e-4 else (0, 0), to=st['al'].get('to'))
        dd_ = d_ - d0_                                      # выделение этой съёмки едет вместе с её картинкой
        if np.hypot(*dd_) > 1e-4 and smp is not None:
            try:
                from .layers import move as _mv
                from . import project as PJ_
                sh_ = stub()
                me_ = sc.st.meta.get('sample_uid')
                k_ = next((i_ for i_, r_ in enumerate(sh_._regions) if r_['uid'] == me_), None)
                if k_ is not None:
                    _mv(sh_, k_, float(dd_[0]), float(dd_[1]))
                    PJ_.save_regions(sh_)
            except Exception as e_:
                L.debug('VIW-011', f"Selection not moved with its picture: {e_}")
        st['_fl_key'] = None
        st['_so_key'] = st['_sg_key'] = None
        st.pop('_lat', None)
        return d_

    def draw_finer(img):
        """Слои: снимки всех выделений по порядку списка (снизу вверх), выбранная съёмка — поверх всех (с ней
        работают); при совмещении (L) — под ними, чтобы видеть, как они ложатся. У каждого — своя прозрачность (O)."""
        Hh, Ww = img.shape[:2]
        Rm = R() * st['s']                                  # мозаика этой -> окно: mid + s R (M - c)
        T_ = np.c_[Rm, mid() - Rm @ np.array([st['cx'], st['cy']])]
        dq_ = np.asarray(st['al']['dq'], float) if st.get('al') else np.zeros(2)   # совмещение: сдвиг на просмотр
        al_ = alphas()
        me_u_ = sc.st.meta.get('sample_uid')
        lays_ = finer_layers()
        a0_ = al_.get(me_u_, 1.0) if not st.get('al') else 0.6
        if not lays_ and a0_ >= 0.99:                       # других слоёв нет: картинка как есть (быстро)
            return img
        base_ok_ = 255 - cv2.inRange(img, (14, 14, 14), (28, 28, 28))   # где у этой съёмки картинка (фон — нет)
        comp_ = np.full_like(img, 20)

        def put_base(c_):
            a_ = a0_                                        # совмещение: сверху, сквозь неё видно остальные
            if a_ >= 0.99:
                return cv2.copyTo(img, base_ok_, c_)
            elif a_ > 0.01:
                return cv2.copyTo(cv2.addWeighted(img, a_, c_, 1 - a_, 0), base_ok_, c_)
            return c_
        for o_, pic_, A_, ring_, uid_, f_, n_, under_ in reversed(lays_):   # снизу списка вверх
            a_ = al_.get(uid_, 1.0)
            if a_ <= 0.01:                                    # прозрачность 0: только контур
                continue
            M_ = T_[:, :2] @ A_[:, :2]
            t_ = T_[:, :2] @ (A_[:, 2] + dq_) + T_[:, 2]
            corners_ = np.array([[0, 0], [pic_.shape[1], 0], [0, pic_.shape[0]], [pic_.shape[1], pic_.shape[0]]], float)
            cv_ = corners_ @ M_.T + t_
            if cv_[:, 0].max() < 0 or cv_[:, 1].max() < 0 or cv_[:, 0].min() > Ww or cv_[:, 1].min() > Hh:
                continue
            if np.hypot(*M_[:, 0]) * pic_.shape[1] < 12:        # совсем мелко — не видно, не тратим время
                continue
            from .util import warp_clean_roi              # дробный сдвиг: без чёрной каймы; только видимый кусок
            got_ = warp_clean_roi(pic_, np.c_[M_, t_], (Ww, Hh))
            if got_ is None:
                continue
            w_, mb_, (xo_, yo_) = got_
            roi_ = comp_[yo_:yo_ + mb_.shape[0], xo_:xo_ + mb_.shape[1]]
            if a_ < 0.99:                                     # полупрозрачная: видно, что под ней
                w_ = cv2.addWeighted(w_, a_, roi_, 1 - a_, 0)
            np.copyto(roi_, w_, where=mb_[..., None])        # вне кадров (чёрное) — не кладём
        comp_ = put_base(comp_)                             # выбранная — поверх всех (совмещение: полупрозрачно)
        return comp_

    def draw_subgrids(img):
        """Сетка каждого выделения из клеток — его цветом, внутри его клеток: края клеток и подсетка (2-5), если есть;
        клетки кисти в подсетке — зелёные."""
        L_ = lat_now()
        if L_ is None:
            return
        from .snap import stage_to_mosaic
        regs_, rf_ = regions_now()
        Hh, Ww = img.shape[:2]
        for r_ in regs_:
            n_ = int((rf_.get(r_['uid']) or {}).get('n') or r_.get('sub') or 1)
            if n_ <= 1 or not r_.get('cells') or r_.get('hidden') or len(r_['cells']) > 4000:
                continue                                    # без подсетки: края клеток — это сетка, граница — контур
            N_ = LT.level(r_)
            col_ = SM.color(r_['num'] - 1)
            for a, b in r_['cells']:
                for t in range(0, n_ + 1):                  # и края клетки: иначе каждая n-я линия пропадает
                    for p0, p1 in ((((a + t / n_) / N_, b / N_), ((a + t / n_) / N_, (b + 1) / N_)),
                                   ((a / N_, (b + t / n_) / N_), ((a + 1) / N_, (b + t / n_) / N_))):
                        q_ = stage_to_mosaic(folder, LT.to_stage(L_, [p0, p1]))
                        v0, v1 = to_view(q_[0]), to_view(q_[1])
                        if max(v0[0], v1[0]) < 0 or min(v0[0], v1[0]) > Ww or max(v0[1], v1[1]) < 0 or min(v0[1], v1[1]) > Hh:
                            continue
                        cv2.line(img, (int(round(v0[0])), int(round(v0[1]))), (int(round(v1[0])), int(round(v1[1]))),
                                 SM.dim(col_, 0.8), 1, cv2.LINE_8)
            M_ = N_ * n_
            pc_ = (rf_.get(r_['uid']) or {}).get('cells', [])
            if pc_:                                         # выбранные клетки подсетки — зелёной рамкой
                clear_ = [to_view_arr(stage_to_mosaic(folder, LT.cell_stage(L_, a, b, M_))) for a, b in pc_]
                cv2.polylines(img, [np.int32(np.round(q_)) for q_ in clear_], True, (90, 200, 90), 1, cv2.LINE_8)

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
                    UI.panel(img, x_l, y_l - 20, x_l + UI.size("Selection (new)", 0.42)[0] + 10, y_l - 2, 0.8, (235, 235, 235))
                    UI.put(img, "Selection (new)", x_l + 5, y_l - 6, 0.42, (235, 235, 235))

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
            if P0_.scan_uid(folder) == r_.get('uid'):
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
                cv2.fillPoly(mask_, [np.int32(np.round(ext_))], 255, cv2.LINE_8)
            for h_ in holes_:
                for ext_, hs_ in UI.geom_shapes(h_, tv_):
                    cv2.fillPoly(mask_, [np.int32(np.round(ext_))], 0, cv2.LINE_8)
            tint_ = cv2.addWeighted(img, 0.72, np.full_like(img, SM.color(r_['num'] - 1)), 0.28, 0)
            cv2.copyTo(tint_, mask_, img)

    def sample_outlines():
        """Выделения образца на этой мозаике: [(кольцо в пикселях мозаики, цвет, подпись, пунктир, ждёт)]."""
        if smp is None:
            return []
        sp_ = smp.file
        key_ = (_mt(sp_),
                _mt(os.path.join(folder, 'grid.json')))
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
                L_c = lat_now() if r_.get('cells') else None
                if L_c is not None:                         # из клеток — по той же решётке, что и их подсетка
                    gm_ = LT.cells_geom(L_c, [tuple(c) for c in r_['cells']], LT.level(r_))
                elif r_.get('shape'):
                    gm_ = from_json(r_['shape'])
                else:
                    (ax, ay), (bx, by) = r_['A'], r_['B']
                    fw_, fh_ = r_.get('fw', 0), r_.get('fh', 0)
                    gm_ = _box(min(ax, bx) - fw_ / 2, min(ay, by) - fh_ / 2, max(ax, bx) + fw_ / 2, max(ay, by) + fh_ / 2)
                ps_ = [gm_] if gm_.geom_type == 'Polygon' else list(getattr(gm_, 'geoms', []))
                kind_ = 'Sel.'
                for p_ in ps_:
                    q_ = stage_to_mosaic(folder, np.array(p_.exterior.coords)[:-1])
                    out_.append((q_, SM.color(r_['num'] - 1), f"{kind_} {r_['num']} {r_.get('obj', '')}x"
                                 if r_.get('kind') != 'draft' else f"Sel. {r_['num']} draft", r_.get('kind') == 'draft', False,
                                 r_['uid']))
        except Exception:
            out_ = []
        st['_so_key'], st['_so'] = key_, out_
        return out_

    h_last = [g_sig(), g_snap(), d_snap(), d_mt()]

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
        if sc.fresh:                                        # фоном подгрузились кадры, которые видно: перерисовать
            sc.fresh = False
            st['dirty'] = True
            st.pop('_reuse', None)
        if st.get('fast_until') and _t.monotonic() >= st['fast_until']:
            st.pop('fast_until')                            # вид остановился: в полном качестве
            st['dirty'] = True
            st.pop('_reuse', None)
        if st['down'] is None and st.get('lab') is None:
            sg_ = g_sig()
            dm_ = d_mt()
            dch_ = dm_ != h_last[3]
            if dch_ and not st.get('acted'):                # файлы образца поменяло окно микроскопа — не шаг здесь
                h_last[2], h_last[3] = d_snap(), dm_
                dch_ = False
            if sg_ != h_last[0] or dch_:                    # что-то сделано: в историю (разметка и файлы образца)
                new_ = g_snap()
                dn_ = d_snap()
                sel_changed_ = sg_[2] != h_last[0][2]
                if sg_ != h_last[0] or dn_ != h_last[2]:
                    hist.push(g_label(), dict(g=h_last[1], d=h_last[2]), dict(g=new_, d=dn_), folder=folder, px=px_um)
                h_last[:] = [sg_, new_, dn_, dm_]
                if sel_changed_:                            # клетки выбраны — «Selection (new)» в списке; Enter — сделать
                    st['car_dirty'] = True
            st['acted'] = False
        if _t.monotonic() - st.get('_sync_t', 0) > 0.5:  # окно микроскопа поменяло образец или кисть —
            st['_sync_t'] = _t.monotonic()                 # показать то же, что и там
            try:                                           # карту там повернули (\, T, зеркала): вид — так же
                from .util import ui_pref as _uq2
                if '_Vq' in st and _uq2('orient') != st['_Vq']:
                    st['_V'] = None
                    st['_layers'] = st['_top'] = None
                    st['dirty'] = True
            except Exception:
                pass
            lsig_ = (_mt(os.path.join(folder, 'index.csv')), _mt(os.path.join(folder, 'join', 'live.json')),
                     _mt(os.path.join(folder, 'join', 'stitch.json')))   # кадры, живая сшивка, сшивка (не правка J)
            if st.get('_lsig') is None:
                st['_lsig'] = lsig_
            elif lsig_ != st['_lsig'] and st['down'] is None and \
                    (_t.monotonic() - st.get('_lsig_t', 0) > 2.0 or st['mode'] == 'cross'):
                # новая сшивка — сразу (и в режиме крестов: размечать надо по актуальной); кресты едут с кадрами
                st['_lsig'], st['_lsig_t'] = lsig_, _t.monotonic()   # скан идёт (кадры шьются на ходу) или пересшили:
                                                           # не чаще раза в 2 с (слабый ПК, 1000 кадров)
                try:                                       # картинка — заново на месте, вид и кресты — с ней
                    sc_n = _Scene(folder)
                    org_o = sc.org
                    d_o = org_o - sc_n.org                 # мозаика начинается с другого кадра: всё — на d_o
                    if g.cr.marks:
                        g.cr.marks = [np.asarray(q_, float) + d_o for q_ in g.cr.marks]
                        g.cr._fit()
                    if g.frames_px is not None:
                        fp_ = np.asarray(g.frames_px, float) + d_o
                        if len(fp_) < len(sc_n.P):
                            fp_ = np.vstack([fp_, sc_n.P[len(fp_):]])
                        g.frames_px = fp_
                    if g.cr.marks and g.frames_px is not None and len(g.frames_px) == len(sc_n.P):
                        g.sync(sc_n.P, sc_n.W, sc_n.H)
                        g.save(px_um)
                    else:
                        g.frames_px = sc_n.P.copy()
                    try:
                        sc.close()
                    except Exception:
                        pass
                    sc = sc_n
                    st['cx'] += float(d_o[0])
                    st['cy'] += float(d_o[1])
                    for k_ in ('_fl_key', '_so_key', '_sg_key', '_outline_px', 'inner', '_layers'):
                        st[k_] = None                      # вид (поворот, отражение) — тот же: не прыгает
                    st.pop('_lat', None)
                    st['dirty'] = st['car_dirty'] = True
                    if not st.get('msg'):
                        st['msg'] = f"{len(sc.P)} frames so far (stitched as they come)"
                except Exception as e_:
                    L.debug('VIW-020', f"live refresh: {type(e_).__name__}: {e_}")
            if smp is not None:                            # очередь окна микроскопа идёт: строка внизу — живая
                qsp_ = os.path.join(smp.folder, '.queue_status.json')
                qmt_ = _mt(qsp_)
                if qmt_ != st.get('_qmt') or (qmt_ and _t.time() - qmt_ < 6 and _t.monotonic() - st.get('_qdraw', 0) > 1):
                    st['_qmt'], st['_qdraw'] = qmt_, _t.monotonic()
                    st['dirty'] = True
            mt_ = tuple(_mt(p_) for p_ in
                        ((smp.file if smp is not None else ''), os.path.join(folder, 'grid.json')))
            if st.get('_mt') is None:
                st['_mt'] = mt_
            elif mt_ != st['_mt'] and st['down'] is None:
                st['_mt'] = mt_
                subs2 = SM.session_of(folder)[1]
                if subs2 != subs:                            # выделение удалили / вернули / сняли новое
                    subs[:] = subs2
                    st['car_dirty'] = st['dirty'] = True
                    if smp is not None and not any(P0_.scan_uid(folder) == r_.get('uid') for r_ in json.load(
                            open(smp.file, encoding='utf-8')).get('selections', [])):
                        st['msg'] = ("This selection was deleted in the microscope window "
                                     "(Ctrl+Z there brings it back)")
                try:
                    g2_ = Grid(folder)
                    if g2_.selected != g.selected:           # кисть поменяли там (удалили черновик и т. п.)
                        g.selected = set(g2_.selected)
                        h_last[:2] = [g_sig(), g_snap()]     # это не своё действие: не в историю
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
                if st['s'] is not None:                     # окно тянут: видно то же место, крупнее / мельче
                    st['s'] *= min((W_ - SM.CAR_W) / max(1, st['Wv']), H_ / max(1, st['Hv']))
                st['Wv'], st['Hv'], st['dirty'] = W_ - SM.CAR_W, H_, True
                st['car_dirty'] = True
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
            st['timing'].mark('show')
            st['timing'].end()
            st['dirty'] = st['car_dirty'] = False
            if not st.get('icon'):                          # окно уже на экране: значок программы (Windows)
                st['icon'] = True
                set_icon(WIN)
        elif (st.get('car_dirty') or st.get('prof_dirty')) and st.get('scene') is not None:   # карусель, профили
            show(np.hstack([with_prof(st['scene']), carousel_img()]))
            st['car_dirty'] = st['prof_dirty'] = False
        key = next(keys, EXIT) if keys is not None else cv2.waitKeyEx(8 if st['down'] else 20)
        if key != -1:
            st['acted'] = True
        if key != -1:
            st.pop('_reuse', None)                         # клавиша могла поменять картинку: слои — заново
        if keys is not None:
            cv2.waitKey(1)
            if isinstance(key, tuple):                      # тест: ('click'|'rclick', x, y[, shift]) или ('drag', ...)
                kind, *a = key
                fl = cv2.EVENT_FLAG_SHIFTKEY if (len(a) > 2 and a[-1] == 'shift') else \
                    (cv2.EVENT_FLAG_CTRLKEY if (len(a) > 2 and a[-1] == 'ctrl') else 0)
                if a and a[-1] == 'ctrl':
                    a = a[:-1]
                if kind == 'call':                          # тест: сделать что-то «из другого окна» (или посмотреть st)
                    a[0](st) if getattr(a[0], '__code__', None) is not None and a[0].__code__.co_argcount else a[0]()
                    key = -1
                    kind = ''
                elif kind == 'sleep':
                    _t.sleep(a[0])
                    key = -1
                    kind = ''
                if kind == 'qwheel':                        # ('qwheel', x, y, +1) — колесо в точке мозаики
                    a = [int(round(v_)) for v_ in to_view(a[:2])] + [a[2]]
                    kind = 'wheel'
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
                elif kind == 'wheel':                       # ('wheel', x, y, +1 | -1)
                    mouse(cv2.EVENT_MOUSEWHEEL, a[0], a[1], (120 if a[2] > 0 else -120) << 16)
                elif kind == 'ldown':                       # тянуть по шагам: нажать, вести, отпустить
                    mouse(cv2.EVENT_LBUTTONDOWN, a[0], a[1], fl)
                elif kind == 'lmove':
                    mouse(cv2.EVENT_MOUSEMOVE, a[0], a[1], fl | cv2.EVENT_FLAG_LBUTTON)
                elif kind == 'lup':
                    mouse(cv2.EVENT_LBUTTONUP, a[0], a[1], fl)
                elif kind == 'drag':
                    mouse(cv2.EVENT_LBUTTONDOWN, a[0], a[1], fl)
                    mouse(cv2.EVENT_MOUSEMOVE, a[2], a[3], fl | cv2.EVENT_FLAG_LBUTTON)
                    mouse(cv2.EVENT_LBUTTONUP, a[2], a[3], fl)
                key = -1
        if key in PROF_KEYS:                                # ` / F12: сколько мс на что уходит (и в журнал)
            st['msg'] = st['timing'].toggle()
            st['dirty'] = True
            continue
        if st.get('pick') and key != -1:                  # список съёмок открыт: Esc — закрыть, прочее — мимо
            if key == 27:
                st.pop('pick', None)
                st['dirty'] = True
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
                h_last[:] = [g_sig(), g_snap(), d_snap(), d_mt()]
                st['msg'] = (f"Undone: {e_['label']}  (Ctrl+Y: redo, {len(hist)} more)" if ud_ == 'undo' else
                             f"Redone: {e_['label']}")
            st['dirty'] = True
            if st['switch'] is not None:
                break
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
        if key in (3014656, 65535, 0x2E0000) and st['mode'] == 'cross':   # Del в режиме крестов: заново
            raw = ch = 'C'
        if g.ready():                                       # последняя готовая сетка: пересозданная получит её индексы
            v1_, v2_ = g.cr.vectors()
            st['regrid'] = (np.array(g.cr.O, float), np.array(v1_, float), np.array(v2_, float))
        step = 0.25 * min(st['Wv'], st['Hv']) / st['s']    # стрелки — четверть окна
        ch = ch.lower()
        # единые клавиши (одна буква — один смысл в обоих окнах): новая клавиша -> внутреннее действие
        if raw and st['mode'] == 'align':
            raw = 'l' if raw in 'aA' else raw
            ch = raw.lower()
        elif raw and not st.get('lab'):
            if key == 1:                                    # Ctrl+A: все клетки (ещё раз — ни одной)
                if st['mode'] in ('select', 'paint') and g.ready():
                    W_a, H_a = sc.size
                    all_ = inside_only(g.cells_in([(0, 0), (W_a, 0), (W_a, H_a), (0, H_a)], sc.size))[0]
                    raw = 'x' if g.selected and set(all_) <= g.selected else 'a'
                else:
                    raw = ''
            elif raw in MICRO_ONLY:
                st['msg'] = f"{raw.upper() if raw != ' ' else 'Space'}: only in the microscope window ({MICRO_ONLY[raw]})"
                st['dirty'] = True
                continue
            else:
                raw = VIEW_KEYS.get(raw, raw)
            if key in (3014656, 65535, 0x2E0000) and st['mode'] == 'cross':   # Del в режиме крестов: заново
                raw = 'C'
            ch = raw.lower()
        if st['mode'] == 'align':                           # L: совместить эту съёмку с более точной
            a_ = _arrow(key)
            u_ = lambda: (f"{st['al']['name']}: " if st['al'].get('name') else "")
            if a_ is not None:                              # стрелки: 1 точка экрана (Shift: 10)
                k_ = 10 if shift_held(key) else 1
                al_set(np.asarray(st['al']['dq'], float) - R().T @ (np.array(a_, float) * k_ / st['s']))
            elif key in (13, 10) or ch == 'l':
                d_ = align_save()
                st['mode'], st['al'] = 'look', None
                st['msg'] = (f"This scan is shifted on the stage by {d_[0]:+.2f}, {d_[1]:+.2f} um (align.json): "
                             "the selections, the map and the export go with it. A again: change it")
                smpA_ = P0_.sample_obj(folder) if smp0 is not None else None
                if smpA_ is not None:                       # сетка образца — одна, по крестам
                    update_lattice_of(smpA_)
                    LA_ = smpA_.data.get('lattice')
                    if st.get('seeded') and LA_ and not LA_.get('stage'):
                        try:                                # кресты не здесь: сетка стоит, картинка уехала под ней
                            g, _m = _seed(folder, Grid(folder), LA_, smpA_, sc, px_um, snap=False)
                            g = _seed_mark(folder, LA_)
                            st['seeded'] = True
                            st['msg'] += ".  The grid of the sample stays put: the picture moved under it"
                        except Exception as e_:
                            L.debug('VIW-012', f"Grid not re-seeded after align: {e_}")
                    elif g.ready():                         # кресты этой съёмки задают сетку: она уехала с ними
                        st['msg'] += (".  These crosses make the grid of the sample: it moved with them "
                                      "(other scans take it when opened)")
                st['_sg_key'] = None
                st.pop('_lat', None)
            elif key == 8:                                  # Backspace: как по столику (сдвиг 0)
                from .snap import geometry
                G_ = geometry(folder)
                al_set(G_['Mo'] @ (np.asarray(G_.get('d_um', (0, 0)), float) / G_['ms']))
                st['msg'] = "Back to the stage coordinates (no shift): Enter saves"
            elif key == 32:                                 # пробел: само, по содержимому
                L_ = layer_at(st['hov']) if st.get('hov') else None
                L_ = L_ or (st['al'].get('L') if st['al'].get('L') else (finer_layers() or [None])[0])
                if L_ is None:
                    st['msg'] = "No picture of a higher objective here to align with"
                else:
                    dq_, r_ = auto_align(L_)
                    if dq_ is None:
                        st['msg'] = f"Not aligned: {r_}"
                    else:
                        al_set(dq_)
                        st['al']['to'] = L_[5]
                        st['msg'] = (f"Aligned with Selection {L_[6]} ({L_[0]}x) by the picture (match {r_:.2f}): "
                                     "check it, Enter saves, Esc cancels")
            elif key == 27:
                al_set((0.0, 0.0))                          # картинка — на прежнее место
                st['mode'], st['al'] = 'look', None
                st['msg'] = "Alignment cancelled"
            st['dirty'] = True
            continue
        if st['mode'] == 'poly' and st.get('poly') is not None and key in (13, 10, 8):   # многоугольник: Enter / Backspace
            if key == 8:
                st['poly']['v'] = st['poly']['v'][:-1]
                st['msg'] = f"{len(st['poly']['v'])} points"
            else:
                st['msg'] = poly_finish()
            st['dirty'] = True
            continue
        if key in (13, 10) and st['mode'] in ('look', 'select', 'paint'):   # Enter: выбранные клетки -> выделение
            st['msg'] = brush_commit()
            st['dirty'] = True
            continue
        if key == 27 and st['mode'] == 'look' and car.get('marks') and not st.get('help'):
            car['marks'] = set()
            st['msg'] = "Unmarked"
            st['car_dirty'] = st['dirty'] = True
            continue
        if key == 27:
            if st.get('help'):
                st['help'] = False
            elif st['mode'] != 'look':
                if st['mode'] == 'poly':
                    st['msg'] = "Outline cancelled"
                st['mode'], st['ruler'], st['poly'] = 'look', [], None
            else:                                           # просмотр закрывается просто Esc (окно микроскопа — Shift+Esc)
                st['switch'] = False
                break
            st['dirty'] = True
        elif raw in (',', '.') and len(subs) > 1:           # предыдущий / следующий образец
            i_ = subs.index(folder) if folder in subs else -1
            st['switch'] = subs[(i_ + (1 if raw == '.' else -1)) % len(subs)]
        elif raw == 'C':                                     # Shift+C: кресты заново (Ctrl+Z вернёт)
            unseed(); g.cr.clear(); g.refined = None; set_straight(False); g.save(px_um)
            if smp0 is not None:                            # это были кресты сетки образца — её больше нет
                update_lattice_of(P0_.sample_obj(folder))
            st['mode'] = 'cross'
            st['msg'] = ("Crosses cleared (Ctrl+Z brings them back); the grid of the sample is gone until new ones "
                         "are marked. Click a cross, then its neighbour in the same row")
        elif ch == 's':                                      # Shift+J: точная сшивка (quality) — в очередь, как в окне микроскопа
            st['msg'] = queue_quality()
        elif ch == 'u':                                      # U: выделение — прямоугольный многоугольник по точкам
            if st['mode'] == 'poly':
                st['mode'], st['poly'] = 'look', None
                st['msg'] = "Outline off"
            else:
                st['mode'], st['poly'] = 'poly', {'v': [], 'R': R().copy()}
                st['msg'] = ("Outline: click the corners - every side goes only across or up / down and snaps to the "
                             "crosses and to the points already put (grey lines). Click the first point or Enter: "
                             "close (the last corner is added by itself)  Backspace: back  Esc: cancel")
        elif ch == 'r':                                      # R: линейка: точка-точка -> прямая-прямая -> точка-прямая -> выкл
            nxt_ = {'pp': 'll', 'll': 'pl', 'pl': None}
            if st['mode'] == 'ruler' and nxt_[st.get('rk', 'pp')] is None:
                st['mode'], st['ruler'], st['rk'] = 'look', [], 'pp'
                st['msg'] = "Ruler off"
            else:
                st['rk'] = nxt_[st.get('rk', 'pp')] if st['mode'] == 'ruler' else 'pp'
                st['mode'], st['ruler'] = 'ruler', []
                st['msg'] = {'pp': "Ruler, point - point: click two points",
                             'll': "Ruler, line - line: two points of the first line, two of the second (angle and distance)",
                             'pl': "Ruler, point - line: two points of the line, then the point"}[st['rk']] + \
                    "  (R: next kind, Esc: off)"
        elif ch == 'l':                                      # L: совместить эту съёмку с более точной (крупнее)
            fl_ = finer_layers()
            if not fl_:
                st['msg'] = "No other stitched selection here: nothing to align this one with"
            else:
                L_ = layer_at(st['hov']) if st.get('hov') else None
                st['mode'] = 'align'
                st['al'] = dict(dq=[0.0, 0.0], dq0=[0.0, 0.0], L=L_, to=L_[5] if L_ else None)
                st['msg'] = ("Align this selection (on top, half transparent): drag it (or arrows, Shift: x10) until it "
                             "sits on the others - they stay; Space: by itself, by the picture under the mouse; Enter: "
                             "save (it moves on the stage with its outline), Backspace: no shift, Esc: cancel")
        elif ch == 'o' and car.get('marks'):                 # O при отметках: всем отмеченным разом
            lv_ = [1.0, 0.75, 0.5, 0.25, 0.0]
            mk_ = sorted(car['marks'])
            a0_ = alphas().get(mk_[0], 1.0)
            i_ = min(range(len(lv_)), key=lambda i2_: abs(lv_[i2_] - a0_))
            a1_ = lv_[(i_ + (-1 if raw == 'O' else 1)) % len(lv_)]
            alphas({u_: a1_ for u_ in mk_})
            st['msg'] = f"{len(mk_)} marked: pictures {100 * a1_:.0f} %" + (" (outline only)" if a1_ == 0 else "")
        elif ch == 'g':                                      # G: объектив отмеченных / под мышью; Shift+G — для новых
            from .config import OBJECTIVES
            cyc_ = ['draft'] + sorted(OBJECTIVES)
            if raw == 'G':
                cur_ = car.get('new_obj', 'draft')
                car['new_obj'] = cyc_[(cyc_.index(cur_) + 1) % len(cyc_) if cur_ in cyc_ else 0]
                st['msg'] = (f"New selections from cells (M): " + ("draft (no objective)" if car['new_obj'] == 'draft'
                                                                     else f"{car['new_obj']}x") + "   Shift+O: next")
            else:
                us_ = targets()
                if not us_:
                    st['msg'] = "O: point at a selection (or Ctrl+click several) to change its objective"
                else:
                    r0_ = next(r_ for r_ in regions_now()[0] if r_['uid'] == us_[0])
                    cur_ = 'draft' if r0_.get('kind') == 'draft' else r0_.get('obj')
                    nxt_ = cyc_[(cyc_.index(cur_) + 1) % len(cyc_)] if cur_ in cyc_ else cyc_[1]
                    st['msg'] = set_objective(us_, nxt_) + "   O: next objective"
        elif ch == 'q':                                      # Q: съёмки отмеченных / под мышью — в очередь
            us_ = targets()
            st['msg'] = queue_scans(us_) if us_ else "S: point at a selection with an objective (or Ctrl+click several)"
        elif key in (3014656, 65535, 0x2E0000):              # Delete: отмеченные / под мышью, не снятые
            us_ = targets()
            regs_ = {r_['uid']: r_ for r_ in regions_now()[0]}
            done_, sk_ = [], []
            for u_ in us_:
                r_ = regs_.get(u_)
                if r_ is None:
                    continue
                if scanned(r_):
                    sk_.append(r_['num'])
                    continue
                st['msg'] = delete_draft(u_, subs.index(folder))
                done_.append(r_['num'])
            car['marks'] = set(car.get('marks', set())) - set(us_)
            st['msg'] = ((f"Deleted Selection{'s' if len(done_) > 1 else ''} {', '.join(map(str, done_))} (Ctrl+Z: the squares back)"
                          if done_ else "Nothing deleted") +
                         (f";  scanned ones stay ({', '.join(map(str, sk_))}): x on their cards removes them" if sk_ else ""))
            st['car_dirty'] = True
            st['_so_key'] = st['_sg_key'] = None
        elif ch == 'y':                                      # Y: картинки других выделений вместе с этой — вкл / выкл
            car['others'] = not car.get('others', True)
            st['_fl_key'] = None
            st['msg'] = ("Pictures of the other selections: on ([ ] - transparency)" if car['others']
                         else "Only this scan and the higher objectives over it")
        elif ch == 'o':                                      # [ ]: прозрачность картинки ВЫБРАННОГО выделения
            fo_u_ = car.get('focus')
            L_ = next((x_ for x_ in finer_layers() if x_[4] == fo_u_), None) if fo_u_ else None
            if L_ is None:                                  # выбрана открытая съёмка
                me_u_ = sc.st.meta.get('sample_uid')
                L_ = (sc.st.meta.get('objective') or 0, None, None, None, me_u_, folder, SM.sample_index(folder))
            if L_ is None:
                st['msg'] = "[ ]: choose a selection (its card) or Ctrl+click several"
            else:
                lv_ = [1.0, 0.75, 0.5, 0.25, 0.0]
                a0_ = alphas().get(L_[4], 1.0)
                i_ = min(range(len(lv_)), key=lambda i2_: abs(lv_[i2_] - a0_))
                a1_ = lv_[(i_ + (-1 if raw == 'O' else 1)) % len(lv_)]
                alphas({L_[4]: a1_})
                st['msg'] = (f"Selection {L_[6]} ({L_[0]}x): picture {100 * a1_:.0f} %" +
                             (" (outline only)" if a1_ == 0 else "") + "   [: more transparent, ]: less")
        elif ch == 'c':                                      # C: отмечать кресты (ещё раз C — выйти)
            if st['mode'] == 'cross' and not g.cr.marks:     # крестов ещё нет (только что Shift+C) — остаться
                st['msg'] = "Click a cross, then its neighbour in the same row (C again — when done)"
            elif st['mode'] == 'cross':
                st['mode'] = 'look'
                st['msg'] = ""
            else:
                src_ = other_grid_src()
                if src_:                                    # сетка образца одна: по крестам другой съёмки — сначала их убрать
                    st['msg'] = (f"The grid of the sample is already made from the crosses of {src_}: there is only one. "
                                 f"To make it here, remove those crosses first (open {src_}, C, Del)")
                else:
                    st['mode'] = 'cross'
                    st['msg'] = ("Click a cross, then its neighbour in the same row: the picture turns level. "
                                 "Del: start again")
        elif ch in 'mb' and ch:                              # M (и B по привычке): выбирать клетки
            L_us_ = smp0.data.get('lattice') if smp0 is not None else None
            if not g.ready() and not g.cr.marks and (L_us_ is None or L_us_.get('stage')):
                st['msg'] = "No grid yet: mark the crosses first (C, then two neighbouring crosses), then M"
                st['dirty'] = True
                continue
            if not g.ready() and not g.cr.marks:            # кресты отмечены руками в другой съёмке образца — по ним
                try:
                    g, m_ = _seed(folder, g, L_us_, smp0, sc, px_um, snap=False)
                    g = _seed_mark(folder, L_us_)
                    st['seeded'] = True
                    st['msg'] = "No crosses marked here: the grid comes from the crosses marked in another scan (C: mark your own)"
                except Exception as e_:
                    st['msg'] = f"No grid: {e_}"
            st['mode'] = 'select' if g.ready() else st['mode']
            st['msg'] = (st['msg'] + "   " if 'No crosses here' in (st['msg'] or '') else "") + (
                "Select: click = square (again = off), drag = box (Shift = unselect); right click = remove "
                "one, Shift + right click = the whole area; they become selections (draft)") if g.ready() \
                else "Mark at least 2 crosses first (C)"
        elif ch == 'n':
            st['mode'] = 'label' if g.ready() else st['mode']
            st['msg'] = ("Click a square with a number written on it and type the number X Y. Two squares in another "
                         "row and column fix the directions; right click = remove a label") if g.ready() \
                else "Mark at least 2 crosses first (C)"
        elif key == 8 and st['mode'] == 'stitch':            # правка сшивки — назад, как сшила программа
            ed_ = st['edit']
            sc.P[:] = ed_['orig']
            ed_['moved'] = False
            se_ = os.path.join(folder, 'join', 'seams_edit.json')
            if os.path.exists(se_) and os.path.exists(os.path.join(folder, 'join', 'positions_auto.csv')):
                os.remove(se_)
            seams_load()
            bk_ = os.path.join(folder, 'join', 'positions_auto.csv')
            if os.path.exists(bk_):                         # и сохранённые раньше правки — тоже
                import shutil as _sh
                _sh.copy2(bk_, os.path.join(folder, 'join', 'positions_all.csv'))
                os.remove(bk_)
                st['msg'] = "Back to the automatic stitch"
                st['switch'] = folder
                break
            st['msg'] = "Stitch edits undone"
        elif st['mode'] == 'stitch' and _arrow(key):        # J: ряд / столбец — на 1 px (Shift: 10)
            dx_, dy_ = _arrow(key)
            k_ = 10 if shift_held(key) else 1
            stitch_move(dx_ * k_, dy_ * k_)
        elif ch == 'j':                                     # J: править сшивку руками
            if st['mode'] == 'stitch':
                st['msg'] = stitch_save()
                st['mode'] = 'look'
                if st.pop('restart', None):
                    st['switch'] = folder
                    break
            elif __import__('microcontrol.quality', fromlist=['x']).stitched_as(folder) == 'quick':
                st['msg'] = queue_quality()                 # quick — обзор: сначала точная сшивка (очередь)
            else:
                st['mode'] = 'stitch'
                st['edit'] = dict(orig=sc.P.copy(), ids=[], sel=None, moved=False)
                seams_load()
                st['msg'] = ("Edit the stitch: dots are seams (green good, yellow so-so, red bad, grey off) - click one "
                             "to switch it off or on; click a frame - its row (Shift+click: its column), arrows move it "
                             "by 1 px (Shift: 10); Backspace: back to the automatic stitch; J: save")
        elif ch == 'k':                                     # K: ровные края (без ступенек) / вся мозаика
            st['crop'] = not st.get('crop', True)
            st['inner'] = None
            st['msg'] = "Straight edges: the steps at the borders are cut off" if st['crop'] else "Whole mosaic, with its steps"
        elif key == 8:
            if st['mode'] == 'cross':
                unseed(); g.cr.undo(); g.save(px_um); st['msg'] = "Last cross mark removed"
        elif ch == 'f':                                     # F: кадры — на кресты (второй шаг сшивки)
            if not g.ready():
                st['msg'] = "Mark 2-3 crosses first (C), then F puts every frame onto them"
            elif not os.path.exists(os.path.join(folder, 'join', 'positions_all.csv')):
                st['msg'] = "Stitch first (S): the preview, then F puts the frames onto the crosses"
            else:
                fit_here()
                break
        elif ch == 'p':                                      # P: окошко профилей креста под мышью
            st['prof'] = not st.get('prof')
            st['prof_pin'] = st['prof_last'] = st['prof_rect'] = st['pl_drag'] = st['prof_drag'] = None
            st['prof_pts'] = []
            pl_save()
            st['msg'] = ("Profiles: click 4 points - the two ends of line A, then of line B; then the plots. P: off"
                         if st['prof'] else "Profiles off")
            st['dirty'] = True
        elif ch == 'h':
            st['help'] = not st.get('help')
        elif ch == 't':                                      # T: ровно (наклон снят) / как снято
            if not g.ready():
                st['msg'] = "The tilt is taken from the crosses: mark two neighbouring crosses in a row (C), then T"
            else:
                set_straight(not st['straight'])
                st['msg'] = (f"Level: the picture is turned by {st['rot']:+.2f} deg, the squares stand straight (T: as shot)"
                             if st['straight'] else "Picture as shot, with its tilt (T: level)")
        elif ch == 'a' and st['mode'] in ('select', 'paint') and g.ready():
            W_, H_ = sc.size
            g.selected |= inside_only(g.cells_in([(0, 0), (W_, 0), (W_, H_), (0, H_)], sc.size))[0]
            g.save(px_um); st['msg'] = f"All {len(g.selected)} squares selected"
        elif ch == 'x' and st['mode'] in ('select', 'paint'):
            g.selected.clear(); g.save(px_um); st['msg'] = "Selection cleared"
        elif ch == 'e' and smp is not None:                  # экспорт образца: всё с галочкой -> <образец>/export/
            g.save(px_um)
            from .drawing import export_sample
            sh_ = stub()
            from .samples import init_areas as _ia, order as _ord
            _ia(sh_)
            ks_ = [k_ for k_ in _ord(sh_) if sh_._regions[k_].get('export', False)]
            r, e = busy("Export", lambda: export_sample(sh_, ks_))
            from .drawing import export_message
            st['msg'] = (str(e) if str(e).startswith("Not exported") else f"Export: {e}") if e is not None \
                else export_message(r, len(ks_))
        elif ch == 'e':                                      # без образца: картинки квадратов + чертёж этой съёмки
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
    if st['mode'] == 'stitch' and (st.get('edit') or {}).get('moved'):   # правку сшивки не потерять
        stitch_save()
    if os.path.isdir(folder) and not st.get('stitched'):   # после сшивки разметку уже записала она (кресты — за кадрами)
        g.save(px_um)
    if os.path.isdir(folder):
        sc.close()
    return st['switch'] or None
