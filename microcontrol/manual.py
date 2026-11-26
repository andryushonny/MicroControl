"""Ручной режим: живая картинка на весь экран, движение по кадрам, отметка зоны (как «приват» в Minecraft).

Координаты знать не нужно: доезжаете до края нужного участка и отмечаете угол.
Мини-карта в углу показывает отмеченную зону и где вы сейчас.

  Стрелки             На шаг X / Y в сторону (буквы — для действий)
  E / Q (PgUp/PgDn)   Столик вверх / вниз на шаг Z (Shift — x5)
  F                   Автофокус;  K — точка фокуса (Z здесь)
  1 / 2               Углы выделения;  Del — удалить выбранное
  O                   Объектив (следующий)
  S, Scan area        Поставить в очередь съёмку выбранного (Ctrl+щелчок — несколько); внизу — очередь, x — стоп
  J / Shift+J         Сверка с мозаикой / точная сшивка выбранного
  X                   Экспорт отмеченного в <образец>/export/
  V                   Открыть последний скан в просмотре, после — снова сюда
  L / I / W           Свет / лазер / баланс белого
  R                   Линейка: два щелчка — расстояние
  [ ]                 Прозрачность картинки выбранного на карте
  Пробел              Снимок в photos/
  0                   Вся картинка
  Settings            Prepare, дисторсия, ноль здесь, шаг = кадр, направления, стрелки, поворот карты
  Ctrl+Z / Ctrl+Y     Отменить / вернуть
  H                   Подсказка
  Shift+Esc           Выход
"""
import math
import time
import os
import copy
import json
from .text import cv2                                     # cv2 со шрифтом JetBrains Mono
import numpy as np
from . import log as L
from .errors import MicroControlError
from .util import wheel_delta, wheel_gesture, open_window, window_size, exit_key, EXIT, set_icon
from .undo import History, undo_key
from . import focus as F_
import threading
from .scan import Area, focus_z
from .config import scan_overlap, UI_FPS
from . import samples as SM
from . import timing as TM
from . import ui as UI

WIN = "MicroControl"
HELP = ["Same keys: one letter means one thing here and in the view (M C N P G A Y - only in the view: they say so here)",
        "Shift+Esc: Exit (or close the window)   H: these keys   Esc: leave the mode   Ctrl+Z / Ctrl+Y: undo / redo   Ctrl+P: Settings panel (left)",
        "Arrows: Move by Step X / Step Y   E / Q (PgUp / PgDn): Stage up / down by Step Z (Shift: x5)",
        "Click to edit steps (X / Y: 5 um increments, Z: 1 um), exposure or objective. Enter to save",
        "O: Next objective (4x, 10x, 20x, 50x, 100x) - put it in place",
        "1 / 2: Corners of a selection, one after another (their Z is kept as focus points)",
        "F: Autofocus   K: Keep Z here as a focus point of this selection",
        "Selections (right; the list order = the layers, top on top): click = choose, Go = drive there, x = delete;"
        "  , / . = previous / next;  Enter = drive;  Del = delete the chosen one (Ctrl+Z brings it back)",
        "S / Scan area: Queue the scan of the chosen selection (Ctrl+click several cards: all of them); each is quick stitched in the background",
        "J: check the live picture against the stitched mosaic   Shift+J / Stitch: exact (quality) stitch of the chosen selection",
        "X / Export: everything ticked on the cards -> <sample>/export/ (all ticked must be quality; pictures, mixed.png, the drawing - when the crosses are marked)",
        "... on a card: all scans of the selection; the star = the main one (maps, view, export)",
        "Queue (bar at the bottom, menu Queue): scans one after another, stitches on other cores; x stops a job, Stop all; the window lives",
        "V: Open the last scan in the view (crosses, squares, export); closing it brings you back here",
        "L: Light on / off   I: Laser on / off   W: White balance (aim at an empty place)   Space: Picture",
        "R: Ruler: click two points (distance, um)   T: Turn the camera picture by 180 degrees (remembered)",
        "[ / ]: the picture of the chosen selection on the map more / less transparent (100 ... 0 % = outline)",
        "Wheel: zoom   Drag (middle button too): move the zoomed picture   0: whole frame",
        "Map (top right): click - the stage goes there, drag - a new area on this objective, wheel - zoom, right drag - move",
        "Ctrl+drag on the map: a box (e.g. around a crystal), not snapped to the squares: edges on motor steps, with a"
        " margin for the backlash and the stage error; Scan area shoots it with the objective in place",
        "Tab: big map / big camera   Double-click a shot area on the map: open it (crosses, brush, export)",
        "Settings panel: Z follows the focus points, One-side approach, Prepare, Distortion, Zero here, Steps = frame,"
        " Check directions, Arrows X / Y the other way, Map 90 deg",
        "Cards: Ctrl+click to mark, then Merge / Minus / Overlap above the list; drag a card up / down: the layer order;"
        " the box: in the export; the objective chip: 4x ... 100x, draft; the eye: hide on the maps",
        "Shift+drag over a selection: move it",
        "D: Distortion   B: One-side approach   \\: Map 90 deg (also buttons of the Settings panel)   Esc during a long job: Stop",
        "Grid: one for the whole sample - from the crosses marked last in the view (C), on any selection",
        "` (or F12): timing - ms per frame for each part of the window (also in the log every 5 s): what is slow on this PC",
        ]
LEFT, UP, RIGHT, DOWN = {2424832, 65361}, {2490368, 65362}, {2555904, 65363}, {2621440, 65364}
INT = 0x110000                       # внутреннее дело (кнопка панели Settings): INT + ord(буква), мимо единых клавиш
DEL_KEYS = {3014656, 65535, 0x2E0000}
# одна буква — один смысл в обоих окнах (просмотр — viewer.VIEW_KEYS): клавиша -> внутреннее дело окна микроскопа
MAN_KEYS = {'o': 'm', 'O': 'm', '0': 'z'}
VIEW_ONLY = {'m': "select cells", 'M': "select cells", 'c': "crosses", 'C': "crosses", 'n': "square numbers",
             'N': "square numbers", 'p': "profiles", 'P': "profiles", 'g': "straight edges", 'G': "straight edges",
             'a': "align the scans", 'A': "align the scans", 'y': "the other pictures", 'Y': "the other pictures"}
SETTINGS_ONLY = {'d': "Distortion", 'D': "Distortion", 'b': "One-side approach", 'B': "One-side approach",
                 '\\': "Map 90 deg"}
PGUP, PGDN = {2162688, 65365}, {2228224, 65366}


def screen_size():
    """Размер экрана, px (Windows — через user32, иначе tkinter)."""
    try:
        import ctypes
        u = ctypes.windll.user32
        u.SetProcessDPIAware()
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        pass
    try:
        import tkinter
        r = tkinter.Tk(); r.withdraw()
        s = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return s
    except Exception:
        return 1280, 960


def fit(img, W, H):
    """Вписать кадр в W×H с чёрными полями."""
    k = min(W / img.shape[1], H / img.shape[0])
    w, h = int(img.shape[1] * k), int(img.shape[0] * k)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    out = np.full((H, W, 3), 24, np.uint8)                 # поля — тёмно-серые, как фон окна
    out[(H - h) // 2:(H - h) // 2 + h, (W - w) // 2:(W - w) // 2 + w] = small
    return out


def text(img, s, org, color=(235, 235, 235), scale=0.6):
    """Надпись на тёмной подложке."""
    (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    img[max(0, y - h - 6):y + 6, max(0, x - 6):x + w + 6] //= 3
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


RED = (40, 40, 255)
DEL = {3014656, 65535}


def dashed(img, p0, p1, color, rect=None, dash=7, gap=5, solid=False):
    """Линия (пунктир или сплошная), обрезанная прямоугольником rect = (x, y, w, h)."""
    lim = lambda q: tuple(int(max(-10 ** 6, min(10 ** 6, c))) for c in q)
    ok, a, b = cv2.clipLine(rect or (0, 0, img.shape[1], img.shape[0]), lim(p0), lim(p1))
    if not ok:
        return
    if True:                                        # всегда сплошная (пунктир убран)
        cv2.line(img, a, b, color, 1, cv2.LINE_8)
        return
    a, b = np.array(a, float), np.array(b, float)
    n = np.linalg.norm(b - a)
    if n < 1:
        return
    u, t = (b - a) / n, 0.0
    while t < n:
        cv2.line(img, tuple((a + u * t).astype(int)), tuple((a + u * min(t + dash, n)).astype(int)), color, 1, cv2.LINE_8)
        t += dash + gap


def draw_lattice(img, cr, to, rect=None, solid=False):
    """Решётка крестов: все линии сплошные; отметки — кружки.
    to(точка, мкм настоящие) -> пиксели."""
    v1, v2 = cr.vectors()
    if v1 is not None:
        i0, i1, j0, j1 = cr.extent()
        if (i1 - i0 + 1) * (j1 - j0 + 1) <= 40000:
            for i in range(i0, i1 + 1):
                dashed(img, to(cr.live(i, j0)), to(cr.live(i, j1)), RED, rect, solid=solid or i in (i0, i1))
            for j in range(j0, j1 + 1):
                dashed(img, to(cr.live(i0, j)), to(cr.live(i1, j)), RED, rect, solid=solid or j in (j0, j1))
    for q in cr.marks:
        u, v = to(q + cr.drift)
        if rect is None or (rect[0] <= u <= rect[0] + rect[2] and rect[1] <= v <= rect[1] + rect[3]):
            UI.dot(img, u + 0.5, v + 0.5, RED)


def zoom_view(shape, W, H, zoom=1.0, zc=None, rot=0.0):
    """Масштаб окна: пиксель кадра <-> пиксель окна. zoom = 1 — кадр целиком; zc — точка кадра в центре окна;
    rot — поворот картинки, градусы (кресты встают ровно)."""
    ih, iw = shape[:2]
    k = min(W / iw, H / ih) * zoom
    zc = np.array([iw / 2, ih / 2], float) if zc is None else np.asarray(zc, float)
    hw, hh = W / 2 / k, H / 2 / k                       # половина видимой части, px кадра
    zc = np.array([min(max(zc[0], min(hw, iw / 2)), max(iw - hw, iw / 2)),
                   min(max(zc[1], min(hh, ih / 2)), max(ih - hh, ih / 2))])
    mid = np.array([W / 2, H / 2])
    t = np.radians(rot)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    i2v = lambda pix: mid + k * (R @ (np.asarray(pix, float) - zc))
    v2i = lambda q: R.T @ ((np.asarray(q, float) - mid) / k) + zc
    return k, zc, i2v, v2i


class _Live:
    """Камера в отдельном потоке: окно перерисовывается сразу (сдвиг, зум, надписи), не дожидаясь кадра.
    hold() — на время действий (движение, фокус, снимок) поток не трогает камеру."""
    def __init__(self, m):
        self.m, self.img, self.n, self.run_, self.paused = m, m.image(0), 1, True, False
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.run_:
            if self.paused:
                time.sleep(0.01)
                continue
            m_ = self.m
            p0_ = (m_._p.get('x'), m_._p.get('y'), m_.objective)
            try:
                img = self.m.image(0)
            except Exception:
                time.sleep(0.05)
                continue
            self.img, self.n = img, self.n + 1
            if p0_ == (m_._p.get('x'), m_._p.get('y'), m_.objective) and not (m_.__dict__.get('_queue') is not None and m_.__dict__['_queue'].inst_busy()):
                fog_add(m_, img)                        # столик стоял весь кадр: кадр — на карту («туман»)

    def hold(self):
        live = self

        class _H:
            def __enter__(self):
                live.paused = True
                live.m.__dict__.setdefault('_cam_lock', threading.RLock()).acquire()   # дождаться текущего кадра

            def __exit__(self, *a):
                live.m._cam_lock.release()
                live.paused = False
        return _H()

    def stop(self):
        self.run_ = False
        self.t.join(1.0)


def render(img, W, H, k, zc, rot=0.0):
    """Кадр в окне W×H с масштабом k, точкой zc в центре и поворотом rot, градусы."""
    if abs(rot) < 1e-6 and abs(k - min(W / img.shape[1], H / img.shape[0])) < 1e-9:
        return fit(img, W, H)
    if abs(rot) < 1e-6:                                 # без поворота: вырезать видимое и только его масштабировать
        ih, iw = img.shape[:2]
        c = np.asarray(zc, float)
        x0, y0 = c[0] - W / 2 / k, c[1] - H / 2 / k
        a0, b0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        a1, b1 = min(iw, int(np.ceil(x0 + W / k)) + 1), min(ih, int(np.ceil(y0 + H / k)) + 1)
        out = np.full((H, W, 3), 24, np.uint8)
        if a1 <= a0 or b1 <= b0:
            return out
        M = np.float32([[k, 0, (a0 - x0) * k], [0, k, (b0 - y0) * k]])
        part = cv2.warpAffine(img[b0:b1, a0:a1], M, (W, H), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(24, 24, 24))
        return part if part.ndim == 3 else cv2.cvtColor(part, cv2.COLOR_GRAY2BGR)
    src, kk, c = img, k, np.asarray(zc, float)
    if k < 1:                                           # сначала уменьшить без ряби, потом сдвинуть и повернуть
        src = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        kk, c = 1.0, c * k
    t = np.radians(rot)
    R = kk * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    M = np.float32(np.c_[R, np.array([W / 2, H / 2]) - R @ c])
    out = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(24, 24, 24))
    return out if out.ndim == 3 else cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def live_map(m, shape, W, H, zoom=1.0, zc=None, rot=0.0):
    """Живое изображение: точка образца (настоящие мкм) <-> пиксель окна.
    Точка S видна в центре кадра, когда столик в S; столик сдвинут на d — картинка сдвинута на K·d."""
    ih, iw = shape[:2]
    k, zc, i2v, v2i = zoom_view(shape, W, H, zoom, zc, rot)
    K, d = m._pixels_per_step(), m.motor_step
    P = np.array([m._p['x'], m._p['y']], float)
    B = getattr(m, '_bl', None)                             # люфт (шаги на знак подхода): где столик на самом деле
    if B is not None:
        P = P + np.asarray(B, float) @ np.array([m._dir.get('x', 0), m._dir.get('y', 0)], float)
    P = P * d + _ar_d(m)                                    # + поправка по картинке (сверка с мозаикой)
    c = np.array([iw / 2, ih / 2])
    to = lambda S: tuple(i2v(c + K @ ((P - np.asarray(S)) / d)))
    back = lambda q: P - d * np.linalg.solve(K, v2i(q) - c)
    return to, back, i2v, v2i, c, k, zc


def camera_box(m, shape):
    """Углы живого кадра на столике (настоящие мкм, с учётом люфта) — рамка «камера здесь» на карте."""
    ih, iw = shape[:2]
    K, d = m._pixels_per_step(), m.motor_step
    P = np.array([m._p['x'], m._p['y']], float)
    B = getattr(m, '_bl', None)
    if B is not None:
        P = P + np.asarray(B, float) @ np.array([m._dir.get('x', 0), m._dir.get('y', 0)], float)
    P = P * d + _ar_d(m)
    c = np.array([iw / 2, ih / 2])
    return np.array([P - d * np.linalg.solve(K, np.array(u, float) - c) for u in ((0, 0), (iw, 0), (iw, ih), (0, ih))])


def fog_of(m):
    """«Туман войны» образца (fog.Fog): где камера уже была — на карте под снятым."""
    from .fog import Fog
    fd_ = m.sample.folder if getattr(m, 'sample', None) is not None else None
    f = m.__dict__.get('_fog')
    if f is None or m.__dict__.get('_fog_dir') != fd_:
        if f is not None:
            f.save()
        f = m.__dict__['_fog'] = Fog(fd_)
        m.__dict__['_fog_dir'] = fd_
    return f


def fog_add(m, img):
    """Живой кадр — в туман карты (если столик сдвинулся с прошлого заметно)."""
    try:
        fog_of(m).add(img, camera_box(m, img.shape))
    except Exception:
        pass


def fog_trace(m):
    """След переезда для тумана: trace(x, y, shot) для samples.drive_to (рабочие мкм). shot False — спросить,
    нужен ли здесь кадр (туман ещё не рассеян), True — столик уже здесь: снять кадр в туман."""
    def trace(x, y, shot):
        f = fog_of(m)
        if not shot:
            # туман перезаписывается новыми кадрами: вдруг на образце что-то изменилось. Снимать только в ещё
            # не виденных местах: m.fog_refresh = False
            return bool(m.fog_refresh) or not f.covered(x + m._off['x'], y + m._off['y'])
        img = m.image(0)
        f.add(img, camera_box(m, img.shape), min_move=0.0)
        return True
    return trace


def _ar_d(m):
    """Поправка положения столика по картинке, мкм: где столик на самом деле минус где он по модели (люфт и
    прочее). Её меряет ar_lock, сверяя живой кадр с уже снятой мозаикой, — кресты на кадре и контуры совпадают."""
    v = m.__dict__.get('_ar_d')
    return np.zeros(2) if v is None else np.asarray(v, float)


def ar_lock(m, img):
    """Сверить живой кадр с мозаикой съёмки образца под ним (самой подробной, накрывающей кадр) и поправить
    m._ar_d. -> (сдвиг, мкм, имя съёмки) или (None, причина)."""
    from .project import latest_scans
    from .snap import stage_to_mosaic, geometry
    if m.sample is None:
        return None, "no sample"
    ih, iw = img.shape[:2]
    K, d = m._pixels_per_step(), m.motor_step
    P = np.array([m._p['x'], m._p['y']], float)
    B = getattr(m, '_bl', None)
    if B is not None:
        P = P + np.asarray(B, float) @ np.array([m._dir.get('x', 0), m._dir.get('y', 0)], float)
    P = P * d + _ar_d(m)
    c = np.array([iw / 2, ih / 2])
    cam = lambda u: P - d * np.linalg.solve(K, np.asarray(u, float) - c)      # пиксель кадра -> мкм столика
    best = None
    for f in latest_scans(m.sample.folder):
        jp = os.path.join(f, 'join', 'preview_all.jpg')
        if not os.path.exists(jp):
            continue
        try:
            if json.load(open(os.path.join(f, 'join', 'stitch.json'), encoding='utf-8')).get('note'):
                continue                                    # сшивка не по образцу (кадры по столику): не опора
        except (OSError, ValueError):
            pass
        try:
            meta = json.load(open(os.path.join(f, 'meta.json'), encoding='utf-8'))
            g = geometry(f)
        except Exception:
            continue
        size = np.asarray(g['pos'], float).max(0) + (g['W'], g['H'])
        k = min(1.0, 4000 / size.max())
        corners = stage_to_mosaic(f, np.array([cam(u) for u in ((0, 0), (iw, 0), (0, ih), (iw, ih))]))
        inside = np.mean([(0 <= q[0] <= size[0]) and (0 <= q[1] <= size[1]) for q in corners])
        if inside < 0.75:
            continue
        o = meta.get('objective') or 0
        if best is None or o > best[0]:
            best = (o, f, k, corners)
    if best is None:
        return None, "no stitched scan under the camera"
    o, f, k, corners = best
    pv = _ar_cache(f)
    if pv is None:
        return None, "no preview"
    src = np.float32([[0, 0], [iw, 0], [0, ih]])
    dst = np.float32(corners[:3] * k)                       # пиксель кадра -> пиксель превью
    A = cv2.getAffineTransform(src, dst)
    s_ = float(np.hypot(*A[:, 0]))                          # пикселей превью на пиксель кадра
    w = min(1.0, max(s_, 160.0 / iw))                       # работать в разрешении превью (не меньше 160 px)
    Ww, Hw = max(32, int(iw * w)), max(32, int(ih * w))
    Aw = A @ np.array([[1 / w, 0, 0], [0, 1 / w, 0], [0, 0, 1]])
    E = cv2.warpAffine(pv, np.float32(Aw), (Ww, Hw), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
    V = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img, (Ww, Hw), interpolation=cv2.INTER_AREA)
    if np.mean(E > 4) < 0.7:
        return None, "the scan does not cover the frame"
    hp = lambda a_: (a_.astype(np.float32) - cv2.GaussianBlur(a_.astype(np.float32), (0, 0), max(2.0, Ww / 24)))
    win = cv2.createHanningWindow((Ww, Hw), cv2.CV_32F)
    (sx, sy), resp = cv2.phaseCorrelate(hp(E), hp(V), win)
    t = np.array([sx, sy]) / w                              # сдвиг содержимого кадра, пиксели камеры
    if resp < 0.05 or np.hypot(*t) > 0.3 * min(iw, ih):
        return None, f"no clear match (match {resp:.2f})"
    dP = d * np.linalg.solve(K, t)                          # c + K (P' - S) / d = c + K (P - S) / d + t
    new_ = _ar_d(m) + dP
    if np.hypot(*new_) > 0.25 * min(m.frame_size):          # столик так не ошибается: сверка ложная (мозаика
        m.__dict__['_ar_d'] = None                          # сшита неверно, пустой образец) — не верим
        return None, "the match is too far from the stage position"
    m.__dict__['_ar_d'] = new_
    return dP, __import__("microcontrol.project", fromlist=["x"]).scan_label(f)


_AR_CACHE = {}


def _ar_cache(f):
    jp = os.path.join(f, 'join', 'preview_all.jpg')
    key = (jp, os.path.getmtime(jp))
    if _AR_CACHE.get(f, (None,))[0] != key:
        im = cv2.imread(jp, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        if len(_AR_CACHE) > 6:
            _AR_CACHE.clear()
        _AR_CACHE[f] = (key, im)
    return _AR_CACHE[f][1]


STEPS = ["1 Overview", "2 Pick areas", "3 Detail", "4 Export"]
STEP_HELP = [
    "Step 1: put a small objective (4x or 10x), drag a rectangle on the map over the chip, then Scan area",
    "Step 2: open the overview (double-click it on the map or V), mark 3 crosses (C), number two squares (N), select the squares with flakes (M) and press Enter",
    "Step 3: put 50x or 100x, choose an area on the right and press Scan area",
    "Step 4: tick the areas on their cards, press Export (X): drawing for AutoCAD, pictures and mixed.png in <sample>/export/"]
BUTTONS = ["Prepare", "New area", "Scan area", "Stitch", "Export", "Settings"]
PREP_STEPS = [                                        # мастер Prepare: три шага (свет, тёмный кадр — не нужны)
    ('flat', "Flat field, white balance, exposure", [
        "Drive (arrows, map) to a clean EMPTY place of the substrate: no crosses, flakes or dust. Focus on it (F).",
        "Done: exposure is set (brightest ~80 %, no overexposure), white balance, then the stage visits 5 places a "
        "quarter frame apart and the flat field is averaged (uneven light, vignetting, dust on the optics).",
        "How often: for EACH objective; again after moving or changing the lamp, the diaphragms, or cleaning the optics. "
        "For lithography: every session."]),
    ('detail', "Focus and stage settling", [
        "Drive to a cross or a flake edge (details for the focus).",
        "Done: autofocus, then the stage moves and the pictures are compared until it stops shaking: that time becomes "
        "the pause before each picture.",
        "How often: every session."]),
    ('distortion', "Lens distortion", [
        "Stay at the cross or flake edge (details all around are good).",
        "Done: 3 x 3 pictures with 50 % overlap; the lens distortion of this objective is measured and used for every scan.",
        "How often: ONCE per objective; again only if the camera or the objective was taken off. Skip if done already."]),
]


MAP_COLS, MAP_ROWS = 11, 9    # видимое окно мини-карты, кадров
MAP_W = 360                   # ширина мини-карты, px


def screen(m):
    """Переход от осей столика к экрану: f(x, y) -> (h, v), где h растёт вправо, v — вниз;
    шаги решётки и размер поля зрения по h и v."""
    ax = m._screen_axes()
    (ah, sgh), (av, sgv) = ax['h'], ax['v']
    # карта идёт за стрелками: стрелка вправо — клетка столика вправо, вверх — вверх (с учётом arrows_flip)
    sgh = -sgh if getattr(m, 'arrows_flip_x', False) else sgh
    sgv = -sgv if getattr(m, 'arrows_flip_y', False) else sgv
    st = m._frame_step()                                           # клетка карты = ровно кадр камеры
    fw, fh = m.frame_size
    size = {'x': fw, 'y': fh}
    f = lambda x, y: (sgh * {'x': x, 'y': y}[ah], sgv * {'x': x, 'y': y}[av])
    return f, st[ah], st[av], size[ah], size[av]


def snap(m, A, pt):
    """Точка -> центр ближайшей клетки решётки с началом в A (угол B встаёт ровно на сетку). Оси столика.
    Клетка = кадр камеры; перекрытие кадров при съёмке добавляет scan.plan."""
    st = m._frame_step()
    ox, oy = A if A else (0.0, 0.0)
    return ox + round((pt[0] - ox) / st['x']) * st['x'], oy + round((pt[1] - oy) / st['y']) * st['y']


def _grid(m, A, B):
    """Решётка в экранных координатах: начало, шаги, клетка столика, зона (c0, c1, r0, r1) или None."""
    f, sh, sv, fw, fh = screen(m)
    oh, ov = f(*(A if A else (0.0, 0.0)))
    p = m.position
    ph, pv = f(p.x, p.y)
    ci, cj = round((ph - oh) / sh), round((pv - ov) / sv)
    zone = None
    if A:
        bi, bj = (round((f(*B)[0] - oh) / sh), round((f(*B)[1] - ov) / sv)) if B else (ci, cj)
        zone = (min(0, bi), max(0, bi), min(0, bj), max(0, bj))
    return f, (oh, ov), (sh, sv), (fw, fh), (ph, pv), (ci, cj), zone


def cell_of(m, A, B):
    """Кадр зоны, где стоит столик (как на экране): (строка, столбец, строк, столбцов); вне зоны — (None, None, ...)."""
    if not (A and B):
        return None
    *_, (ci, cj), (c0, c1, r0, r1) = _grid(m, A, B)
    nx, ny = c1 - c0 + 1, r1 - r0 + 1
    if c0 <= ci <= c1 and r0 <= cj <= r1:
        return cj - r0, ci - c0, ny, nx
    return None, None, ny, nx


def minimap(img, m, A, B, size=MAP_W, zcolor=None, others=(), right=0):
    """Мини-карта: бесконечная серая решётка кадров вокруг столика (рисуются только видимые клетки),
    зелёная зона (после клавиши 1 — предпросмотр от A до текущей клетки, после 2 — зафиксированная),
    жёлтая клетка — где столик, белая рамка — поле зрения. Номера кадров зоны по краям, сколько кадров
    зоны скрыто с каждой стороны и обзор всей зоны под картой, если она не помещается.
    Всё в экранных осях: h — вправо, v — вниз (с учётом swap_xy и mirror_x/y)."""
    f, (ox, oy), (sx, sy), (w, h), (px, py), (ci, cj), zone = _grid(m, A, B)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # окно карты — «комната»: стоит на месте, пока клетка столика не дошла до края (запас 1 клетка);
    # дошла — окно перелистывается, и столик оказывается у противоположного края новой комнаты
    if True:
        win = getattr(m, '_map_win', None)
        i0, j0 = win if win else (ci - (MAP_COLS - 1) // 2, cj - (MAP_ROWS - 1) // 2)
        if not (i0 + 1 <= ci <= i0 + MAP_COLS - 2):
            i0 = ci - 1 if ci > i0 + MAP_COLS - 2 else ci - (MAP_COLS - 2)
        if not (j0 + 1 <= cj <= j0 + MAP_ROWS - 2):
            j0 = cj - 1 if cj > j0 + MAP_ROWS - 2 else cj - (MAP_ROWS - 2)
    m._map_win = (i0, j0)
    k = (size - 70) / (MAP_COLS * sx)
    W = size
    H = int(MAP_ROWS * sy * k) + 60
    X0, Y0 = img.shape[1] - W - 8 - right, 8
    if X0 < 0 or Y0 + H > img.shape[0]:                                   # окно слишком мало для карты
        return
    sub = img[Y0:Y0 + H, X0:X0 + W]
    under = sub.copy()                                                   # для полупрозрачности
    sub[:] = (sub * 0.25 + np.array((26, 27, 31)) * 0.75).astype(sub.dtype)          # как остальные панели
    gx0, gy0 = 46, 30
    gw, gh = MAP_COLS * sx * k, MAP_ROWS * sy * k

    def to(hh, vv):         # экранные мкм -> пиксели карты
        return (int(round(gx0 + (hh - (ox + (i0 - 0.5) * sx)) * k)),       # округление (не отбрасывание):
                int(round(gy0 + (vv - (oy + (j0 - 0.5) * sy)) * k)))       # кадр и клетка совпадают до пикселя

    def cell(i, j, color, t=1):
        a = to(ox + (i - 0.5) * sx, oy + (j - 0.5) * sy)
        b = to(ox + (i + 0.5) * sx, oy + (j + 0.5) * sy)
        cv2.rectangle(sub, a, b, color, t)

    def label(txt, xy, color=(170, 230, 170), sc=0.38):
        (tw, th), _ = cv2.getTextSize(txt, font, sc, 1)
        cv2.putText(sub, txt, (int(xy[0] - tw / 2), int(xy[1] + th / 2)), font, sc, color, 1, cv2.LINE_AA)

    for i in range(i0, i0 + MAP_COLS):                                  # бесконечная серая решётка
        for j in range(j0, j0 + MAP_ROWS):
            cell(i, j, (70, 70, 70))
    if zone:
        c0, c1, r0, r1 = zone
        fixed = B is not None
        zc_ = zcolor or (80, 200, 80)
        fill, line = (SM.dim(zc_, 0.45), zc_) if fixed else (SM.dim(zc_, 0.35), tuple(min(255, int(v * 1.2)) for v in zc_))
        for i in range(max(c0, i0), min(c1, i0 + MAP_COLS - 1) + 1):
            for j in range(max(r0, j0), min(r1, j0 + MAP_ROWS - 1) + 1):
                cell(i, j, fill, -1)
                cell(i, j, line)
        ux0, vy0 = to(ox + (c0 - 0.5) * sx, oy + (r0 - 0.5) * sy)            # контур — только видимая часть
        ux1, vy1 = to(ox + (c1 + 0.5) * sx, oy + (r1 + 0.5) * sy)
        gx1, gy1 = gx0 + int(gw), gy0 + int(gh)
        cu0, cu1, cv0, cv1 = max(ux0, gx0), min(ux1, gx1), max(vy0, gy0), min(vy1, gy1)
        if cu0 < cu1 and cv0 < cv1:
            for (x_a, y_a, x_b, y_b, show) in ((cu0, cv0, cu1, cv0, vy0 >= gy0), (cu0, cv1, cu1, cv1, vy1 <= gy1),
                                               (cu0, cv0, cu0, cv1, ux0 >= gx0), (cu1, cv0, cu1, cv1, ux1 <= gx1)):
                if show:
                    cv2.line(sub, (x_a, y_a), (x_b, y_b), line, 2)
    gx1_, gy1_ = gx0 + int(gw), gy0 + int(gh)
    for (rx0, ry0, rx1, ry1), oc, olab, *rr_ in others:                 # другие области — своим цветом
        if rr_ and rr_[0]:                                               # контур (не прямоугольник): как есть
            ovm = sub.copy()
            for ring_ in rr_[0]:
                pf = np.array([to(*f(x_, y_)) for x_, y_ in ring_], float)
                pts = np.int32(np.round(pf))
                cv2.fillPoly(ovm, [pts], SM.dim(oc, 0.5), cv2.LINE_8)
                cv2.polylines(ovm, [pts], True, oc, 1, cv2.LINE_8)
            g1 = (gx0 + int(gw), gy0 + int(gh))
            sub[gy0:g1[1], gx0:g1[0]] = cv2.addWeighted(ovm, 0.75, sub, 0.25, 0)[gy0:g1[1], gx0:g1[0]]
            continue
        corners_ = [to(*f(xx, yy)) for xx, yy in ((rx0, ry0), (rx1, ry0), (rx0, ry1), (rx1, ry1))]
        u0, u1 = min(c_[0] for c_ in corners_), max(c_[0] for c_ in corners_)
        v0, v1 = min(c_[1] for c_ in corners_), max(c_[1] for c_ in corners_)
        cu0, cu1, cv0, cv1 = max(u0, gx0), min(u1, gx1_), max(v0, gy0), min(v1, gy1_)
        if cu0 < cu1 and cv0 < cv1:
            ovr = sub.copy()
            cv2.rectangle(ovr, (int(cu0), int(cv0)), (int(cu1), int(cv1)), SM.dim(oc, 0.6), -1)
            sub[:] = cv2.addWeighted(ovr, 0.5 if not olab else 0.25, sub, 0.5 if not olab else 0.75, 0)
            cv2.rectangle(sub, (int(cu0), int(cv0)), (int(cu1), int(cv1)), oc, 2 if not olab else 1)
            if olab and cu1 - cu0 > 26 and cv1 - cv0 > 14:              # другой объектив (слой): подпись
                cv2.putText(sub, olab, (int(cu0) + 3, int(cv0) + 12), font, 0.36, oc, 1, cv2.LINE_AA)
    a_, b_ = to(px - sx / 2, py - sy / 2), to(px + sx / 2, py + sy / 2)  # жёлтое — кадр = клетка карты
    ov_ = sub.copy()
    cv2.rectangle(ov_, (int(max(a_[0], gx0)), int(max(a_[1], gy0))),
                  (int(min(b_[0], gx0 + int(gw))), int(min(b_[1], gy0 + int(gh)))), (0, 200, 255), -1)
    sub[:] = cv2.addWeighted(ov_, 0.55, sub, 0.45, 0)
    # номера столбцов и строк — всегда, у каждой видимой клетки: от угла зоны (1 — первый кадр зоны),
    # без зоны — от угла A или рабочего нуля. Зелёные — внутри зоны, серые — вне, жёлтые — где столик.
    b0c, b0r = (zone[0], zone[2]) if zone else (0, 0)
    for c in range(i0, i0 + MAP_COLS):
        inside = zone and zone[0] <= c <= zone[1]
        col = (0, 220, 255) if c == ci else ((170, 230, 170) if inside else (150, 150, 150))
        label(str(c - b0c + 1), (to(ox + c * sx, oy)[0], gy0 - 10), col, 0.34)
    for r in range(j0, j0 + MAP_ROWS):
        inside = zone and zone[2] <= r <= zone[3]
        col = (0, 220, 255) if r == cj else ((170, 230, 170) if inside else (150, 150, 150))
        label(str(r - b0r + 1), (gx0 - 20, to(ox, oy + r * sy)[1]), col, 0.34)
    if zone:
        left, right = max(0, i0 - c0), max(0, c1 - (i0 + MAP_COLS - 1))
        up, down = max(0, j0 - r0), max(0, r1 - (j0 + MAP_ROWS - 1))
        hid = "  ".join(f"{a}{n_}" for a, n_ in (('<', left), ('>', right), ('^', up), ('v', down)) if n_)
        nx_, ny_ = c1 - c0 + 1, r1 - r0 + 1
        try:                                                             # кадров съёмки (с перекрытием)
            Bp = B if B else (m.position.x, m.position.y)
            fx_, fy_ = Area(A[0], A[1], *snap(m, A, Bp)).size(m._frame_step(1 - scan_overlap(m.objective)))
            from . import timing as TM
            nfr = f" fields, {fx_ * fy_} frames, " + \
                TM.fmt(TM.estimate(m, fx_ * fy_, 'points' if getattr(m, '_use_fp', True) else 'off')[0])
        except Exception:
            nfr = ""
        label(f"{'Area' if fixed else 'Preview'} {nx_} x {ny_}{nfr}",
              (W / 2, H - 12), line, 0.45)
    else:
        label("Key 1: corner of a new area", (W / 2, H - 12), (170, 170, 170), 0.45)
    cr = getattr(m, '_crosses', None)
    if cr is not None and cr.marks:                                      # решётка крестов (красная)
        offv = np.array([m._off['x'], m._off['y']])
        draw_lattice(sub, cr, lambda S: to(*f(*(np.asarray(S) - offv))), (gx0, gy0, int(gw), int(gh)), solid=True)
    nreg = len(getattr(m, '_regions', None) or [])
    for num, k_, f_ in SM.numbered(m):                                   # точки фокуса: сквозные номера
        u, v = to(*f(f_['x'] - m._off['x'], f_['y'] - m._off['y']))
        if gx0 <= u <= gx0 + gw and gy0 <= v <= gy0 + gh:
            pc = SM.col(m, k_ if k_ >= 0 else nreg)
            cv2.circle(sub, (u, v), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(sub, (u, v), 3, pc, -1, cv2.LINE_AA)
            cv2.putText(sub, str(num), (u + 6, v - 4), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(sub, to(px - sx / 2, py - sy / 2), to(px + sx / 2, py + sy / 2), (255, 255, 255), 2)   # кадр
    cv2.rectangle(sub, (gx0, gy0), (gx0 + int(gw), gy0 + int(gh)), (120, 120, 120), 1)
    sub[:] = cv2.addWeighted(sub, 0.85, under, 0.15, 0)                 # карта слегка прозрачная
    cv2.rectangle(sub, (0, 0), (sub.shape[1] - 1, sub.shape[0] - 1), (78, 80, 88), 1)
    if zone and (c1 - c0 >= MAP_COLS or r1 - r0 >= MAP_ROWS):          # обзор всей зоны
        nx_, ny_ = c1 - c0 + 1, r1 - r0 + 1
        ok = 120 / max(nx_ * sx, ny_ * sy)
        oW, oH = max(8, int(nx_ * sx * ok)), max(8, int(ny_ * sy * ok))
        oxp, oyp = img.shape[1] - 12 - oW - 4 - right, Y0 + H + 6
        ov = img[oyp:oyp + oH + 4, oxp:oxp + oW + 4]
        ov[:] = (ov * 0.25).astype(ov.dtype)
        to2 = lambda i, j: (int(2 + (i - c0 + 0.5) * sx * ok), int(2 + (j - r0 + 0.5) * sy * ok))
        cv2.rectangle(ov, (2, 2), (oW + 1, oH + 1), (80, 200, 80), 1)
        cv2.rectangle(ov, to2(i0 - 0.5, j0 - 0.5), to2(i0 + MAP_COLS - 0.5, j0 + MAP_ROWS - 0.5), (0, 200, 255), 1)
        cv2.circle(ov, to2(ci, cj), 2, (255, 255, 255), -1)


Z_STEP, SHIFT = 100.0, 5          # шаг по Z в ручном режиме, мкм; Shift+E / Shift+Q — в 5 раз больше


def R_scr(rot):
    t = np.radians(rot)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def smove(m, cr, ax, d):
    """Шаг по оси ax на d мкм; при включённой синхронизации — ещё и по другой оси вдоль наклона решётки:
    по X на Δx -> по Y на Δx·v1y/v1x (= tg α · Δx); по Y на Δy -> по X на Δy·v2x/v2y.
    Остаток меньше шага мотора копится и уходит в следующий шаг."""
    v1, v2 = cr.vectors()
    if not getattr(m, '_sync', False) or v1 is None:
        return m.move(**{'d' + ax: d})
    other = 'y' if ax == 'x' else 'x'
    ratio = v1[1] / v1[0] if ax == 'x' else v2[0] / v2[1]
    m._sync_rem[other] += d * ratio
    n = round(m._sync_rem[other] / m.motor_step)
    extra = n * m.motor_step
    m._sync_rem[other] -= extra
    return m.move(**{'d' + ax: d, **({'d' + other: extra} if n else {})})


def default_steps(m):
    """Шаги ручного режима, мкм: X, Y — ровно клетка мини-карты (кадр, округлённый вниз до целого числа шагов
    мотора; не больше max_move) — столик при каждом нажатии встаёт точно на сетку; Z — 100 (кратно 5)."""
    st = m._frame_step()
    s = m.motor_step
    cell = lambda v: round(min(v, math.floor(m.max_move / s + 1e-9) * s), 2)
    r5 = lambda v, lim: float(max(5, min(lim, v) // 5 * 5))
    return {'x': cell(st['x']), 'y': cell(st['y']), 'z': r5(Z_STEP, m.max_move_z)}


def step_limit(m, ax):
    return m.max_move if ax in 'xy' else m.max_move_z


FIELDS = ('x', 'y', 'z', 'exp', 'obj', 'afs', 'afr')   # порядок полей (Tab); afs, afr — автофокус этого объектива


def af_auto(m):
    """Шаг и окно автофокуса объектива: заданные (m.af_step, m.af_range) или от глубины резкости (0 = авто)."""
    dof = m.depth_of_field
    return (m.af_step or round(max(dof, m.motor_step), 1)), (m.af_range or round(max(6 * dof, 3 * max(dof, m.motor_step)), 0))


def draw_fields(view, steps, edit, scale, top, m=None):
    """Поля в левом верхнем углу: шаги X, Y, Z, экспозиция, объектив. Возвращает прямоугольники полей."""
    rects = {}
    y = top
    cur = dict(steps)
    if m is not None:
        cur.update(exp=m.exposure, obj=m.objective, afs=af_auto(m)[0], afr=af_auto(m)[1])
    for ax in (FIELDS if m is not None else 'xyz'):
        val = (edit[1] + "_") if (edit and edit[0] == ax) else f"{cur[ax]:g}"
        if ax == 'exp':
            txt = f"Exposure [ {val:>6} ] ms"
        elif ax == 'obj':
            txt = f"Objective[ {val:>6} ] x   (4, 10, 20, 100; M: next)"
        elif ax in ('afs', 'afr'):
            txt = f"{'AF step ' if ax == 'afs' else 'AF window'}[ {val:>6} ] um"
        else:
            txt = f"Step {ax.upper()}  [ {val:>6} ] um" + ("   Shift: x5" if ax == 'z' else "")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x0, y0, x1, y1 = 8, y - th - 6, 16 + tw, y + 6
        active = edit and edit[0] == ax
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 200, 255) if active else (90, 90, 90), 1)
        cv2.putText(view, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 220, 255) if active else (235, 235, 235), 1, cv2.LINE_AA)
        rects[ax] = (x0, y0, x1, y1)
        y += th + 16
    return rects, y


def run(m, fullscreen=False, _keys=None):
    """Окно; если что-то упало — полный текст ошибки пишется в журнал (logs/MC_*.log), окно закрывается."""
    try:
        return _run(m, fullscreen=fullscreen, _keys=_keys)
    except Exception:
        import traceback
        L.event('MAN-999', "The window stopped with an error:\n" + traceback.format_exc(), 40)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        raise


def _run(m, fullscreen=False, _keys=None):
    m._need_cam()
    WH0 = open_window(WIN, fullscreen)
    if fullscreen:
        SW, SH = screen_size()
    L.event('MAN-001', f"Manual mode at {m.position}.")
    show_help, msg, msg_t, sharp, n = False, "Click the map to go there, drag on it for a new area. Tab: big map. H: all keys", time.monotonic(), 0.0, 0
    SM.init_areas(m)                                     # области образцов (настоящие мкм)
    car = dict(scroll=0, hits=[], confirm=None, t=0.0, multi=[], tools=[])   # карусель справа; multi — отмеченные
    m.__dict__['_ui_car'] = car                            # тесты: где кнопки карусели
    mdr = dict(on=False)                                 # Shift + тянуть: сдвиг выделения (индекс, начало, сдвиг)
    gsel, gsel0 = {}, {}                                 # квадраты и подсетки в съёмках (сейчас / как было) — для Ctrl+Z

    def grid_state(fd_):
        try:
            d_ = json.load(open(os.path.join(fd_, 'grid.json'), encoding='utf-8'))
            return dict(selected=d_.get('selected', []), refine=d_.get('refine') or {})
        except (OSError, ValueError):
            return {}

    def refresh_gs():
        """Выбранное в просмотре (grid.json последних съёмок) — в состояние окна: Ctrl+Z отменяет и его,
        и всё, что из него вышло (черновики), вместе — иначе синхронизация тут же вернула бы отменённое."""
        from .layers import _latest_grids
        for _r, fd_, _d in _latest_grids(m):
            st_ = grid_state(fd_)
            gsel0.setdefault(fd_, st_)
            gsel[fd_] = st_

    def note_brush(reg):
        """Перед операцией — квадраты съёмок уже в состоянии (refresh_gs); оставлено для ясности вызовов."""
        return None

    def noted_done():
        refresh_gs()
    A = B = None
    if not getattr(m, '_manual_steps', None):
        m._manual_steps = default_steps(m)
    steps = m._manual_steps                              # шаги X, Y, Z, мкм (помнятся между входами)
    edit = None                                          # (ось, набираемые цифры) — поле в режиме ввода
    clicks = []
    from .crosses import Crosses
    if getattr(m, '_crosses', None) is None:
        m._crosses = Crosses()
    cr = m._crosses
    if getattr(m, '_guide', None) is None:               # красные направляющие: точка (от центра кадра, px) и угол
        m._guide = dict(u=0.0, v=0.0, ang=0.0, show=True)
    gd = m._guide
    drag = False
    if not hasattr(m, '_sync'):
        m._sync, m._sync_rem = False, {'x': 0.0, 'y': 0.0}
    if not hasattr(m, '_use_fp'):
        m._use_fp = True                                 # Z кадров скана — по точкам фокуса
    track = []
    if getattr(m, '_focus_pts', None) is None:            # точки фокуса: углы A, B (клавиши 1, 2) и F — {x, y, z} настоящие
        m._focus_pts = []

    hover = [None, 0.0]                                  # где мышь и когда двигалась (надписи прячутся)
    zm = dict(z=1.0, c=None, pan=None)                   # масштаб окна: колесо — зум, средняя кнопка / Shift — сдвиг

    def on_field(x, y):
        return any(x0 <= x <= x1 and y0 <= y <= y1 for (x0, y0, x1, y1) in rects.values())

    menu = dict(hover=None, hits=[], thits=[], thover=None)   # меню; под ним — шаги и большие кнопки
    from .navmap import NavMap
    if getattr(m, '_nav', None) is None:
        m._nav = NavMap()
        m._panel_hidden = True                           # настройки — по кнопке Settings
    nav = m._nav
    nv = dict(big=getattr(m, '_nav_big', False), mr=None, inset=None, press=None, drag=None, pan=None)

    def to_logical(x, y):
        """x окна -> x раскладки по умолчанию (камера | карта | список справа): рисуется по умолчанию,
        на экран — переставленным; щелчки — обратно."""
        yy_ = y - UI.MENU_H - UI.TOOL_H
        if yy_ < 0 or lay.get('Wl') is None:
            return x
        Wl_ = lay['Wl']
        if lay['list'] == 'left':
            x = x + Wl_ if x < SM.CAR_W else x - SM.CAR_W
        if lay['panes'] == 'map|camera' and lay.get('Wc') and x < Wl_ and yy_ < (lay.get('Hp') or 0):
            Wm_ = Wl_ - lay['Wc']
            x = x + lay['Wc'] if x < Wm_ else x - Wm_
        return x

    def in_rect(r_, x, y):
        return r_ is not None and r_[0] <= x <= r_[2] and r_[1] <= y <= r_[3]

    pick = {}                                              # список съёмок выделения посреди окна («...» на карточке)
    m.__dict__['_ui_pick'] = pick

    def on_mouse(ev, x, y, flags, *_):
        if pick.get('rects'):                              # список съёмок: пока открыт — только он
            oy_ = UI.MENU_H + UI.TOOL_H
            hit_ = next((h_ for r_, h_ in pick['rects'][1:] if in_rect(r_, x, y - oy_)), None)
            if ev == cv2.EVENT_MOUSEMOVE:
                pick['hover'] = (x, y - oy_)
            elif ev == cv2.EVENT_LBUTTONDOWN:
                pick['click'] = hit_ if hit_ is not None else ('panel' if in_rect(pick['rects'][0][0], x, y - oy_) else 'close')
            elif ev == cv2.EVENT_MOUSEWHEEL:
                pick['scroll'] = max(0, pick.get('scroll', 0) - int(np.sign(wheel_delta(flags))))
            return
        x = to_logical(x, y)                               # раскладка (список слева, карта слева): x как у логики
        if wiz.get('rects'):                               # кнопки мастера Prepare — раньше всего остального
            hit_ = next((k2 for k2, r2 in wiz['rects'].items() if r2[0] <= x <= r2[2] and r2[1] <= y <= r2[3]), None)
            if ev == cv2.EVENT_MOUSEMOVE:
                wiz['hover'] = (x, y - UI.MENU_H - UI.TOOL_H)
            if hit_ is not None:
                if ev == cv2.EVENT_LBUTTONDOWN:
                    wiz['click'] = hit_
                return
        on_mouse_(ev, x, y, flags)

    def on_mouse_(ev, x, y, flags, *_):
        # тянуть любой кнопкой по картинке (левой — не по полям) — сдвиг увеличенной картинки; колесо, + / - — зум
        if y < UI.MENU_H and not zm['pan'] and nv['press'] is None:     # строка меню
            menu['hover'] = x if ev == cv2.EVENT_MOUSEMOVE else menu['hover']
            if ev == cv2.EVENT_LBUTTONDOWN:
                clicks.append(('menu', x))
            return
        menu['hover'] = None
        if y < UI.MENU_H + UI.TOOL_H and not zm['pan'] and nv['press'] is None:   # шаги и кнопки
            menu['thover'] = x if ev == cv2.EVENT_MOUSEMOVE else menu['thover']
            if ev == cv2.EVENT_LBUTTONDOWN:
                clicks.append(('tbar', x))
            return
        menu['thover'] = None
        y -= UI.MENU_H + UI.TOOL_H
        if qbar['rects'] and ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_MOUSEWHEEL) and \
                in_rect(qbar['rects'][0][0], x, y) and not zm['pan'] and nv['press'] is None:   # панель очереди
            if ev == cv2.EVENT_LBUTTONDOWN:
                h_ = next((r_ for r_ in qbar['rects'][1:] if in_rect(r_[0], x, y)), None)
                clicks.append(('queue', h_[1], h_[2]) if h_ else ('queue', 'menu', None))
            return
        if nv['press'] is not None or nv['pan'] is not None or \
                (nv['mr'] is not None and in_rect(nv['mr'], x, y) and (not nv['big'] or not in_rect(nv['inset'], x, y))
                 and not (nv['big'] and nv.get('pbox') and in_rect(nv['pbox'], x, y))
                 and not zm['pan'] and not mdr['on']):            # большая карта или маленькая в углу
            mx_, my_ = x - nv['mr'][0], y - nv['mr'][1]                # карта (как в стратегиях: щёлкнул — поехал)
            if ev == cv2.EVENT_LBUTTONDBLCLK:
                clicks.append(('mapdbl', mx_, my_))
            elif ev == cv2.EVENT_LBUTTONDOWN:
                nv['press'] = (mx_, my_)
            elif ev == cv2.EVENT_MOUSEMOVE and nv['press'] is not None and flags & cv2.EVENT_FLAG_LBUTTON:
                if abs(mx_ - nv['press'][0]) + abs(my_ - nv['press'][1]) > 6:
                    nv['drag'] = (*nv['press'], mx_, my_)
                    nv['free'] = bool(flags & cv2.EVENT_FLAG_CTRLKEY)   # Ctrl: свободная рамка (кристалл)
            elif ev == cv2.EVENT_LBUTTONUP and nv['press'] is not None:
                free_ = bool(flags & cv2.EVENT_FLAG_CTRLKEY) or nv.get('free', False)
                clicks.append(('maprect', *nv['drag'], free_) if nv['drag'] else ('mapclick', *nv['press']))
                nv['press'], nv['drag'], nv['free'] = None, None, False
            elif ev in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
                nv['pan'] = (mx_, my_, None if nav.c is None else nav.c.copy())
            elif ev == cv2.EVENT_MOUSEMOVE and nv['pan'] is not None:
                clicks.append(('mappan', mx_, my_))
            elif ev in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
                nv['pan'] = None
            elif ev in (cv2.EVENT_MOUSEWHEEL, getattr(cv2, 'EVENT_MOUSEHWHEEL', -99)):   # колесо / тачпад
                gs_ = wheel_gesture(ev, flags)
                clicks.append(('mapwheel', mx_, my_, gs_[1]) if gs_[0] == 'zoom' else ('mapscroll', *gs_[1:]))
            return
        if in_rect(nv['inset'], x, y) and nv['big'] and ev == cv2.EVENT_LBUTTONDOWN:   # камера в углу — крупно
            clicks.append(('swap',))
            return
        if ev == cv2.EVENT_MOUSEMOVE:
            hover[0], hover[1] = (x, y), time.monotonic()
        if car.get('press') is not None:                 # тянут карточку или полосу прокрутки карусели
            if ev == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
                clicks.append(('cmove', x, y))
            elif ev == cv2.EVENT_LBUTTONUP or (ev == cv2.EVENT_MOUSEMOVE and not flags & cv2.EVENT_FLAG_LBUTTON):
                clicks.append(('cdrop', x, y))
            return
        if x >= W - SM.CAR_W and not zm['pan']:          # карусель образцов справа
            if ev == cv2.EVENT_LBUTTONDOWN:
                clicks.append(('car', x, y, bool(flags & cv2.EVENT_FLAG_CTRLKEY)))
            elif ev == cv2.EVENT_MOUSEWHEEL:
                clicks.append(('carw', x, y, wheel_delta(flags)))
            return
        if mdr['on']:                                    # тянут маску (Shift + левая кнопка)
            if ev == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
                clicks.append(('mdrag', x, y))
            elif ev == cv2.EVENT_LBUTTONUP or (ev == cv2.EVENT_MOUSEMOVE and not flags & cv2.EVENT_FLAG_LBUTTON):
                mdr['on'] = False
                clicks.append(('mdrag1', x, y))
            return
        if ev == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_SHIFTKEY and not on_field(x, y):
            mdr['on'] = True
            clicks.append(('mdrag0', x, y))
            return
        if ev in (cv2.EVENT_MOUSEWHEEL, getattr(cv2, 'EVENT_MOUSEHWHEEL', -99)):   # колесо / тачпад
            gs_ = wheel_gesture(ev, flags)
            clicks.append(('wheel', x, y, gs_[1]) if gs_[0] == 'zoom' else ('scroll', x, y, *gs_[1:]))
        elif ev in (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) or (ev == cv2.EVENT_LBUTTONDOWN and not on_field(x, y)):
            zm['down'] = (x, y)
            clicks.append(('pan0', x, y))
        elif ev == cv2.EVENT_MOUSEMOVE and zm['pan'] and \
                flags & (cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_MBUTTON | cv2.EVENT_FLAG_RBUTTON):
            clicks.append(('pan', x, y))
        elif ev in (cv2.EVENT_MBUTTONUP, cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP) and zm['pan']:
            clicks.append(('pan1', x, y))
            d0 = zm.pop('down', None)
            if ev == cv2.EVENT_LBUTTONUP and d0 and abs(x - d0[0]) + abs(y - d0[1]) <= 4:
                clicks.append((x, y))                                # просто щелчок по картинке
        elif ev == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
    if _keys is None:
        cv2.setMouseCallback(WIN, on_mouse)
    keys = iter(_keys) if _keys is not None else None
    if m.scan_check and not getattr(m, '_orient_checked', False):   # раз за сеанс: знаки осей картинки камерой
        try:
            r = m.check_orientation(swap=False)       # поворот на 90° — только клавишей O
            m._orient_checked = 'Reason' not in r
            msg, msg_t = "Orientation: " + r.get('Changed', r.get('Result', r.get('Reason', ''))), time.monotonic()
            if 'Changed' in r:
                msg += ". If the arrow keys now go the wrong way: [ - left / right, ] - up / down"
        except MicroControlError as e:
            msg, msg_t = str(e), time.monotonic()
    rects = {}
    live = _Live(m) if keys is None or getattr(m, '_test_live', False) else None   # в тестах — синхронно
    try:                                                  # окно упало посреди съёмки: снятое — quick-сшить
        from .jobs import unstitched as _uns
        from .project import scan_label as _sl
        for f_ in _uns(m.sample.folder):
            m.queue.stitch(f_, 'quick', title=f"Quick stitch {_sl(f_)} (interrupted)")
    except Exception:
        pass
    qbar = {'rects': [], 'paused': False, 'h': 0}
    from .util import Prof, PROF_KEYS
    prof = Prof('microscope')                            # `: сколько мс на что уходит в кадре окна
    FAST_OK = keys is None or bool(os.environ.get('MC_FAST'))   # карта едет — вдвое грубее (не в проверках)
    QBUSY = "wait for the queue or stop its job at the instrument (x on the bar at the bottom)"

    def qbusy():
        """Прибор занят очередью (съёмка, пересъёмка): движение, свет, фокус, объектив — нельзя."""
        q_ = m.__dict__.get('_queue')
        return q_ is not None and q_.inst_busy()
    last_n, last_t, img = -1, 0.0, None
    W, H = WH0 if WH0 else (1280, 900)

    def work(P):                                         # настоящие -> рабочие мкм
        return (P[0] - m._off['x'], P[1] - m._off['y'])

    def reg_rect(reg):
        fw_, fh_ = m.frame_size
        return SM.rect(reg, fw_, fh_)               # размер кадра — объектива самого выделения

    def reg_area(reg):
        return Area(*work(reg['A']), *work(reg['B']))

    def reg_extent(reg):                                   # для списка: по контуру — его рамка, не A-B
        if reg.get('shape'):
            from .layers import geom as _gm
            b_ = _gm(m, reg).bounds
            return Area(*work(b_[:2]), *work(b_[2:]))
        return reg_area(reg)

    def first_num(k):                                    # номера точек фокуса области k
        nums = [n_ for n_, kk, _p in SM.numbered(m) if kk == k]
        return (f"{nums[0]}-{nums[-1]}" if len(nums) > 1 else str(nums[0])) if nums else "-"

    def stage_backlash():
        """Люфт столика (шаги на знак подхода, 2x2) — по сшивке съёмки образца с наибольшим числом кадров."""
        from .project import Selection
        from .snap import geometry
        best = None
        for rg_ in m._regions:
            if rg_.get('kind') == 'group':
                continue
            sc_ = Selection(m.sample, rg_).latest
            if not sc_ or not os.path.exists(os.path.join(sc_.folder, 'join', 'positions_all.csv')):
                continue
            g_ = geometry(sc_.folder)
            if len(g_['p']) >= 4 and (best is None or len(g_['p']) > best[0]):
                best = (len(g_['p']), g_['B'])
        return best[1] if best else getattr(m, '_bl', None)

    def draft_to_selection(k):
        """Черновик -> выделение под объектив, что стоит (снимается, идёт в экспорт). Обратного хода нет."""
        from .layers import draft_to_selection as _d2s
        ok_, msg_ = _d2s(m, k, m.objective)
        if ok_:
            SM.persist(m)
        return msg_

    def cycle_objective(k):
        """Объектив выделения — следующий из списка (только пока его не снимали: папка съёмок — по объективу)."""
        from .project import Selection
        from .config import OBJECTIVES
        rg_ = m._regions[k]
        if rg_.get('kind') == 'group':
            return "A group has no objective"
        if rg_.get('kind') == 'draft':                      # без объектива -> объектив, что стоит (или следующий)
            return draft_to_selection(k)
        from .layers import next_objective_of, to_draft
        objs_ = sorted(OBJECTIVES)
        if SM.obj(m, rg_) == objs_[-1]:                     # после самого крупного — draft (без объектива)
            msg_ = to_draft(m, k)
            SM.persist(m)
            return msg_
        ok_, msg_ = next_objective_of(m, k, objs_)
        if ok_:
            SM.persist(m)
        return msg_

    def car_tools():
        """Кнопки операций под заголовком — по отмеченным карточкам."""
        mk_ = [k_ for k_ in car['multi'] if k_ < len(m._regions)]
        car['multi'] = mk_
        if len(mk_) >= 2:
            car['tools'] = ["Merge", "Minus", "Overlap"]
        elif len(mk_) == 1:
            car['tools'] = []
        else:
            car['tools'] = []
        if not car['tools'] and any(r_.get('hidden') for r_ in m._regions):
            car['tools'] = ["Show all"]
        return car['tools']

    def layer_op(name):
        """Операция над отмеченными выделениями (первое отмеченное — цель)."""
        from . import layers as LY
        if name == "Show all":
            n_ = sum(1 for r_ in m._regions if r_.pop('hidden', None))
            SM.persist(m)
            return f"{n_} hidden selection{'s' if n_ != 1 else ''} shown again"
        fn = {"Merge": LY.union, "Minus": LY.subtract, "Overlap": LY.intersect, "Group": LY.group,
              "Ungroup": LY.ungroup}[name]
        ks_ = list(car['multi'])
        for k_ in ks_:                                 # квадраты кисти — для Ctrl+Z (операция их отпускает)
            note_brush(m._regions[k_])
        try:
            r_ = fn(m, ks_)
        except ValueError as e:
            return str(e)
        car['multi'] = []
        noted_done()
        SM.persist(m)
        L.event('MAN-012', r_ + ".")
        return r_ + " (Ctrl+Z: back)"

    def pending_ring():
        """Прямоугольник новой области: кадры от угла 1 до того, где столик (настоящие мкм), или None."""
        if not m._pending or SM.obj(m, m._pending) != m.objective:
            return None
        rp = m.real_position
        x0_, y0_, x1_, y1_ = SM.rect(dict(m._pending, B=(rp.x, rp.y)), *m.frame_size)
        ring_ = np.array([(x0_, y0_), (x1_, y0_), (x1_, y1_), (x0_, y1_)], float)
        try:                                               # пирамидка: видно сразу, где обрежется
            from .layers import clip_to_parent, geom as _g, box as _bx, set_geom as _sg
            tmp_ = dict(m._pending, B=(rp.x, rp.y))
            _sg(m, tmp_, _bx(x0_, y0_, x1_, y1_))
            if clip_to_parent(m, tmp_)[1]:
                c_ = _g(m, tmp_)
                c_ = max(getattr(c_, 'geoms', [c_]), key=lambda p_: p_.area)
                if c_.geom_type == 'Polygon' and not c_.is_empty:
                    ring_ = np.array(c_.exterior.coords)[:-1]
        except Exception:
            pass
        return ring_

    def nav_here():
        """Карта — на то место, где сейчас камера (не искать её глазами после Tab)."""
        try:
            cam_ = camera_box(m, (m.frame_size[1], m.frame_size[0]))
            if nav.c is not None:
                nav.c = np.asarray(cam_, float).mean(0)
                m._map_win = None
        except Exception:
            pass

    def draw_nav(view, Wl, bh_, shape):
        """Карта образца: в углу кадра (камера крупно) или на всё поле (Tab), камера тогда — в углу."""
        from .layers import rings as _rings
        mw_, mh_ = min(400, Wl // 2), min(290, H // 2)
        inset = (Wl - mw_ - 10, 10, Wl - 10, 10 + mh_)
        mr = (0, 0, Wl, H - bh_) if nv['big'] else inset
        if lay.get('Wc'):                                  # два вида: карта — своя половина окна, во всю высоту
            mr, inset = (lay['Wc'], 0, Wl, H - bh_), None
        nv['mr'], nv['inset'] = mr, inset
        w_, h_ = mr[2] - mr[0], mr[3] - mr[1]
        cam = camera_box(m, shape)
        nav.turn = int(getattr(m, 'map_turn', 0) or 0) % 360   # поворот карты (\\: ещё на 90°, помнится)
        from .util import orient_of
        Q_ = orient_of(m._pixels_per_step())                # образец на карте — как в камере (T, зеркала)
        if not np.array_equal(Q_, nav.Q):
            nav.Q = Q_
            try:                                           # просмотр (свой процесс) — так же
                if not os.environ.get('MC_TEST') or os.environ.get('MC_UI'):
                    set_ui_pref('orient', [int(v) for v in np.ravel(Q_)])
            except OSError:
                pass
        if nav.c is None:                                  # первый раз: всё, что есть, и камера
            bx = [SM._rect(m, r_) for r_ in m._regions if r_.get('kind') != 'group']
            bx.append((*cam.min(0), *cam.max(0)))
            if len(bx) == 1:
                c_ = cam.mean(0)
                fw_ = max(m.frame_size)
                bx = [(c_[0] - 4 * fw_, c_[1] - 3 * fw_, c_[0] + 4 * fw_, c_[1] + 3 * fw_)]
            nav.fit(bx, w_, h_)
        camsmall = cv2.resize(view[:mh_ * 0 + H - bh_, :Wl], (mw_, int(mw_ * (H - bh_) / Wl))) \
            if nv['big'] and not lay.get('Wc') else None
        ls_ = latest_scans()
        from .layers import rings as _rings
        al_ = map_alpha()                                  # прозрачность картинок выделений ([ ]); по контуру выделения
        clip_ = {}
        for f_, k_ in ls_:                                 # снятое за цветной границей не видно: граница = край картинки
            try:
                clip_[f_] = [np.asarray(r_, float) for r_ in _rings(m, m._regions[k_])]
            except Exception:
                pass
        mp = nav.base([f_ for f_, _k in ls_], w_, h_, clip=clip_,
                      alpha={f_: al_.get(m._regions[k_]['uid'], 1.0) for f_, k_ in ls_}, fog=fog_of(m),
                      fast=FAST_OK and time.monotonic() - nv.get('fast_t', -9.0) < 0.2)
        boxes_ = {}
        from .layers import geom as _gm
        base0_ = mp.copy()
        chosen_ = m._sel is not None and m._sel < len(m._regions) and not SM.hidden(m, m._sel)
        clear_map = []                                      # выделения — чистые (там смотрят), вокруг — темнее
        for k_, rg_ in ([] if chosen_ else enumerate(m._regions)):   # выбрано одно — только оно (ниже)
            if rg_.get('kind') != 'group' and not SM.hidden(m, k_):
                clear_map += UI.geom_shapes(_gm(m, rg_), nav.to_px)
        UI.tint_outside(mp, clear_map, (30, 30, 30), 0.45)
        for k_, rg_ in ([] if chosen_ else enumerate(m._regions)):   # пирамидка: внутри — всё, кроме детей
            if rg_.get('kind') == 'group' or SM.hidden(m, k_):
                continue
            kids_ = [r2 for k2, r2 in enumerate(m._regions) if r2.get('parent') == rg_.get('uid')
                     and r2.get('kind') != 'group' and not SM.hidden(m, k2)]
            if not kids_:
                continue
            mk_ = np.zeros(mp.shape[:2], np.uint8)
            for ext_, hs_ in UI.geom_shapes(_gm(m, rg_), nav.to_px):
                cv2.fillPoly(mk_, [np.int32(np.round(ext_))], 255, cv2.LINE_8)
            for r2 in kids_:
                for ext_, hs_ in UI.geom_shapes(_gm(m, r2), nav.to_px):
                    cv2.fillPoly(mk_, [np.int32(np.round(ext_))], 0, cv2.LINE_8)
            tn_ = cv2.addWeighted(mp, 0.75, np.full_like(mp, SM.col(m, k_)), 0.25, 0)
            cv2.copyTo(tn_, mk_, mp)
        if chosen_:
            ms_ = np.zeros(mp.shape[:2], np.uint8)          # выбранное: внутри — как есть, снаружи — на 15 % темнее
            for ext_, hs_ in UI.geom_shapes(_gm(m, m._regions[m._sel]), nav.to_px):
                cv2.fillPoly(ms_, [np.int32(np.round(ext_))], 255, cv2.LINE_8)
                for h_ in hs_:
                    cv2.fillPoly(ms_, [np.int32(np.round(h_))], 0, cv2.LINE_8)
            if ms_.any():
                mp = cv2.convertScaleAbs(mp, alpha=0.85)
                cv2.copyTo(base0_, ms_, mp)
        lattice_lines(mp, w_, h_, m._regions[m._sel] if chosen_ else None)   # сетка, по которой встанут квадраты
        profile_lines(mp)                                   # линии профилей из просмотра (P) и края на них
        for k_, rg_ in sorted(enumerate(m._regions), key=lambda t: SM.obj(m, t[1])):
            if rg_.get('kind') == 'group' or SM.hidden(m, k_):
                continue
            c_ = SM.col(m, k_)
            for ring_ in _rings(m, rg_):
                nav.poly(mp, ring_, c_ if k_ == m._sel else SM.dim(c_, 0.85), 2 if k_ == m._sel else 1,
                         dashed=rg_.get('kind') == 'draft')
            pts_ = np.vstack([nav.to_px(r_) for r_ in _rings(m, rg_)]) if _rings(m, rg_) else None
            if pts_ is not None:
                boxes_[k_] = (*pts_.min(0), *pts_.max(0))
        def anc_(k_):                                       # выделения, в которых лежит k_ (на них подпись можно)
            out_, p_ = set(), SM.parent(m, k_)
            while p_ is not None and p_ not in out_ and len(out_) < 30:
                out_.add(p_)
                p_ = SM.parent(m, p_)
            return out_
        lab_ = []                                           # подписи: выбранное первым, потом крупные слои
        for k_ in sorted(boxes_, key=lambda k2: (k2 != m._sel, SM.obj(m, m._regions[k2]))):
            b_ = boxes_[k_]
            if max(b_[2] - b_[0], b_[3] - b_[1]) < 8:        # участок в 5-10 раз меньше родителя: на этом масштабе —
                cx_, cy_ = (b_[0] + b_[2]) / 2, (b_[1] + b_[3]) / 2   # точка, чтобы не потерялся
                cv2.circle(mp, (int(cx_), int(cy_)), 5, SM.col(m, k_), 1, cv2.LINE_AA)
                cv2.circle(mp, (int(cx_), int(cy_)), 2, SM.col(m, k_), -1, cv2.LINE_AA)
                b_ = (cx_ - 6, cy_ - 6, cx_ + 6, cy_ + 6)
            mine_ = anc_(k_) | {k_}
            lab_.append((f"{SM.num(m, k_)} draft" if m._regions[k_].get('kind') == 'draft' else
                         f"{SM.num(m, k_)}  {SM.obj(m, m._regions[k_])}x", b_, SM.col(m, k_),
                         [q_ for k2, q_ in boxes_.items() if k2 not in mine_]))
        nav.labels(mp, lab_, [(inset[0] - mr[0], inset[1] - mr[1], inset[2] - mr[0], inset[3] - mr[1])]
                   if nv['big'] and inset is not None else [])
        pr_ = pending_ring()
        if pr_ is not None:                                 # новая область (угол 1 — столик): пунктир её цвета
            nav.poly(mp, pr_, SM.col(m, len(m._regions)), 2, dashed=True)
        for _n, k_, p_ in SM.numbered(m):                  # точки фокуса — точками цвета своей области
            if k_ >= 0 and (SM.hidden(m, k_) or m._regions[k_].get('kind') == 'group'):
                continue
            q_ = nav.to_px((p_['x'], p_['y']))
            if 0 <= q_[0] < mp.shape[1] and 0 <= q_[1] < mp.shape[0]:
                c_ = SM.col(m, k_ if k_ >= 0 else len(m._regions))
                cv2.circle(mp, (int(round(q_[0])), int(round(q_[1]))), 3, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(mp, (int(round(q_[0])), int(round(q_[1]))), 2, c_, -1, cv2.LINE_AA)
        nav.poly(mp, cam, (40, 230, 255), 2)                # камера здесь
        cx_, cy_ = nav.to_px(cam.mean(0))
        cv2.drawMarker(mp, (int(cx_), int(cy_)), (40, 230, 255), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
        if nv['drag']:                                      # новый участок: прямоугольник мышью
            a_, b_, c2_, d2_ = nv['drag']
            cv2.rectangle(mp, (int(a_), int(b_)), (int(c2_), int(d2_)), (255, 255, 255), 1, cv2.LINE_8)
            S0_, S1_ = nav.to_um((a_, b_)), nav.to_um((c2_, d2_))
            cv2.putText(mp, f"{size_txt(abs(S1_[0] - S0_[0]), abs(S1_[1] - S0_[1]))}, {m.objective}x"
                        + (", free box" if nv.get('free') else ""),
                        (int(min(a_, c2_)), int(max(b_, d2_)) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_8)
        nav.scale_bar(mp)
        hint_ = ("Map: click - go there, drag - new area, Ctrl+drag - free box, wheel - zoom, right drag - move, "
                 "double-click - open, Tab - camera"
                 if nv['big'] or lay.get('Wc') else "Map (Tab: bigger)  click - go, drag - new area, wheel - zoom")
        UI.panel(mp, 0, 0, UI.size(hint_, 0.42)[0] + 16, 26, 0.8)        # подсказка — на тёмной плашке
        UI.put(mp, hint_, 8, 18, 0.42, (225, 225, 225), 1)
        view[mr[1]:mr[3], mr[0]:mr[2]] = mp
        cv2.rectangle(view, (mr[0], mr[1]), (mr[2] - 1, mr[3] - 1), (90, 92, 104), 1)
        if camsmall is not None:                            # камера — в углу
            x0_, y0_ = inset[0], inset[1]
            hc_ = min(camsmall.shape[0], H - bh_ - y0_)
            view[y0_:y0_ + hc_, x0_:x0_ + mw_] = camsmall[:hc_]
            cv2.rectangle(view, (x0_, y0_), (x0_ + mw_ - 1, y0_ + hc_ - 1), (40, 230, 255), 1)
            UI.put(view, "Camera (Tab or click: bigger)", x0_ + 8, y0_ + 18, 0.42, UI.TEXT, 1)
            nv['inset'] = (x0_, y0_, x0_ + mw_, y0_ + hc_)

    def car_clamp(v):
        return int(max(0, min(car.get('content', 0) - H + 20, v)))

    def car_target(y):
        """Куда бросить карточку: (элемент car['order'], после него?) по высоте y, или None."""
        cards = [h_ for h_ in car.get('hits', []) if h_[4] == 'sel' and h_[5] < len(car.get('order', []))]
        if not cards:
            return None
        h_ = min(cards, key=lambda h: 0 if h[1] <= y <= h[3] else min(abs(y - h[1]), abs(y - h[3])))
        return car['order'][h_[5]], y > (h_[1] + h_[3]) / 2

    def car_items():
        """Карточки деревом слоёв: карта, под ней (с отступом) её участки большего увеличения и т. д.
        car['order'][позиция] -> индекс выделения (len(m._regions) — новое)."""
        items, order = [], []
        for k, depth in SM.tree(m):
            reg = m._regions[k]
            o_ = SM.obj(m, reg)
            x0_, y0_, x1_, y1_ = SM._rect(m, reg)
            mm_ = f"{(x1_ - x0_) / 1000:.2f} x {(y1_ - y0_) / 1000:.2f} mm"
            if reg.get('kind') == 'group':
                nk_ = sum(1 for r_ in m._regions if r_.get('parent') == reg['uid'])
                items.append(dict(title=SM.title(m, k), color=SM.col(m, k), depth=depth, chosen=k in car['multi'],
                                  hidden=bool(reg.get('hidden')), muted=SM.hidden(m, k),
                                  lines=[f"{nk_} inside, {mm_}"] + (["hidden on the maps"] if SM.hidden(m, k) else [])))
                order.append(k)
                continue
            if reg.get('kind') == 'draft':                  # маска: черновик кисти, не снимается
                hd_ = SM.hidden(m, k)
                items.append(dict(title=SM.title(m, k), obj="draft", color=SM.col(m, k), chosen=k in car['multi'],
                                  check=reg.get('export', False),
                                  confirm=car['confirm'] == k, depth=depth, hidden=bool(reg.get('hidden')),
                                  muted=hd_, lines=[mm_ + (", from the view (M)" if reg.get('src') else "") + (f", grid 1/{reg['sub']}" if reg.get('sub') else ""),
                                                    f"no objective: click 'draft' to scan it with {m.objective}x" if not hd_
                                                    else "hidden on the maps"]))
                order.append(k)
                continue
            nx_, ny_ = SM.frames_of(m, k)                  # кадры и время — для объектива этого выделения
            nfr_ = nx_ * ny_
            fr_ = (f"{nx_} x {ny_} = {nfr_} frames, " if ny_ > 1 or not reg.get('shape') else f"{nfr_} frames, ") + \
                TM.fmt(TM.estimate(m, nfr_, 'points' if m._use_fp and reg['pts'] else 'off', o_)[0])
            hd_ = SM.hidden(m, k)
            sh_ = shot_of(k)
            items.append(dict(title=SM.title(m, k), obj=f"{o_}x", color=SM.col(m, k), check=reg.get('export', False),
                              more=bool(sh_), chosen=k in car['multi'], thumbs=[] if hd_ else reg.get('thumbs', []), confirm=car['confirm'] == k, depth=depth,
                              alpha=map_alpha().get(reg['uid'], 1.0) if sh_ else None,
                              muted=o_ != m.objective or hd_, hidden=bool(reg.get('hidden')),
                              lines=([(f"scanned {sh_}"
                                       if sh_ else "not scanned yet") +
                                      ("" if sh_ or o_ == m.objective else f": put {o_}x in place"),
                                      fr_, mm_ + (",  brush" if reg.get('src') else "")])
                              if not hd_ else [mm_ + ", hidden on the maps"]))
            order.append(k)
        car['order'] = order                               # новая (только угол 1) — в списке после угла 2
        return items

    MENU = ["Open", "New", "Info", "Scans", "Queue", "View", "Layout", "Help"]
    from .util import ui_pref, set_ui_pref
    lay = dict(panes=ui_pref('panes'), list=ui_pref('list'), Wc=None, Wl=None, Hp=None)   # раскладка окна
    from .project import SCAN_HEAD as PICK_HEAD
    PICK_NOTE = "Click a star (or a row): the main scan - maps, view, export. A new scan becomes the main one. Esc: close"


    def menu_img(W_):
        sm_ = m.sample
        img_, menu['hits'] = UI.menu_bar(W_, f"{sm_.title}  [{sm_.project.uid}]" + (f"  —  {sm_.description}" if sm_.description else ""),
                                         MENU, menu['hover'])
        tb_, menu['thits'] = UI.tool_bar(W_, [], None, BUTTONS, menu['thover'])   # шаги работы убраны: путали
        return np.vstack([img_, tb_])

    def external_sync():
        """Окно просмотра — отдельный процесс: выделения, которые оно поменяло (sample.json), — сюда; его съёмки в
        очередь (<образец>/.queue_requests.jsonl) — в очередь; состояние очереди — ему (.queue_status.json).
        -> сообщение или None."""
        out_ = None
        vp_ = m.__dict__.get('_view_proc')
        if vp_ is not None and vp_.poll() is not None:     # окно просмотра закрылось: с ошибкой — сказать почему
            m.__dict__['_view_proc'] = None
            if vp_.returncode not in (0, None):
                try:
                    ln_ = [l_.strip() for l_ in open(vp_.error_log, encoding='utf-8', errors='replace') if l_.strip()]
                except (OSError, AttributeError):
                    ln_ = []
                out_ = "The view closed with an error: " + (ln_[-1] if ln_ else f"code {vp_.returncode}")
        sf_ = m.sample.file
        mt_ = os.path.getmtime(sf_) if os.path.exists(sf_) else 0
        if mt_ != car.get('ext_mt'):
            car['ext_mt'] = mt_
            from .layers import refresh_from_disk
            if refresh_from_disk(m):
                car['shot_t'] = -9
                m._map_win = None
                out_ = "Selections changed in the view: taken here"
            d_ = json.load(open(sf_, encoding='utf-8'))
            fo_ = d_.pop('focus', None)                    # просмотр сделал / выбрал выделение — оно и выбрано здесь
            if fo_:
                m._sel = next((i_ for i_, r_ in enumerate(m._regions) if r_['uid'] == fo_), m._sel)
                m.sample.data.pop('focus', None)
                m.sample.save()
                car['ext_mt'] = os.path.getmtime(sf_)
        rq_ = os.path.join(m.sample.folder, '.queue_requests.jsonl')
        if os.path.exists(rq_):                            # съёмки, поставленные в просмотре
            try:
                lines_ = open(rq_, encoding='utf-8').read().splitlines()
                os.remove(rq_)
            except OSError:
                lines_ = []
            uids_ = []
            for ln_ in lines_:
                try:
                    d2_ = json.loads(ln_)
                except ValueError:
                    continue
                if d2_.get('op') == 'scan':
                    uids_ += list(d2_.get('uids') or [])
                elif d2_.get('op') == 'quality' and d2_.get('folder') and os.path.isdir(d2_['folder']):   # J в просмотре
                    f2_ = d2_['folder']
                    q2_ = m.queue
                    if not any(j_.folder == f2_ and j_.kind in ('quality', 'stitch') for j_ in q2_.active()):
                        from .project import scan_uid as _su
                        r2_ = next((r_ for r_ in m._regions if r_.get('uid') == _su(f2_)), None)
                        same_ = r2_ is not None and SM.obj(m, r2_) == m.objective
                        q2_.quality(f2_, title="Quality stitch " + (SM.title(m, m._regions.index(r2_)) if r2_ else
                                                                    os.path.basename(f2_)), reshoot=same_)
                        out_ = "Quality stitch from the view queued" + ("" if same_ else " (other objective: no re-shooting)")
            if uids_:
                from .layers import refresh_from_disk
                refresh_from_disk(m)
                q_ = m.queue
                inq_ = {j_.uid for j_ in q_.active() if j_.kind == 'scan'}
                b_ = object()
                n_ = 0
                for u_ in dict.fromkeys(uids_):
                    if u_ not in inq_ and any(r_['uid'] == u_ for r_ in m._regions):
                        q_.scan(u_, batch=b_)
                        n_ += 1
                out_ = f"Queued from the view: {n_} scan{'s' if n_ != 1 else ''}" if n_ else out_
        qq_ = m.__dict__.get('_queue')                     # состояние очереди — просмотру (строка внизу)
        stp_ = os.path.join(m.sample.folder, '.queue_status.json')
        txt_ = qq_.summary() if qq_ is not None else ""
        if txt_ != car.get('ext_q') or (txt_ and time.monotonic() - car.get('ext_qt', 0) > 2):
            car['ext_q'], car['ext_qt'] = txt_, time.monotonic()
            try:
                with open(stp_, 'w', encoding='utf-8') as fo_:
                    json.dump(dict(summary=txt_, t=time.time()), fo_)
            except OSError:
                pass
        return out_

    def map_alpha(new=None):
        """Прозрачность картинок выделений (общая с просмотром: view.json образца). new — {uid: 0..1}."""
        p_ = os.path.join(m.sample.folder, 'view.json')
        try:
            d_ = json.load(open(p_, encoding='utf-8'))
        except (OSError, ValueError):
            d_ = {}
        if new is not None:
            al_ = dict(d_.get('alpha', {}))
            al_.update(new)
            d_['alpha'] = {k_: v_ for k_, v_ in al_.items() if v_ < 0.99}
            with open(p_, 'w', encoding='utf-8') as fo_:
                json.dump(d_, fo_, indent=1)
        return d_.get('alpha', {})

    def latest_scans():
        """Основные съёмки видимых выделений: [(папка, индекс выделения)] — снизу списка вверх (верх списка рисуется
        последним, поверх всех)."""
        from .project import Selection
        out = []
        for k_ in reversed(SM.order(m)):
            rg_ = m._regions[k_]
            if rg_.get('kind') == 'group' or SM.hidden(m, k_):
                continue
            sc_ = Selection(m.sample, rg_).latest
            if sc_ and os.path.exists(os.path.join(sc_.folder, 'join', 'preview_all.jpg')):
                out.append((sc_.folder, k_))
            elif sc_:                                      # последняя ещё не сшита — показать прежнюю сшитую
                ok_ = [x_.folder for x_ in reversed(Selection(m.sample, rg_).scans)
                       if os.path.exists(os.path.join(x_.folder, 'join', 'preview_all.jpg'))]
                if ok_:
                    out.append((ok_[0], k_))
        return out

    def shot_of(k):
        """Когда снято выделение k (последняя съёмка, 'дд.мм чч:мм') или None; кэш на секунду."""
        if time.monotonic() - car.get('shot_t', -9) > 1.0:
            from .project import Selection
            car['shot'] = {}
            for r_ in m._regions:
                if r_.get('kind') == 'group':
                    continue
                sc_ = Selection(m.sample, r_).latest
                if sc_:
                    b_ = os.path.basename(sc_.folder)          # 2026-11-22_123831
                    car['shot'][r_['uid']] = f"{b_[8:10]}.{b_[5:7]} {b_[11:13]}:{b_[13:15]}" if len(b_) >= 15 else b_
            car['shot_t'] = time.monotonic()
        return car['shot'].get(m._regions[k].get('uid'))

    def step_now():
        """Шаг работы: 0 — ничего не снято, 1 — есть обзор, но нет участков внутри, 2 — участки ждут съёмки,
        3 — всё снято (чертёж)."""
        if time.monotonic() - car.get('step_t', -9) < 1.0:
            return car.get('step', 0)
        from .project import Selection
        real_ = [r_ for r_ in m._regions if r_.get('kind') != 'group']
        shot = [r_ for r_ in real_ if Selection(m.sample, r_).latest]
        kids = [r_ for r_ in real_ if r_.get('parent')]
        car['step'] = 0 if not shot else (1 if not kids else (2 if any(r_ not in shot for r_ in kids) else 3))
        car['step_t'] = time.monotonic()
        return car['step']

    pl_file = {}

    def profile_lines(mp):
        """Линии профилей, закреплённые в просмотре (P): голубая и сиреневая, края — жёлтые засечки.
        Берутся из profile_lines.json образца (мкм столика)."""
        pth = os.path.join(m.sample.folder, 'profile_lines.json') if getattr(m, 'sample', None) else None
        if not pth or not os.path.exists(pth):
            return
        try:
            mt = os.path.getmtime(pth)
            if pl_file.get('t') != mt:
                with open(pth, encoding='utf-8') as fh:
                    pl_file.update(t=mt, d=json.load(fh))
            d_ = pl_file['d']
        except Exception:
            return
        ip = lambda q: (int(round(q[0])), int(round(q[1])))
        for k, ln in enumerate(d_.get('lines', [])):
            col = ((255, 230, 0), (255, 80, 255))[k % 2]
            E = nav.to_px(np.asarray(ln['ends'], float))
            cv2.line(mp, ip(E[0]), ip(E[1]), col, 1, cv2.LINE_8)
            u = (E[1] - E[0]) / max(float(np.hypot(*(E[1] - E[0]))), 1e-6)
            v = np.array([-u[1], u[0]])
            for q in (nav.to_px(np.asarray(ln['edges'], float)) if ln.get('edges') else []):
                cv2.line(mp, ip(q - 5 * v), ip(q + 5 * v), (0, 220, 255), 1, cv2.LINE_8)

    def lattice_lines(mp, w_, h_, rsel):
        """Сетка решётки образца на карте — тонко, полупрозрачно, красным: по ней встают квадраты новых областей.
        Выбрано выделение с подсеткой (2-5) — внутри него ещё и она (новая область в нём — из её клеток);
        черновик из клеток — его клетки."""
        from .layers import lattice_of, lattice_for, geom as _gm
        from . import lattice as LT
        L_ = (lattice_for(m, m._regions[m._sel], self_too=True)        # сетка выбранного в меню: своя разметка
              if m._sel is not None and m._sel < len(m._regions) else lattice_of(m))   # или родителя, или образца
        if L_ is None:
            return
        cs_ = nav.to_um(np.array([(0, 0), (w_, 0), (0, h_), (w_, h_)], float))
        uv_ = LT.to_uv(L_, cs_)
        lo_, hi_ = np.floor(uv_.min(0)).astype(int), np.ceil(uv_.max(0)).astype(int)
        px_per_cell = float(np.hypot(*(nav.to_px(LT.to_stage(L_, [(1, 0)])[0]) - nav.to_px(LT.to_stage(L_, [(0, 0)])[0]))))
        ov_ = np.zeros_like(mp)

        def lines(img, M_, a0, a1, b0, b1):
            for a in range(a0, a1 + 1):
                p_ = nav.to_px(LT.to_stage(L_, [(a / M_, b0 / M_), (a / M_, b1 / M_)]))
                cv2.line(img, tuple(np.int32(np.round(p_[0]))), tuple(np.int32(np.round(p_[1]))), (60, 60, 255), 1, cv2.LINE_8)
            for b in range(b0, b1 + 1):
                p_ = nav.to_px(LT.to_stage(L_, [(a0 / M_, b / M_), (a1 / M_, b / M_)]))
                cv2.line(img, tuple(np.int32(np.round(p_[0]))), tuple(np.int32(np.round(p_[1]))), (60, 60, 255), 1, cv2.LINE_8)
        if px_per_cell >= 5 and (hi_ - lo_).max() <= 400:
            lines(ov_, 1, lo_[0], hi_[0], lo_[1], hi_[1])
        own_ = np.zeros_like(mp)                                     # сетка каждого выделения из клеток — его цветом
        for k_r, r_ in enumerate(m._regions):
            if r_.get('kind') == 'group' or not r_.get('cells') or SM.hidden(m, k_r):
                continue
            fac_ = list(r_.get('factors') or [])
            N_ = LT.level(r_)
            M_ = N_ * int(r_.get('sub') or 1)
            if px_per_cell / M_ < 4:
                continue
            g_ = _gm(m, r_)
            b_ = np.array(g_.bounds).reshape(2, 2)
            pb_ = nav.to_px(np.array([b_[0], b_[1]]))
            if pb_[:, 0].max() < 0 or pb_[:, 1].max() < 0 or pb_[:, 0].min() > w_ or pb_[:, 1].min() > h_:
                continue
            uvr_ = LT.to_uv(L_, np.array([b_[0], (b_[1][0], b_[0][1]), b_[1], (b_[0][0], b_[1][1])]))
            a0, b0 = np.floor(uvr_.min(0) * M_).astype(int)
            a1, b1 = np.ceil(uvr_.max(0) * M_).astype(int)
            if max(a1 - a0, b1 - b0) > 600:
                continue
            fine_ = np.zeros_like(mp)
            lines(fine_, M_, a0, a1, b0, b1)
            fine_[fine_.any(2)] = SM.col(m, k_r)
            mk_ = np.zeros(mp.shape[:2], np.uint8)                   # только внутри выделения
            for ext_, hs_ in UI.geom_shapes(g_, nav.to_px):
                cv2.fillPoly(mk_, [np.int32(np.round(ext_))], 255, cv2.LINE_8)
                for h2_ in hs_:
                    cv2.fillPoly(mk_, [np.int32(np.round(h2_))], 0, cv2.LINE_8)
            cv2.copyTo(fine_, mk_, own_)
            cv2.copyTo(np.zeros_like(mp), mk_, ov_)                  # внутри — своя сетка, не общая
        ow_ = own_.any(2)
        if ow_.any():
            mp[ow_] = cv2.addWeighted(mp, 0.35, own_, 0.65, 0)[ow_]
        on_ = ov_.any(2)
        if on_.any():
            mp[on_] = cv2.addWeighted(mp, 0.6, ov_, 0.4, 0)[on_]

    def size_txt(w_, h_):
        return f"{w_ / 1000:.2f} x {h_ / 1000:.2f} mm" if max(w_, h_) >= 1000 else f"{w_:.0f} x {h_:.0f} um"

    def new_area(S0, S1, free=False):
        """Новый участок на текущем объективе. Решётка образца есть — клетками: квадраты литографии или подсетка
        выделения, в котором тянули (его клетки уровня N·n, строго внутри него); нет — прямоугольником (обзор).
        free (Ctrl): рамка ровно как нарисована, без клеток — например, вокруг кристалла; кадры — только те, что её
        касаются (рамка меньше кадра — один кадр по её середине)."""
        from .layers import lattice_of, lattice_for, geom as _gm
        from shapely.geometry import Point as _Pt
        L_ = lattice_of(m)
        if L_ is not None and not free:                                  # клетки — в решётке того, внутри кого тянут
            c_ = ((S0[0] + S1[0]) / 2, (S0[1] + S1[1]) / 2)
            hold_ = [r_ for r_ in m._regions if r_.get('kind') not in ('group', 'draft') and SM.obj(m, r_) <= m.objective
                     and _gm(m, r_).contains(_Pt(*c_))]
            if hold_:
                L_ = lattice_for(m, min(hold_, key=lambda r_: _gm(m, r_).area), self_too=True) or L_
            return new_area_cells(L_, S0, S1)
        from .layers import new_num, set_geom, box as _box
        from .errors import MicroControlError as _E
        x0_, x1_ = sorted((S0[0], S1[0]))
        y0_, y1_ = sorted((S0[1], S1[1]))
        fw_, fh_ = m.frame_size
        if free:
            return new_free_box(x0_, y0_, x1_, y1_)
        r_ = dict(uid=__import__('uuid').uuid4().hex, num=new_num(m), obj=m.objective, fw=fw_, fh=fh_, pts=[],
                  thumbs=[])
        set_geom(m, r_, _box(x0_, y0_, max(x1_, x0_ + 1), max(y1_, y0_ + 1)))
        if free:
            r_['free'] = True                              # рамка как нарисована: контур — она сама
        else:
            r_.pop('shape')                                # прямоугольник: хватает углов A, B
        from .layers import geom as _geom, clip_to_parent
        try:
            gp_, cut_ = clip_to_parent(m, r_)              # пирамидка: не шире того, в чём лежит
        except ValueError as e:
            return str(e)
        g_ = _geom(m, r_)
        m._regions.append(r_)
        SM.to_top(m, r_)
        m._sel = len(m._regions) - 1
        SM.persist(m)
        nx_, ny_ = SM.frames_of(m, m._sel)
        inside_ = f". Inside {SM.title(m, gp_)}: choose it and Scan area to shoot all its {m.objective}x boxes" \
            if free and gp_ is not None and SM.obj(m, m._regions[gp_]) != m.objective else ". Scan area shoots it"
        return (f"{SM.title(m, m._sel)} ({m.objective}x{', free box' if free else ''}): {size_txt(x1_ - x0_, y1_ - y0_)}, "
                f"{nx_ * ny_} frame{'s' if nx_ * ny_ != 1 else ''}" + (f", cut to the edge of {SM.title(m, gp_)}" if cut_ else "")
                + inside_)

    def reach_margin():
        """Запас рамки на каждую сторону (мкм): шаг мотора (столик встаёт только на шаги) + половина люфта
        (по последней сшитой съёмке образца: модель столика) + разброс шага столика. -> (мкм, откуда)."""
        ms_ = float(m.motor_step)
        half_bl, src_ = 0.0, "motor step"
        try:
            from .project import latest_scans as _ls
            from .snap import geometry
            fs_ = sorted(_ls(m.sample.folder), key=lambda f_: os.path.getmtime(os.path.join(f_, 'join', 'positions_all.csv'))
                         if os.path.exists(os.path.join(f_, 'join', 'positions_all.csv')) else 0)
            if fs_:
                g_ = geometry(fs_[-1])
                half_bl = 0.5 * max(g_.get('backlash_um') or [0.0])
                half_bl += g_.get('resid', 0.0) * g_['ms'] / max(1e-9, float(np.abs(g_['Mo']).max()))
                src_ = "motor step, half the backlash and the stage scatter of the last scan"
        except Exception:
            pass
        return ms_ + half_bl, src_

    def new_free_box(x0_, y0_, x1_, y1_):
        """Ctrl+тянуть: черновик-рамка (кристалл), ровно где нарисовали, с запасом на люфт и погрешность столика;
        края — на целых шагах мотора (столик туда доедет точно). Снимается объективом, что стоит, при Scan area
        (сам или все рамки внутри выбранного обзора)."""
        from .layers import new_num, set_geom, box as _box, clip_to_parent
        ms_ = float(m.motor_step)
        mg_, src_ = reach_margin()
        bx0, by0 = np.floor((x0_ - mg_) / ms_) * ms_, np.floor((y0_ - mg_) / ms_) * ms_
        bx1, by1 = np.ceil((x1_ + mg_) / ms_) * ms_, np.ceil((y1_ + mg_) / ms_) * ms_
        fw_, fh_ = m.frame_size
        r_ = dict(uid=__import__('uuid').uuid4().hex, num=new_num(m), obj=m.objective, free=True, fw=fw_, fh=fh_,
                  pts=[], thumbs=[], margin_um=round(mg_, 2))
        set_geom(m, r_, _box(bx0, by0, bx1, by1))
        try:
            gp_, cut_ = clip_to_parent(m, r_)
        except ValueError as e:
            return str(e)
        m._regions.append(r_)
        SM.to_top(m, r_)
        m._sel = len(m._regions) - 1
        SM.persist(m)
        inside_ = ""
        return (f"Selection {r_['num']} (free box, {m.objective}x): {size_txt(x1_ - x0_, y1_ - y0_)} drawn, "
                f"{size_txt(bx1 - bx0, by1 - by0)} with {mg_:.1f} um each side for the {src_}; edges on motor steps."
                f" Scan area shoots it with {m.objective}x.{inside_}")

    def new_area_cells(L_, S0, S1):
        from . import lattice as LT
        from .layers import geom as _geom, set_geom, new_num
        from shapely.geometry import Polygon as _Poly, Point as _Pt
        x0_, x1_ = sorted((S0[0], S1[0]))
        y0_, y1_ = sorted((S0[1], S1[1]))
        c_ = ((x0_ + x1_) / 2, (y0_ + y1_) / 2)
        par_ = [k_ for k_, r_ in enumerate(m._regions) if r_.get('sub') and r_.get('cells')
                and _geom(m, r_).contains(_Pt(*c_))]
        par_k = max(par_, key=lambda k2: LT.level(m._regions[k2]) * m._regions[k2]['sub']) if par_ else None
        if par_k is not None:
            pr_ = m._regions[par_k]
            n_ = pr_['sub']
            fac = (pr_.get('factors') or []) + [n_]
            inside = {tuple(c) for c in pr_['cells']}
        else:
            fac, inside, n_ = [], None, 1
        M_ = int(np.prod(fac)) if fac else 1
        box_ = np.array([(x0_, y0_), (x1_, y0_), (x1_, y1_), (x0_, y1_)])
        uv_ = LT.to_uv(L_, box_)
        a0, b0 = np.floor(uv_.min(0) * M_).astype(int) - 1
        a1, b1 = np.ceil(uv_.max(0) * M_).astype(int) + 1
        rect_ = _Poly(box_)
        cells = [(a, b) for a in range(a0, a1 + 1) for b in range(b0, b1 + 1)
                 if rect_.contains(_Pt(*LT.to_stage(L_, [((a + 0.5) / M_, (b + 0.5) / M_)])[0]))
                 and (inside is None or (a // n_, b // n_) in inside)]
        if not cells:
            return ("Drag over whole cells: " + (f"the 1/{n_} grid of {SM.title(m, par_k)}" if par_k is not None
                                                  else "lithography squares"))
        if len(cells) > 4000:
            return f"Too many cells ({len(cells)}): drag a smaller box"
        fw_, fh_ = m.frame_size
        r_ = dict(uid=__import__('uuid').uuid4().hex, num=new_num(m), obj=m.objective, fw=fw_, fh=fh_, pts=[],
                  thumbs=[], cells=[list(c) for c in sorted(cells)], factors=fac)
        set_geom(m, r_, LT.cells_geom(L_, cells, M_))
        from .layers import clip_to_parent
        try:
            gp_, cut_ = clip_to_parent(m, r_, par_k)       # пирамидка: только клетки целиком внутри большего
        except ValueError as e:
            return str(e)
        cells = [tuple(c) for c in r_['cells']]
        g_ = _geom(m, r_)
        m._regions.append(r_)
        SM.to_top(m, r_)
        m._sel = len(m._regions) - 1
        SM.persist(m)
        return (f"{SM.title(m, m._sel)} ({m.objective}x): {len(cells)} cell{'s' if len(cells) > 1 else ''} "
                f"of {'1/' + str(M_) if M_ > 1 else 'lithography squares'}, from {LT.address(*cells[0], fac)}"
                + (f"; cut to the edge of {SM.title(m, gp_)}" if cut_ else ""))

    def map_go(S):
        """Щелчок по карте: столик едет туда (Z — по точкам фокуса выделения под точкой, если есть)."""
        from .scan import focus_z
        k_ = SM.area_at(m, S[0], S[1])
        rg_ = m._regions[k_] if k_ is not None else None
        z_ = focus_z(rg_['pts'], S[0], S[1]) - m._off['z'] if rg_ and rg_.get('pts') and SM.obj(m, rg_) == m.objective else None
        SM.drive_to(m, float(S[0]), float(S[1]), z_, trace=fog_trace(m))   # по пути — кадры в туман карты
        return f"At X {S[0] - m._off['x']:.0f}, Y {S[1] - m._off['y']:.0f} um"

    def _chain(k_):
        """Выделение и его родители (от верхнего к нему): индексы."""
        out_, seen_ = [k_], {k_}
        while True:
            p_ = SM.parent(m, out_[0])
            if p_ is None or p_ in seen_:
                return out_
            out_.insert(0, p_); seen_.add(p_)

    def export_drawing():
        """Export: всё, что отмечено галочкой на карточках, — в <образец>/export/ (прежнее заменяется): чертёж для
        AutoCAD (DXF / DWG), снимки выделений и mixed.png (снимки с наложением по слоям списка; пиксель — самого точного снимка)."""
        nonlocal msg, msg_t
        from .drawing import export_sample
        msg_t = time.monotonic()
        ks_ = [k_ for k_ in SM.order(m) if m._regions[k_].get('kind') != 'group' and m._regions[k_].get('export', False)]
        if not ks_:
            msg = "Nothing ticked for the export: the box on a card puts it into the export"
            return
        res = busy("Export: drawing, pictures, mixed.png", lambda: export_sample(m, ks_))
        msg_t = time.monotonic()
        if 'e' in res and os.environ.get('MC_TEST'):
            print("Export failed:", res['e'])
        from .drawing import export_message
        msg = (str(res['e']) if str(res['e']).startswith("Not exported") else f"Export failed: {res['e']}") \
            if 'e' in res else export_message(res['r'], len(ks_))

    def busy(title, fn, plan=None, expect=None, on_stop=None, bg=None):
        """Долгое дело в потоке; в окне — полоса прогресса с кнопкой Stop (Esc) и строка состояния внизу.
        -> dict(r=результат | e=ошибка; Stop -> e = Stopped)."""
        Wl_ = W - SM.CAR_W
        oy_ = UI.MENU_H + UI.TOOL_H
        fs_ = 0.5
        bh_ = UI.size("Xg", fs_)[1] + 20

        def frame(stt):
            v0_ = lay.get('view0') if lay.get('view0') is not None else view   # раскладка по умолчанию
            pic = bg() if bg is not None else None              # своя подложка (скан: последний кадр)
            if pic is None:
                pic = (v0_[:, :Wl_].astype(np.float32) * 0.6).astype(np.uint8) if v0_ is not None else \
                    np.zeros((H, Wl_, 3), np.uint8)
            r_ = UI.progress_panel(pic, title, stt['stage'] if stt['stage'] != title else "",
                                   stt['frac'], stt['text'], time.monotonic() - stt['t0'], stt['left'], "",
                                   stopping=stt['stopping'])
            r2_ = UI.status_bar(pic, 0, H - bh_, Wl_, H, stt, scale=fs_)
            fr_ = v0_.copy() if v0_ is not None else np.zeros((H, W, 3), np.uint8)
            fr_[:, :Wl_] = pic
            hit_ = r_ if not stt['stopping'] else None
            hit_ = (hit_[0], hit_[1] + oy_, hit_[2], hit_[3] + oy_) if hit_ else \
                ((r2_[0], r2_[1] + oy_, r2_[2], r2_[3] + oy_) if r2_ else None)
            if lay['list'] == 'left':                       # раскладка: список слева — и здесь
                fr_ = np.ascontiguousarray(np.hstack([fr_[:, W - SM.CAR_W:], fr_[:, :W - SM.CAR_W]]))
                hit_ = (hit_[0] + SM.CAR_W, hit_[1], hit_[2] + SM.CAR_W, hit_[3]) if hit_ else None
            return np.vstack([menu_img(W), fr_]), hit_
        # клавиша держит камеру (live.hold) в этом потоке, а дело берёт кадры в своём: замок — делу на всё время,
        # иначе первый же кадр (автофокус, снимок, баланс белого, Prepare) ждёт вечно
        lk_, n_ = m.__dict__.get('_cam_lock'), 0
        while lk_ is not None:
            try:
                lk_.release()
                n_ += 1
            except RuntimeError:                            # этот поток его больше не держит
                break
        try:
            return UI.run_busy(WIN, title, fn, frame, plan=plan, expect=expect, wait_ms=60 if keys is None else 1,
                               on_stop=on_stop)
        finally:
            for _ in range(n_):
                lk_.acquire()
            cv2.setMouseCallback(WIN, on_mouse)

    def busy_live(title, fn, expect=None):
        """Действие с прибором в потоке, камера на паузе; в окне — полоса и время (видно, что не зависло).
        Stop (Esc): дело выходит на ближайшем шаге (столик доезжает начатое движение).
        -> результат; ошибка прибора или Stop -> MicroControlError наружу, как без полосы."""
        if qbusy():
            from .errors import SafetyError
            raise SafetyError('E707', QBUSY)
        if live is not None:
            live.paused = True

        def halt():
            m._abort = True                                 # скан / G-код: после текущей команды
        try:
            res_ = busy(title, fn, expect=expect, on_stop=halt)
        finally:
            if live is not None:
                live.paused = False
            m._abort = False
        if 'e' in res_:
            raise res_['e']
        return res_.get('r')

    def prepare_here(what='prepare'):
        """Prepare (кнопка, A): подготовка к снимкам здесь; D — дисторсия объектива. Полоса — в окне."""
        nonlocal msg, msg_t
        if qbusy():
            msg, msg_t = "The instrument is busy: " + QBUSY, time.monotonic()
            return
        if live is not None:
            live.paused = True
        try:
            res_ = busy("Prepare before the scan" if what == 'prepare' else f"Distortion of {m.objective}x",
                        (lambda: m.prepare()) if what == 'prepare' else (lambda: m.calibrate_distortion()))
        finally:
            if live is not None:
                live.paused = False
        if 'e' in res_:
            msg = f"{'Prepare' if what == 'prepare' else 'Distortion'} failed: {res_['e']}"
        else:
            msg = "   ".join(f"{k_}: {v_}" for k_, v_ in list(res_['r'].items()))
        msg_t = time.monotonic()

    def ask_ready(title, lines):
        """Условие посреди окна и кнопки Done / Cancel (Enter / Esc). -> True, если Done."""
        if keys is not None:                               # тесты: сразу «Готово»
            return True
        ev_ = {}

        def cb(ev, x, y, flags, *_):
            if ev == cv2.EVENT_MOUSEMOVE:
                ev_['hover'] = (x, y)
            elif ev == cv2.EVENT_LBUTTONDOWN:
                ev_['click'] = (x, y)
        cv2.setMouseCallback(WIN, cb)
        try:
            while True:
                base_ = np.vstack([menu_img(W), view])
                base_ = (base_.astype(np.float32) * 0.55).astype(np.uint8)
                rects_ = UI.confirm_panel(base_, title, lines, ("Done", "Cancel"), ev_.get('hover'))
                cv2.imshow(WIN, base_)
                k_ = cv2.waitKeyEx(30)
                if k_ in (13, 10):
                    return True
                if k_ == 27 or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    return False
                c_ = ev_.pop('click', None)
                if c_:
                    for nm_, r_ in rects_.items():
                        if r_[0] <= c_[0] <= r_[2] and r_[1] <= c_[1] <= r_[3]:
                            return nm_ == "Done"
        finally:
            cv2.setMouseCallback(WIN, on_mouse)

    wiz = {}                                              # мастер Prepare: шаги по очереди, окно живёт

    def prep_mem():
        return m._mem.data.setdefault('prepared', {}) if getattr(m, '_mem', None) is not None else \
            m.__dict__.setdefault('_prepared', {})

    def prepare_start():
        """Prepare: шаги по очереди (PREP_STEPS). Уже готово для этого объектива — спросить: заново или оставить."""
        nonlocal msg, msg_t
        if qbusy():
            msg, msg_t = "The instrument is busy: " + QBUSY, time.monotonic()
            return
        old_ = prep_mem().get(str(m.objective))
        wiz.clear()
        dev_ = (getattr(m, 'device', None) or {}).get('Serial', 'this microscope')
        if old_:
            wiz.update(i=-1, log=[], old=old_, title=f"Already prepared: {dev_}, {m.objective}x",
                       lines=[f"Saved {old_.get('time', '?')}: " + ", ".join(old_.get('done', [])) + ".",
                              f"Exposure {old_.get('exposure', '-')} ms, pause before a picture {old_.get('move_pause', '-')} s.",
                              "Overwrite: go through the steps again. Keep: use the saved ones."],
                       buttons=("Overwrite", "Keep"))
        else:
            wiz.update(i=0, log=[], done=[])
            wiz_step()

    def wiz_step():
        k_, t_, l_ = PREP_STEPS[wiz['i']]
        wiz.update(title=f"Prepare {m.objective}x - step {wiz['i'] + 1} of {len(PREP_STEPS)}: {t_}", lines=l_,
                   buttons=("Done", "Skip", "Cancel"), key=k_)

    def wiz_finish():
        nonlocal msg, msg_t
        rec = dict(time=time.strftime('%d.%m.%Y %H:%M'), done=wiz.get('done', []), exposure=m.exposure,
                   move_pause=m.move_pause, log=wiz.get('log', []))
        prep_mem()[str(m.objective)] = rec
        if getattr(m, '_mem', None) is not None:
            m._mem.save()
        msg, msg_t = f"Prepared {m.objective}x: " + ("; ".join(wiz.get('log', [])) or "nothing measured"), time.monotonic()
        wiz.clear()

    def wiz_action(name):
        """Кнопка мастера: Done — сделать шаг, Skip — дальше, Cancel — выйти; Overwrite / Keep — для старых данных."""
        nonlocal msg, msg_t
        if not wiz.get('title'):                           # мастер закрыт — кнопок нет
            return
        if name == "Cancel":
            wiz.clear()
            msg, msg_t = "Prepare cancelled", time.monotonic()
            return
        if name == "Keep":
            old_ = wiz['old']
            try:
                m.exposure = float(old_.get('exposure', m.exposure))
                m.move_pause = float(old_.get('move_pause', m.move_pause))
            except MicroControlError:
                pass
            wiz.clear()
            msg, msg_t = (f"Kept the preparation of {old_.get('time')}: exposure {m.exposure:g} ms, "
                          f"pause {m.move_pause:g} s, flat field as saved"), time.monotonic()
            return
        if name == "Overwrite":
            wiz.update(i=0, log=[], done=[])
            wiz_step()
            return
        k_ = wiz.get('key')
        if name == "Done":
            from . import prepare as PP
            jobs = {'lamp': lambda: f"lamp: {PP.light_drift(m):.1f} % in 3 s",
                    'kohler': lambda: "light by Koehler",
                    'dark': lambda: f"dark frame (level {m.dark_frame().get('Mean level')})",
                    'flat': lambda: (lambda e: (m.white_balance(), f"exposure {e[0]:.2f} ms ({100 * e[1]:.0f} %), white balance, "
                                     f"flat field (level {m.empty_frame().get('Mean level')})")[1])(PP.auto_exposure(m)),
                    'detail': lambda: (m.focus(), (lambda t: (setattr(m, 'move_pause', round(max(0.05, 1.3 * t), 2)),
                                                              f"focus, pause {round(max(0.05, 1.3 * t), 2)} s")[1])(PP.settle_time(m)))[1],
                    'distortion': lambda: (lambda r: f"distortion {r.info['corner_px']:.2f} px at the edge")(m.calibrate_distortion())}
            if live is not None:
                live.paused = True
            try:
                res_ = busy(wiz['title'], jobs[k_])
            finally:
                if live is not None:
                    live.paused = False
            if 'e' in res_:
                msg, msg_t = f"{k_}: {res_['e']} - fix it and press Done again, or Skip", time.monotonic()
                return
            wiz['log'].append(res_['r'])
            wiz['done'].append(k_)
            msg, msg_t = res_['r'], time.monotonic()
        wiz['i'] += 1
        if wiz['i'] >= len(PREP_STEPS):
            wiz_finish()
        else:
            wiz_step()

    def tbar_do(what):
        nonlocal msg, msg_t
        msg_t = time.monotonic()
        if what[0] == 'step':
            msg = STEP_HELP[what[1]]
        elif what[1] == "Prepare":
            prepare_start()
        elif what[1] == "Scan area":
            scan_here()
        elif what[1] == "Stitch":
            stitch_quality()
        elif what[1] == "New area":
            nv['big'] = m._nav_big = True
            msg = "Drag a rectangle on the map (it takes the objective in place). Click the map: go there"
        elif what[1] == "Export":
            export_drawing()
        elif what[1] == "Settings":
            m._panel_hidden = not getattr(m, '_panel_hidden', False)

    def set_layout(k_, v_):
        """Раскладка: panes (one | camera|map | map|camera) или list (right | left); помнится."""
        nonlocal msg, msg_t
        lay[k_] = v_
        set_ui_pref(k_, v_)
        if k_ == 'panes':
            nv['big'] = m._nav_big = False
            nav_here()
        msg, msg_t = f"Layout: {'list ' + v_ if k_ == 'list' else v_.replace('one', 'one view')}", time.monotonic()

    def menu_do(item):
        """Пункт меню: открыть / создать проект и образец, сведения, съёмки, просмотр, подсказка."""
        nonlocal msg, msg_t, show_help
        from . import dialogs as D
        from . import project as P_
        msg_t = time.monotonic()
        q_ = m.__dict__.get('_queue')
        if item == "Queue":                                # все задачи очереди: остановить одну или все
            js_ = (q_.jobs if q_ is not None else [])[-40:]
            if not js_:
                msg = "The queue is empty: Scan area (Ctrl+click several cards) and Stitch put jobs here"
                return
            from .jobs import tm as _qtm
            rows = [f"{j_.id:>3}  {j_.title[:34]:<34}  {j_.state:<8} {('cpu' if j_.lane == 'cpu' else 'instrument' if j_.lane == 'inst' else ''):<10} "
                    f"{_qtm(j_.elapsed())}  ~{_qtm(j_.remaining())} left  {(j_.note or (str(j_.error) if j_.error else ''))[:50]}"
                    for j_ in js_]
            act_ = [i_ for i_, j_ in enumerate(js_) if j_.active]
            r_ = D.choose(f"Queue: {q_.summary() or 'all done'}", rows, act_[0] if act_ else None, ("Stop", "Stop all"),
                          header="  #  Job                                 State    Lane       Time   Left")
            if r_ and r_[0] == "Stop" and r_[1] is not None:
                msg = f"Stopping: {js_[r_[1]].title}" if q_.stop(js_[r_[1]]) else f"{js_[r_[1]].title}: {js_[r_[1]].state}"
            elif r_ and r_[0] == "Stop all":
                q_.stop_all()
                msg = "Stopping all jobs of the queue"
            return
        if item in ("Open", "New") and q_ is not None and q_.active():
            msg = f"The queue is working on this sample ({q_.summary()}): wait or stop it (menu Queue), then {item}"
            return
        if live is not None:
            live.paused = True
        try:
            if item == "Open":
                ss = P_.all_samples(m.folder)
                head_, rows = P_.sample_table(ss)
                cur = next((i for i, x in enumerate(ss) if x == m.sample), None)
                r_ = D.choose("Open a sample (project id, project / sample)", rows, cur,
                              ("Open", "View in a new window", "By project id...", "New project...", "New sample..."),
                              header=head_)
                if r_ is None:
                    return
                if r_[0] == "View in a new window":                # посмотреть другой проект, не трогая столик
                    from .viewer import view_window
                    ls_ = P_.latest_scans(ss[r_[1]].folder)
                    if ls_:
                        view_window(ls_[0])
                        msg = f"{ss[r_[1]].title}: opened in a new window"
                    else:
                        msg = f"{ss[r_[1]].title}: nothing scanned yet"
                    return
                if r_[0] == "Open":
                    m.sample = ss[r_[1]]
                    msg = f"Opened {m.sample.title}: {len(m._regions)} selections"
                elif r_[0].startswith("By project id"):
                    u_ = D.ask("Open a project by its id", "Project id (4-5 characters, e.g. k3f9x)")
                    if not u_:
                        return
                    if P_.by_uid(u_, m.folder) is None:
                        msg = f"No project with id {u_}"
                        return
                    pj = P_.Project(u_, folder=m.folder)
                    sms = sorted(pj.samples, key=lambda x: x.data.get('updated', ''))
                    m.sample = sms[-1] if sms else pj.sample("Sample 1")
                    msg = f"Opened {m.sample.title} [{pj.uid}]: {len(m._regions)} selections"
                else:
                    return menu_do(r_[0].rstrip('.'))
            elif item == "New":                                    # New: что создать — из списка
                r_ = D.choose("New", ["Project   - a new project with its first sample",
                                      f"Sample    - a new sample (substrate) in {m.project.name}"], 1, ("Create",))
                if r_ is None or r_[1] is None:
                    return
                return menu_do("New project" if r_[1] == 0 else "New sample")
            elif item == "New project":
                a_ = D.new_project("New project", "Project name", "What the project is about")
                if not a_:
                    return
                pj = P_.Project(a_[0], a_[1], folder=m.folder)
                b_ = D.new_project(f"First sample of {pj.name}", "Sample name (the substrate on the stage)",
                                   "Short description of the sample") or ("Sample 1", "")
                m.sample = pj.sample(*b_)
                msg = f"New project {m.sample.title}"
            elif item == "New sample":
                a_ = D.new_project(f"New sample in {m.project.name}", "Sample name (the substrate on the stage)",
                                   "Short description of the sample")
                if not a_:
                    return
                m.sample = m.project.sample(*a_)
                msg = f"New sample {m.sample.title}: mark selections with 1 and 2"
            elif item == "Info":
                sm_ = m.sample
                from collections import Counter
                objs = Counter(r.get('obj') for r in m._regions)
                lines = [f"Project    {sm_.project.name}   (id {sm_.project.uid})", f"Sample     {sm_.name}",
                         "Selections " + (", ".join(f"{n} x {o}x" for o, n in sorted(objs.items())) or "none"),
                         f"Scans      {len(P_.all_scans(sm_.folder))}", f"Created    {sm_.data.get('created', '')}",
                         f"Folder     {sm_.folder}"]
                r_ = D.describe(sm_.title, lines, [("Project description", sm_.project.description),
                                                   ("Sample description", sm_.description)])
                if r_:
                    sm_.project.description, sm_.description = r_
                    msg = "Descriptions saved"
            elif item == "Scans" and m._sel is not None and m._sel < len(m._regions) and \
                    m._regions[m._sel].get('kind') not in ('group',) and scan_history(m._sel):
                k_ = m._sel
                rows, sc_ = scan_history(k_)
                cur_ = next((i_ for i_, x_ in enumerate(sc_) if x_ == P_.Selection(m.sample, m._regions[k_]).latest.folder), len(sc_) - 1)
                r_ = D.choose(f"Scans of {SM.title(m, k_)} (oldest first)", rows, cur_, ("View", "Make main"),
                              header="Date              Stitch    Frames  Seams, um  Picture, um         ",
                              extra="The main scan goes to the maps, the view and the export (by default - the newest)")
                if r_ and r_[0] == "View":
                    m._last_view, m._view_pick = sc_[r_[1]], True
                    open_view()
                elif r_ and r_[0] == "Make main":
                    m._regions[k_]['main'] = os.path.basename(sc_[r_[1]])
                    SM.persist(m)
                    car['shot_t'] = -9
                    msg = f"{SM.title(m, k_)}: main scan {os.path.basename(sc_[r_[1]])} (maps, view, export)"
            elif item == "Scans":
                lst = m.list()
                if not lst:
                    msg = "No scans of this sample yet: mark a selection (1, 2) and press S"
                    return
                rows = [l_ for l_ in repr(lst).split("\n")[2:-1:2]]
                r_ = D.choose(f"Scans of {m.sample.title}", rows, len(rows) - 1, ("View",))
                if r_:
                    m._last_view, m._view_pick = lst[r_[1]]['folder'], True
                    open_view()
            elif item == "View":
                open_view()
            elif item == "Layout":
                LAY_ = [("One view: camera big, the map in the corner (Tab swaps them)", 'panes', 'one'),
                        ("Two views: camera | map", 'panes', 'camera|map'),
                        ("Two views: map | camera", 'panes', 'map|camera'),
                        ("Selections list on the right", 'list', 'right'),
                        ("Selections list on the left", 'list', 'left')]
                cur_ = next((i_ for i_, (_t, k_, v_) in enumerate(LAY_) if k_ == 'panes' and lay['panes'] == v_), 0)
                r_ = D.choose("Layout of the windows", [("* " if lay[k_] == v_ else "  ") + t_ for t_, k_, v_ in LAY_],
                              cur_, ("Apply",), extra="* - now. Both windows (the view: the list side). Kept for next time.")
                if r_:
                    _t, k_, v_ = LAY_[r_[1]]
                    set_layout(k_, v_)
            elif item == "Help":
                show_help = not show_help
        except Exception as e:
            msg = f"{item}: {e}"
        finally:
            if live is not None:
                live.paused = False

    def pick_open(k_):
        """Список съёмок выделения k посреди окна: звезда — основная (карты, просмотр, экспорт)."""
        from .project import scan_table
        rows_, fs_, st_ = scan_table(m.sample.folder, m._regions[k_])
        if not rows_:
            return f"{SM.title(m, k_)}: no scans yet"
        pick.clear()
        pick.update(k=k_, rows=rows_, folders=fs_, star=st_, cur=st_, scroll=max(0, len(rows_) - 14),
                    title=f"Scans of {SM.title(m, k_)} (oldest first)")
        return f"{SM.title(m, k_)}: {len(rows_)} scan{'s' if len(rows_) != 1 else ''} - the star is the main one"

    def pick_do(c_):
        """Щелчок в списке съёмок: звезда / строка — основная, View — открыть, x / мимо / Esc — закрыть."""
        if c_ in ('close',):
            pick.clear()
            return None
        if c_ == 'panel' or not isinstance(c_, tuple):
            return None
        what_, i_ = c_
        k_, fs_ = pick['k'], pick['folders']
        if not (0 <= i_ < len(fs_)) or k_ >= len(m._regions):
            return None
        if what_ == 'view':
            pick.clear()
            m._last_view, m._view_pick = fs_[i_], True
            open_view()
            return f"View: {os.path.basename(fs_[i_])}"
        rg_ = m._regions[k_]
        if i_ == len(fs_) - 1:
            rg_.pop('main', None)                          # самая новая: основная и дальше — новая съёмка
        else:
            rg_['main'] = os.path.basename(fs_[i_])
        pick['star'] = pick['cur'] = i_
        SM.persist(m)
        car['shot_t'] = -9
        return f"{SM.title(m, k_)}: main scan {os.path.basename(fs_[i_])} (maps, view, export)"

    def scan_history(k_, fields=False):
        """Съёмки выделения k (старые -> новые): (строки для списка, папки) или None."""
        from .project import Selection as _Sel
        from . import quality as Q
        scs_ = _Sel(m.sample, m._regions[k_]).scans
        if not scs_:
            return None
        main_ = m._regions[k_].get('main')
        rows_, fs_ = [], []
        for x_ in scs_:
            b_ = os.path.basename(x_.folder)
            md_ = Q.stitched_as(x_.folder) or "not stitched"
            q_ = Q.info(x_.folder) if md_ == 'quality' else None
            try:
                nf_ = json.load(open(os.path.join(x_.folder, 'meta.json'), encoding='utf-8')).get('frames', '')
            except (OSError, ValueError):
                nf_ = ''
            sm_ = f"{q_['seam_um'][1]:.3f}" if q_ and q_.get('seam_um') else "-"
            pic_ = f"{q_['size_um'][0]:.1f} x {q_['size_um'][1]:.1f}" if q_ and q_.get('size_um') else "-"
            mk_ = "  <- main" if (main_ == b_ or (not main_ and x_ is scs_[-1])) else ""
            rows_.append((f"{b_[8:10]}.{b_[5:7]}.{b_[:4]} {b_[11:13]}:{b_[13:15]}:{b_[15:17]}", md_, nf_, sm_, pic_)
                         if fields else f"{b_[:17]:<17} {md_:<9} {str(nf_):>6}  {sm_:>9}  {pic_:<18}{mk_}")
            fs_.append(x_.folder)
        return rows_, fs_

    def inner_of(rs_):
        """Выделения текущего объектива, лежащие внутри выделения rs_ (по месту: вложенности в списке нет)."""
        from .layers import geom as _g
        try:
            G_ = _g(m, rs_).buffer(0.5)
        except Exception:
            return []
        out_ = []
        for k_, rg in enumerate(m._regions):
            if rg is rs_ or rg.get('kind') == 'group':
                continue
            try:
                inside_ = G_.contains(_g(m, rg))
            except Exception:
                continue
            if not inside_:
                continue
            if rg.get('kind') == 'draft' and rg.get('free'):   # рамка-черновик: выделением этого объектива
                draft_to_selection(k_)
            if rg.get('kind') != 'draft' and SM.obj(m, rg) == m.objective:
                out_.append(rg['uid'])
        return out_

    def scan_here():
        """Scan area, S: поставить в очередь съёмку выделений текущего объектива (выбранное; Ctrl+щелчком —
        несколько карточек); каждое после съёмки сшивается quick в фоне. Окно живёт, внизу — очередь (x — стоп)."""
        nonlocal msg, msg_t
        import threading
        from .util import Progress
        from .stitch import stitch as _stitch
        msg_t = time.monotonic()
        only_ = None
        qw_ = m.__dict__.get('_queue')
        if qw_ is not None and qw_.waiting_for().get(m.objective):   # съёмки ждали этот объектив — в работу
            n_ = qw_.release(m.objective)
            msg = f"Started {n_} scan{'s' if n_ != 1 else ''} that waited for {m.objective}x (the queue at the bottom)"
            if keys is not None and not getattr(m, '_test_async', False):
                qw_.wait()
            return
        if m._pending:                                     # новая область без угла 2 — она и выделена в меню
            msg = "Finish the new area first: key 2 sets its second corner (Del: drop it)"
            return
        mult_ = [k_ for k_ in car['multi'] if k_ < len(m._regions)]
        if len(mult_) > 1:                                 # Ctrl+щелчок по нескольким карточкам: все — в очередь
            only_ = []
            for k_ in mult_:
                rs_ = m._regions[k_]
                if rs_.get('kind') == 'group':
                    gu_ = {rs_['uid']}
                    for _ in range(8):
                        gu_ |= {rg['uid'] for rg in m._regions if rg.get('parent') in gu_}
                    only_ += [rg['uid'] for rg in m._regions if rg['uid'] in gu_ and rg.get('kind') not in ('group', 'draft')
                              and SM.obj(m, rg) == m.objective]
                elif rs_.get('kind') == 'draft':
                    draft_to_selection(k_)
                    if m._regions[k_].get('kind') != 'draft':
                        only_.append(rs_['uid'])
                elif SM.obj(m, rs_) != m.objective:
                    only_ += inner_of(rs_)
                else:
                    only_.append(rs_['uid'])
            if not only_:
                msg = f"None of the chosen selections is for {m.objective}x"
                return
        if not mult_[1:] and (m._sel is None or m._sel >= len(m._regions)):    # ничего не выбрано: одно выделение — его, иначе спросить
            own_ = [k_ for k_, rg in enumerate(m._regions) if SM.obj(m, rg) == m.objective
                    and rg.get('kind') not in ('group', 'draft')]
            if len(own_) > 1:
                msg = f"Choose what to scan: click its card in the list on the right ({len(own_)} areas for {m.objective}x)"
                return
            if own_:
                m._sel = own_[0]
        if mult_[1:]:
            pass
        elif m._sel is not None and m._sel < len(m._regions) and m._regions[m._sel].get('kind') == 'group':
            gu_, sub_ = {m._regions[m._sel]['uid']}, []    # группа: все выделения в ней (и во вложенных группах)
            for _ in range(8):
                for rg in m._regions:
                    if rg.get('parent') in gu_ and rg['uid'] not in gu_:
                        gu_.add(rg['uid'])
            only_ = [rg['uid'] for rg in m._regions if rg['uid'] in gu_ and rg.get('kind') not in ('group', 'draft')
                     and SM.obj(m, rg) == m.objective]
            if not only_:
                msg = f"{SM.title(m, m._sel)} has no area for {m.objective}x"
                return
        elif m._sel is not None and m._sel < len(m._regions):   # выбран черновик: он становится выделением
            rs_ = m._regions[m._sel]                          # этого объектива (пойдёт в экспорт) и снимается
            if rs_.get('kind') == 'draft':
                msg = draft_to_selection(m._sel)
                if m._regions[m._sel].get('kind') == 'draft':  # не вышло (объектив мал)
                    return
                only_ = [rs_['uid']]
            elif rs_.get('kind') != 'group':                  # выбранное выделение — только его
                if SM.obj(m, rs_) != m.objective:
                    in_ = inner_of(rs_)                        # обзор другого объектива: все области этого внутри
                    if not in_:
                        msg = (f"Selection {rs_['num']} is for {SM.obj(m, rs_)}x, the objective is {m.objective}x: "
                               f"switch it (O), choose another selection or Ctrl+drag boxes inside it on the map")
                        return
                    only_ = in_
                else:
                    only_ = [rs_['uid']]
        ks = [k_ for k_, rg in enumerate(m._regions) if SM.obj(m, rg) == m.objective
              and rg.get('kind') not in ('group', 'draft') and (not only_ or rg['uid'] in only_)]
        if not ks:
            msg = (f"Only selections without an objective here: click the 'draft' chip on a card to give it {m.objective}x, then Scan area"
                   if any(r_.get('kind') == 'draft' for r_ in m._regions) else
                   f"No area for {m.objective}x: drag a rectangle on the map (or switch the objective, O)")
            return
        q_ = m.queue
        inq_ = {j_.uid for j_ in q_.active() if j_.kind == 'scan'}
        new_ = [k_ for k_ in ks if m._regions[k_]['uid'] not in inq_]
        if not new_:
            msg = "Already in the queue (the bar at the bottom): x there stops a job"
            return
        nfr = sum(int(np.prod(SM.frames_of(m, k_))) for k_ in new_)
        fm_ = 'points' if m._use_fp else 'off'
        est = sum(TM.estimate(m, int(np.prod(SM.frames_of(m, k_))), fm_ if m._regions[k_]['pts'] else 'off')[0] for k_ in new_)
        L.event('MAN-011', f"Scan from the window, to the queue: {len(new_)} selection(s) of {m.objective}x, {nfr} frames, ~{TM.fmt(est)}.")
        b_ = object()
        for k_ in new_:
            q_.scan(m._regions[k_]['uid'], batch=b_)
        if keys is not None and not getattr(m, '_test_async', False):
            q_.wait()                                      # тесты по клавишам: как раньше — снято, потом дальше
        from .jobs import tm as _qtm
        msg = (f"Queued {len(new_)} scan{'s' if len(new_) > 1 else ''} of {m.objective}x ({nfr} frames, {TM.fmt(est)}), "
               f"each quick stitched in the background; the queue is done in ~{_qtm(q_.total_left())}. "
               "The bar at the bottom: x stops a job")

    def stitch_quality():
        """Stitch: точная сшивка (quality) последней съёмки выбранного выделения — в очередь: сшить (в фоне), проверить,
        доснять у прибора (размытые кадры, кадры с объектом без швов, непокрытые места рамки), ровный снимок рамки."""
        nonlocal msg, msg_t
        from . import quality as Q
        from .project import Selection as _Sel
        msg_t = time.monotonic()
        k_ = m._sel
        if k_ is None or k_ >= len(m._regions) or m._regions[k_].get('kind') in ('group', 'draft'):
            msg = "Choose a scanned selection (its card on the right), then Stitch"
            return
        sc_ = _Sel(m.sample, m._regions[k_]).latest
        if sc_ is None:
            msg = f"{SM.title(m, k_)} is not scanned yet: Scan area first"
            return
        same_ = SM.obj(m, m._regions[k_]) == m.objective
        q_ = m.queue
        if any(j_.folder == sc_.folder and j_.kind in ('quality', 'stitch') for j_ in q_.active()):
            msg = f"{SM.title(m, k_)}: its stitch is already in the queue (the bar at the bottom)"
            return
        q_.quality(sc_.folder, title=f"Quality stitch {SM.title(m, k_)}", reshoot=same_)
        if keys is not None and not getattr(m, '_test_async', False):
            q_.wait()
        from .jobs import tm as _qtm
        msg = (f"Quality stitch of {SM.title(m, k_)} queued (in the background" +
               (", re-shooting at the instrument if needed" if same_ else
                f"; the objective is not {SM.obj(m, m._regions[k_])}x: without re-shooting") +
               f"); the queue is done in ~{_qtm(q_.total_left())}")

    def open_view():
        """V: открыть последний скан в просмотре (кресты, квадраты, экспорт); закрыли — снова ручной режим."""
        nonlocal msg, msg_t
        from .viewer import view as _view
        from . import project as P_
        msg_t = time.monotonic()
        f_ = None
        lv = getattr(m, '_last_view', None)                # выбрали в меню Scans (любую версию)
        if lv and os.path.isdir(lv) and P_.sample_of(lv) == m.sample.folder and getattr(m, '_view_pick', False):
            f_ = lv
            m._view_pick = False
        elif m._sel is not None and m._sel < len(m._regions):   # выбранное выделение — его сшитая съёмка
            sel_ = P_.Selection(m.sample, m._regions[m._sel])
            sc_ = sel_.shown
            f_ = sc_.folder if sc_ else None
            from .live import running as _lrun
            if sel_.latest is not None and _lrun(sel_.latest.folder):   # снимается сейчас — её, кадры появляются на глазах
                f_ = sel_.latest.folder
            if f_ is None and sel_.latest is not None:     # снимается (или ещё не сшита): кадры — как они есть,
                from .jobs import n_frames as _nf          # просмотр дошивает их на ходу
                if _nf(sel_.latest.folder) > 0:
                    f_ = sel_.latest.folder
                else:
                    msg = f"Selection {m._regions[m._sel]['num']}: no frames yet - the view opens after the first ones"
                    return
            if f_ is None:
                msg = (f"Selection {m._regions[m._sel]['num']} is not scanned yet: S scans it"
                       + ("" if not P_.latest_scans(m.sample.folder) else " (Scans in the menu: the others)"))
                return
        stitched_ = lambda x_: os.path.exists(os.path.join(x_, 'join', 'preview_all.jpg'))
        if f_ is None and lv and os.path.isdir(lv) and P_.sample_of(lv) == m.sample.folder and stitched_(lv):
            f_ = lv                                        # только что сняли (и уже сшито)
        if f_ is None:                                     # иначе — самая новая сшитая съёмка образца
            ls = sorted(P_.latest_scans(m.sample.folder, stitched=True), key=os.path.basename)
            f_ = ls[-1] if ls else None
        if not f_:
            msg = "Nothing scanned in this sample yet: mark a selection (1, 2) and press S"
            return
        if keys is None and not getattr(m, '_view_inline', False):   # просмотр — своим окном, своим процессом:
            from .viewer import view_window                            # это окно живёт (очередь, камера, линейка)
            try:
                m.__dict__['_view_proc'] = view_window(f_)
                msg = (f"The view of {__import__('microcontrol.project', fromlist=['x']).scan_label(f_)} opens in its own "
                       "window; this one keeps working (the queue, the camera). Changes there come here by themselves")
            except Exception as e_:
                msg = f"The view did not open: {e_}"
            return
        if live is not None:
            live.paused = True
        err_ = None
        try:
            if view is not None:                           # пока грузится просмотр — видно, что идёт работа
                Wl_ = W - SM.CAR_W
                pic_ = view.copy()
                pic_[:, :Wl_] = (pic_[:, :Wl_].astype(np.float32) * 0.4).astype(np.uint8)
                UI.progress_panel(pic_[:, :Wl_], "Opening the view", __import__("microcontrol.project", fromlist=["x"]).scan_label(f_),
                                  None, "loading the stitched picture", 0.0, None, "")
                cv2.imshow(WIN, np.vstack([menu_img(W), pic_]))
                cv2.waitKey(1)
            _view(f_, fullscreen=fullscreen, _keys=keys)          # в тестах — те же нажатия
            try:                                           # просмотр мог менять выделения (объектив, удалить)
                from .layers import refresh_from_disk
                if refresh_from_disk(m):
                    car['shot_t'] = -9
                    m._map_win = None
                fo_ = json.load(open(m.sample.file, encoding='utf-8')).get('focus')   # просмотр сделал выделение —
                m.sample.data.pop('focus', None)           # оно и выбрано здесь
                if fo_:
                    m._sel = next((i_ for i_, r_ in enumerate(m._regions) if r_['uid'] == fo_), m._sel)
                    m.sample.save()
            except Exception as e_:
                L.debug('MAN-021', f"Selections from the view not taken: {e_}")
            try:                                           # решётку по столику просмотр мог записать в образец
                sm_ = getattr(m, '_sample', None)
                if sm_ is not None and sm_.data.get('lattice') is None:
                    L_disk_ = json.load(open(sm_.file, encoding='utf-8')).get('lattice')
                    if L_disk_ is not None:
                        sm_.data['lattice'] = L_disk_
            except Exception:
                pass
        except Exception as e:                             # просмотр закрылся с ошибкой — окно микроскопа живёт
            err_ = e
        finally:
            if live is not None:
                live.paused = False
        if keys is None:
            try:
                gone_ = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
            except cv2.error:
                gone_ = True
            if gone_:                                      # крестик просмотра закрыл и это окно — открыть снова
                open_window(WIN, fullscreen)
                cv2.setMouseCallback(WIN, on_mouse)
        msg = "Back in manual mode" if err_ is None else f"The view closed: {type(err_).__name__}: {err_} (details in the log)"

    def go_region(k):
        nonlocal msg
        msg = busy_live(f"Going to {SM.title(m, k)}", lambda: SM.goto_area(m, k))

    def delete_region(k):
        nonlocal msg
        if k < len(m._regions):
            note_brush(m._regions[k])
        msg = SM.delete_area(m, k)
        car['multi'] = []
        noted_done()
        L.event('MAN-003', msg + ".")
        car['confirm'] = None

    def commit(ed):
        """Применить набранное значение: кратно 5 мкм, от 5 до предела оси."""
        nonlocal msg, msg_t
        ax, buf = ed
        if not buf or buf == '.':
            return
        msg_t = time.monotonic()
        if ax == 'exp':
            v = float(buf)
            try:
                if live is not None:
                    with live.hold():
                        m.exposure = v
                else:
                    m.exposure = v
                msg = f"Exposure {m.exposure:g} ms"
            except MicroControlError as e:
                msg = str(e)
            return
        if ax in ('afs', 'afr'):                           # автофокус этого объектива (0 — авто)
            try:
                setattr(m, 'af_step' if ax == 'afs' else 'af_range', float(buf))
                a_, r_ = af_auto(m)
                msg = (f"Autofocus {m.objective}x: step {a_:g} um, window ±{r_:g} um (remembered for this objective; 0 = auto). "
                       f"Z motor step {m.motor_step:.2f} um, depth of field {m.depth_of_field:.1f} um")
            except MicroControlError as e:
                msg = str(e)
            return
        if ax == 'obj':
            v = int(float(buf))
            try:
                m.objective = v
                d0 = default_steps(m)
                steps['x'], steps['y'] = d0['x'], d0['y']
                msg = f"Objective {v}x: put it in place. Steps X, Y = one frame: {steps['x']:g}, {steps['y']:g} um"
            except MicroControlError as e:
                msg = str(e)
            return
        v = int(float(buf))
        lim = step_limit(m, ax)
        increment = 1 if ax == 'z' else 5
        max_step = math.floor(lim / increment) * increment
        step = max(increment, min(round(v / increment) * increment, max_step))
        steps[ax] = step
        msg = f"Step {ax.upper()} = {step:g} um"
        if v > lim:
            msg += f" (the limit is {lim:g} um)"
        elif ax == 'z' and not m._u2p(step):
            msg += f" (one motor step is {m.motor_step:.2f} um)"
        elif step != v and step < max_step and ax in 'xy':
            msg += " (rounded to a multiple of 5)"
        L.event('MAN-004', f"Manual step {ax.upper()} = {step:g} um.")

    rul = m.__dict__.setdefault("_ruler", dict(on=False, pts=[]))   # линейка (L): точки на образце, настоящие мкм
    back_v = None

    def ruler_text(a, b):
        d = np.asarray(b, float) - np.asarray(a, float)
        return f"{np.hypot(*d):.2f} um   (dX {d[0]:+.2f}, dY {d[1]:+.2f})"

    # история (Ctrl+Z / Ctrl+Y): после каждого действия — сравнить состояние с прежним
    hist = m.__dict__.setdefault('_history', History())

    H_FLAGS = ('turn_180', 'arrows_flip_x', 'arrows_flip_y', 'map_turn', 'one_side', '_use_fp')

    def h_misc():
        """Остальное, что меняют руками: свет, лазер, шаги, повороты и стрелки, подъезд, точки фокуса, прозрачность
        картинок, решётка образца."""
        return dict(light=m.light, laser=m.laser, steps=dict(steps), flags={k_: getattr(m, k_, None) for k_ in H_FLAGS},
                    alpha=map_alpha(), lattice=copy.deepcopy(m.sample.data.get('lattice')))

    def h_sig():
        strip = lambda r: None if r is None else {k_: v_ for k_, v_ in r.items() if k_ != 'thumbs'}
        return json.dumps([[strip(r) for r in m._regions], strip(m._pending), m._sel, getattr(m, '_area_seq', 0),
                           m._p, m.objective, m._off, gsel, h_misc()], sort_keys=True, default=str)

    def h_snap():
        return dict(regions=copy.deepcopy(m._regions), pending=copy.deepcopy(m._pending), sel=m._sel,
                    seq=getattr(m, '_area_seq', 0), p=dict(m._p), obj=m.objective, off=dict(m._off), gs=copy.deepcopy(gsel),
                    misc=h_misc())

    def h_label(a_, b_):
        if a_['p'] != b_['p'] and json.dumps(a_['regions'], default=str) == json.dumps(b_['regions'], default=str):
            s_ = m.motor_step
            return f"move to X {b_['p']['x'] * s_ - b_['off']['x']:.0f}, Y {b_['p']['y'] * s_ - b_['off']['y']:.0f}, " \
                   f"Z {b_['p']['z'] * s_ - b_['off']['z']:.0f} um"
        if a_['obj'] != b_['obj']:
            return f"objective {a_['obj']}x -> {b_['obj']}x"
        ma_, mb_ = a_.get('misc') or {}, b_.get('misc') or {}
        if json.dumps(a_['regions'], default=str) == json.dumps(b_['regions'], default=str) and ma_ != mb_:
            if ma_.get('light') != mb_.get('light'):
                return f"light {ma_.get('light')} -> {mb_.get('light')}"
            if ma_.get('laser') != mb_.get('laser'):
                return f"laser {ma_.get('laser')} -> {mb_.get('laser')} mW"
            if ma_.get('steps') != mb_.get('steps'):
                return "steps " + ", ".join(f"{a2_.upper()} {mb_['steps'][a2_]:g}" for a2_ in mb_['steps']
                                           if ma_['steps'].get(a2_) != mb_['steps'][a2_]) + " um"
            if ma_.get('alpha') != mb_.get('alpha'):
                return "transparency of pictures"
            if ma_.get('lattice') != mb_.get('lattice'):
                return "grid of the sample"
            for k_ in H_FLAGS:
                if (ma_.get('flags') or {}).get(k_) != (mb_.get('flags') or {}).get(k_):
                    return f"{k_.strip('_').replace('_', ' ')} -> {(mb_.get('flags') or {}).get(k_)}"
        if a_['sel'] != b_['sel'] and json.dumps(a_['regions'], default=str) == json.dumps(b_['regions'], default=str):
            return (f"chose Selection {m._regions[b_['sel']]['num']}" if b_['sel'] is not None and b_['sel'] < len(m._regions)
                    else "chose nothing")
        return (msg or "change").split(":")[0][:60]

    def h_restore(t):
        """Вернуть выделения, ноль, объектив и положение столика (переезд — тем же безопасным порядком,
        что Go: сначала вниз, потом X/Y отрезками, потом вверх)."""
        m._regions[:] = copy.deepcopy(t['regions'])
        m._pending, m._sel, m._area_seq = copy.deepcopy(t['pending']), t['sel'], t['seq']
        for fd_, s0_ in gsel0.items():                      # квадраты кисти в съёмках
            to_ = t.get('gs', {}).get(fd_, s0_)
            if grid_state(fd_) != to_:
                gp_ = os.path.join(fd_, 'grid.json')
                d_ = json.load(open(gp_, encoding='utf-8'))
                d_.update(to_)
                with open(gp_, 'w', encoding='utf-8') as f_:
                    json.dump(d_, f_, indent=1)
            gsel[fd_] = to_
        if t['obj'] != m.objective:
            m.objective = t['obj']
            d0 = default_steps(m)
            steps['x'], steps['y'] = d0['x'], d0['y']
        m._off.update(t['off'])
        mi_ = t.get('misc')
        if mi_:
            if mi_['light'] != m.light:
                m.light = mi_['light']
            if mi_['laser'] != m.laser:                     # лазер отмена только гасит: включить — клавишей I
                if not mi_['laser']:
                    m.laser = 0
            steps.update(mi_['steps'])
            for k_, v_ in (mi_.get('flags') or {}).items():
                if v_ is not None and getattr(m, k_, None) != v_:
                    setattr(m, k_, v_)
            if mi_.get('alpha') != map_alpha():
                p_ = os.path.join(m.sample.folder, 'view.json')
                try:
                    d_ = json.load(open(p_, encoding='utf-8'))
                except (OSError, ValueError):
                    d_ = {}
                d_['alpha'] = mi_.get('alpha') or {}
                with open(p_, 'w', encoding='utf-8') as f_:
                    json.dump(d_, f_, indent=1)
            if mi_.get('lattice') != m.sample.data.get('lattice'):
                if mi_.get('lattice') is None:
                    m.sample.data.pop('lattice', None)
                else:
                    m.sample.data['lattice'] = copy.deepcopy(mi_['lattice'])
                m.sample.save()
            m._map_win = None
        if t['p'] != m._p:
            s_ = m.motor_step
            tgt = {a_: t['p'][a_] * s_ - m._off[a_] for a_ in 'xyz'}
            if tgt['z'] < m.position.z:
                m.goto(z=tgt['z'])
            p0 = m.position
            nl = max(1, int(np.ceil(np.hypot(tgt['x'] - p0.x, tgt['y'] - p0.y) / (m.max_jump * 0.95))))
            for i_ in range(1, nl + 1):
                m.goto(x=p0.x + (tgt['x'] - p0.x) * i_ / nl, y=p0.y + (tgt['y'] - p0.y) * i_ / nl)
            while tgt['z'] > m.position.z + 0.5 * m.motor_step:
                m.goto(z=min(tgt['z'], m.position.z + m.max_move_z * 0.95))

    refresh_gs()
    h_last = (h_sig(), h_snap())
    arl = dict(on=True, pos=None, t=0.0, done=False, note='')   # сверка дополненной реальности с мозаикой (J)
    while True:
        if m._sel is not None and not 0 <= m._sel < len(m._regions):   # выбранное исчезло (удалили, отменили)
            m._sel = len(m._regions) - 1 if m._regions else None
        if any(not 0 <= k_ < len(m._regions) for k_ in car['multi']):
            car['multi'] = [k_ for k_ in car['multi'] if 0 <= k_ < len(m._regions)]
        if mdr.get('k') is not None and not 0 <= mdr['k'] < len(m._regions):
            mdr.pop('k', None); mdr.pop('d', None); mdr['on'] = False
        sg_ = h_sig()
        if sg_ != h_last[0]:                              # что-то сделано: в историю (с квадратами съёмок)
            refresh_gs()
            sg_ = h_sig()
            new_ = h_snap()
            if not qbusy() and not m.__dict__.pop('_h_ext', False):   # прибор водит очередь / пришло извне — не шаг
                hist.push(h_label(h_last[1], new_), h_last[1], new_)
            h_last = (sg_, new_)
        if time.monotonic() - car.get('sync_t', 0) > 1.0:   # кисть в просмотре -> выделения (раз в секунду)
            car['sync_t'] = time.monotonic()
            try:
                if time.monotonic() - car.get('bl_t', -99) > 5:   # люфт столика — из сшивки последних съёмок
                    car['bl_t'] = time.monotonic()
                    m._bl = stage_backlash()
                from .layers import sync_brush, idx as _ix
                keep_ = (m._regions[m._sel]['uid'] if m._sel is not None and m._sel < len(m._regions) else None,
                         [m._regions[k_]['uid'] for k_ in car['multi'] if k_ < len(m._regions)])
                r_ = sync_brush(m, make=False)       # новые выделения из клеток — только Enter в просмотре
                m._sel = _ix(m, keep_[0]) if keep_[0] else m._sel   # индексы сдвинулись — по uid
                car['multi'] = [k_ for k_ in (_ix(m, u_) for u_ in keep_[1]) if k_ is not None]
                if r_:
                    SM.persist(m)
                    msg, msg_t = r_, time.monotonic()
                    L.event('MAN-013', r_ + ".")
                refresh_gs()
                h_last = (h_sig(), h_snap())             # следствие выбранного в просмотре — не шаг Ctrl+Z
            except Exception as e:
                L.debug('MAN-013', f"Brush sync skipped: {e}")
        qb_ = qbusy()                                      # очередь снимает: камера — ей, в окне — её кадр
        if live is not None:
            if qb_ and not qbar['paused']:
                live.paused, qbar['paused'] = True, True
            elif not qb_ and qbar['paused']:
                live.paused, qbar['paused'] = False, False
        if time.monotonic() - car.get('ext_t', 0) > 0.5:  # просмотр (свой процесс) поменял образец — взять;
            car['ext_t'] = time.monotonic()                # его запросы в очередь — в очередь; ей — состояние очереди
            try:
                msg_x_ = external_sync()
                if msg_x_:
                    msg, msg_t = msg_x_, time.monotonic()
                if h_sig() != h_last[0]:                   # поменяло окно просмотра — шаг его истории, не этой
                    refresh_gs()
                    h_last = (h_sig(), h_snap())
            except Exception as e_:
                L.debug('MAN-022', f"External sync skipped: {e_}")
        qq_ = m.__dict__.get('_queue')
        if qq_ is not None:                                # закончилось в очереди: сообщение, карты заново
            for txt_, j_ in qq_.pop_events():
                msg, msg_t = txt_, time.monotonic()
                m.__dict__['_h_ext'] = True                # итог работы очереди — не шаг Ctrl+Z
                if j_.folder and j_.state == 'done':
                    m._last_view = j_.folder
                m._map_win = None
                car['shot_t'] = -9
                ch_ = m.__dict__.pop('_ori_changed', None)
                if ch_:
                    msg = (f"Camera directions fixed before the scan ({ch_}). If the arrow keys now go the wrong way: "
                           "[ - left / right, ] - up / down (remembered)")
        need = True
        if live is not None:                               # перерисовать: новый кадр, события мыши или раз в 0.3 с
            now_ = time.monotonic()                        # не чаще UI_FPS: камере 30 кадров/с не нужно
            need = bool(clicks) or now_ - last_t > 0.3 or (live.n != last_n and now_ - last_t >= 1.0 / UI_FPS)
        if need:
            prof.start()
            li_ = getattr(m, '_last_img', None) if qb_ else None
            img = li_ if li_ is not None else (img if qb_ and img is not None else
                                               (live.img if live is not None else m.image(0)))
            last_n, last_t = (live.n if live is not None else last_n), time.monotonic()
            if live is None and not qb_:                       # без потока камеры (проверки): туман — здесь
                fog_add(m, img)
            pos_ = (m._p['x'], m._p['y'], m.objective)          # дополненная реальность: сверка с мозаикой,
            if pos_ != arl['pos']:                              # когда столик постоял
                an_ = arl.get('anchor')                         # от места первой сверки (не от прошлой остановки:
                if an_ is None or (np.hypot(pos_[0] - an_[0], pos_[1] - an_[1]) *   # стрелками по кадру поправка копилась
                                   m.motor_step > 2 * max(m.frame_size) or pos_[2] != an_[2]):   # без конца)
                    m.__dict__['_ar_d'] = None                  # уехали далеко / другой объектив: заново
                    arl['anchor'] = pos_
                arl.update(pos=pos_, t=time.monotonic(), done=False)
            elif arl['on'] and not arl['done'] and time.monotonic() - arl['t'] > 0.4:
                arl['done'] = True
                try:
                    dP_, nm_ = ar_lock(m, img)
                    arl['note'] = (f"AR on the scan ({_ar_d(m)[0]:+.1f}, {_ar_d(m)[1]:+.1f} um)" if dP_ is not None
                                   else f"AR not checked: {nm_}")
                    if dP_ is not None and np.hypot(*dP_) > 0.3:
                        L.event('MAN-020', f"AR locked to the scan {nm_}: corrected by {dP_[0]:+.2f}, {dP_[1]:+.2f} um.")
                except Exception as e_:
                    arl['note'] = f"AR not checked: {e_}"
            if m._pending:                                 # карта и строка состояния — по новой или выбранной области
                A, B, zk = work(m._pending['A']), None, len(m._regions)
            elif m._sel is not None and SM.obj(m, m._regions[m._sel]) == m.objective and not SM.hidden(m, m._sel) \
                    and m._regions[m._sel].get('kind') != 'group':
                A, B, zk = work(m._regions[m._sel]['A']), work(m._regions[m._sel]['B']), m._sel
            else:                                          # выделение другого объектива — только контуром на карте
                A = B = zk = None
            if n % 4 == 0:                                     # резкость по тому же кадру (не дёргаем камеру ещё раз)
                sharp = round(F_.sharpness(img, m.focus_method), 1)
            n += 1
            W, H = (SW, SH) if fullscreen else window_size(WIN, (W, H + UI.MENU_H + UI.TOOL_H))
            H -= UI.MENU_H + UI.TOOL_H                         # сверху — меню и строка шагов
            rot = 0.0
            Wl = W - SM.CAR_W                                  # картинка — слева от карусели, по центру
            split_ = lay['panes'] in ('camera|map', 'map|camera')
            Wcam = Wl                                          # два вида: камера — во всю высоту (ширина — по кадру),
            if split_:                                         # карта — всё остальное
                Wcam = int(np.clip(H * img.shape[1] / max(1, img.shape[0]), 0.3 * Wl, 0.7 * Wl))
            lay['Wc'], lay['Wl'] = (Wcam if split_ else None), Wl
            zz_ = zm['z']
            if split_:                                         # два вида: кадр заполняет свою часть целиком (без полос)
                ih_, iw_ = img.shape[:2]
                if rot % 180:
                    ih_, iw_ = iw_, ih_
                r_ = (Wcam / iw_) / (H / ih_)
                zz_ = max(zz_, r_ if r_ > 1 else 1 / r_)
            to_v, back_v, i2v, v2i, cimg, kz, zm['c'] = live_map(m, img.shape, Wcam, H, zz_, zm['c'], rot)
            base = render(img, Wcam, H, kz, zm['c'], rot)
            view = np.zeros((H, W, 3), np.uint8)
            view[:, :Wcam] = base
            cam_poly_ = np.array([i2v(u_) for u_ in ((0, 0), (img.shape[1], 0), (img.shape[1], img.shape[0]),
                                                     (0, img.shape[0]))], float)
            rp_ = np.array([m.real_position.x, m.real_position.y])     # контуры выделений на кадре (строго:
            rad_ = 1.5 * max(m.frame_size) / max(1.0, zm['z']) + 2 * max(m.frame_size)   # с учётом люфта)
            ov_ = None
            clear_cam = []                                  # выделения этого объектива — чистые, вокруг — темнее
            from .layers import rings, geom as _gq
            for k_, rg_ in enumerate(m._regions):
                if rg_.get('kind') in ('group', 'draft') or SM.hidden(m, k_) or SM.obj(m, rg_) != m.objective:
                    continue
                x0_, y0_, x1_, y1_ = SM._rect(m, rg_)
                if x1_ < rp_[0] - rad_ or x0_ > rp_[0] + rad_ or y1_ < rp_[1] - rad_ or y0_ > rp_[1] + rad_:
                    continue
                d_ = mdr['d'] if mdr.get('k') == k_ else np.zeros(2)
                clear_cam += UI.geom_shapes(_gq(m, rg_), lambda A_: np.array([to_v(S_ + d_) for S_ in A_], float))
            if m._sel is not None and m._sel < len(m._regions) and not SM.hidden(m, m._sel):
                d_ = mdr['d'] if mdr.get('k') == m._sel else np.zeros(2)   # выбранное: внутри — как есть,
                cs_ = UI.geom_shapes(_gq(m, m._regions[m._sel]),             # снаружи — на 15 % темнее
                                     lambda A_: np.array([to_v(S_ + d_) for S_ in A_], float))
                if cs_:
                    clear_cam, cam_a = cs_, 0.15
                else:
                    cam_a = 0.35
            else:
                cam_a = 0.35
            if clear_cam:
                sub_ = view[:, :Wcam].copy()
                UI.tint_outside(sub_, clear_cam, (0, 0, 0) if cam_a < 0.3 else (40, 40, 40), cam_a)
                view[:, :Wcam] = sub_
            for k_, rg_ in enumerate(m._regions):
                if rg_.get('kind') == 'group' or SM.hidden(m, k_):
                    continue
                x0_, y0_, x1_, y1_ = SM._rect(m, rg_)
                if x1_ < rp_[0] - rad_ or x0_ > rp_[0] + rad_ or y1_ < rp_[1] - rad_ or y0_ > rp_[1] + rad_:
                    continue
                c_k = SM.col(m, k_)
                d_ = mdr['d'] if mdr.get('k') == k_ else np.zeros(2)
                for ring_ in rings(m, rg_):
                    pts = np.int32(np.round(np.array([to_v(S_ + d_) for S_ in ring_], float)))   # 1/16 px
                    if rg_.get('kind') == 'draft':                   # маска — пунктиром, без заливки
                        UI.dashed_poly(view[:, :Wcam], pts / 16.0, c_k if k_ == m._sel else SM.dim(c_k, 0.9),
                                       2 if k_ == m._sel else 1)
                    else:
                        cv2.polylines(view[:, :Wcam], [pts], True, c_k if k_ == m._sel else SM.dim(c_k, 0.85),
                                      2 if k_ == m._sel else 1, cv2.LINE_8)
            pr_ = pending_ring()
            if pr_ is not None:                             # новая область: от угла 1 до столика, пунктиром
                UI.dashed_poly(view[:, :Wcam], np.array([to_v(S_) for S_ in pr_], float),
                               SM.col(m, len(m._regions)), 2)
            if rul['on'] and rul['pts']:                    # линейка (точки — на образце: едут вместе с ним)
                a_ = to_v(rul['pts'][0])
                if len(rul['pts']) == 2:
                    b_, q2 = to_v(rul['pts'][1]), rul['pts'][1]
                elif hover[0] is not None and hover[0][0] < Wcam:
                    b_ = (hover[0][0] + 0.5, hover[0][1] + 0.5)
                    q2 = back_v(b_)
                else:
                    b_ = q2 = None
                UI.ruler(view, a_, b_, ruler_text(rul['pts'][0], q2) if q2 is not None else "")
            cm_ = np.zeros((H, Wcam), np.uint8)              # дополненная реальность — только на кадре камеры
            cv2.fillPoly(cm_, [np.int32(np.round(cam_poly_))], 255, cv2.LINE_8)
            view[:, :Wcam] = np.where(cm_[..., None] > 0, view[:, :Wcam], base)
            if m.laser:                                    # лазер включён — видно всегда
                t_ = f"LASER ON  {m.laser:g} mW   (I: off)"
                tw_ = UI.size(t_, 0.7, 2)[0]
                x_ = int((Wcam - tw_) / 2)
                cv2.rectangle(view, (x_ - 14, 12), (x_ + tw_ + 14, 52), (20, 20, 200), -1)
                UI.put(view, t_, x_, 42, 0.7, (255, 255, 255), 2)
            p = m.position
            sc = max(0.7, H / 1100)
            line = f"Sharpness {sharp:.0f}" + (f"    {arl['note']}" if arl['note'] and arl['on'] else "")
            if A and B:
                r, c, ny, nx = cell_of(m, A, B)
                if zk is not None and zk < len(m._regions):
                    nx, ny = SM.frames_of(m, zk)            # как на карточке и в съёмке
                    r = None if r is None or r >= ny else r
                    c = None if c is None or c >= nx else c
                    r = None if c is None else r
                line += f"    Selection {SM.num(m, zk)} ({m.objective}x): {nx} x {ny} = {nx * ny} frames"
            else:
                *_, (ci_, cj_), _z = _grid(m, A, B)
                b0c, b0r = (_z[0], _z[2]) if _z else (0, 0)
                line += (f"    Selection {SM.num(m, zk)}: corner 1 set, press 2 at the other corner" if A else
                         f"    Key 1: corner of a new selection ({m.objective}x)")   # номер кадра людям не нужен
            q = i2v(cimg + np.array([gd['u'], gd['v']]))                 # точка направляющих в окне

            # левая панель: координаты крупно, поля ввода в один столбец, точки фокуса цветом своей области
            FS = 0.55 * sc
            th0 = UI.size("Xg", FS)[1]
            LH = th0 + 16
            nav_early = nv['big'] or split_                  # большая карта — под панелью, а не панель в углу
            prof.mark('camera')
            if nav_early:
                draw_nav(view, Wl, th0 + 20, img.shape)
                prof.mark('map')
            rects = {}
            ops = []                                         # (что рисовать) — сначала размеры, потом панель
            x_l = 22
            lw = max(UI.size(t_, FS)[0] for t_ in ("Step X", "Exposure", "Objective")) + 14
            vw = UI.size("000000", FS)[0] + 16
            y = 22 + int(th0 * 1.6)
            big = 0.8 * sc
            pos = [("X", f"{p.x:.1f}"), ("Y", f"{p.y:.1f}"), ("Z", f"{p.z:.1f}")]
            xx = x_l
            for a_, v_ in pos:
                ops.append(('t', a_, xx, y, big, UI.DIM, 1))
                xx += UI.size(a_ + " ", big)[0]
                ops.append(('t', v_, xx, y, big, UI.TEXT, 2))
                xx += UI.size(v_ + "   ", big, 2)[0]
            u_ = "um" + ("  (zero moved)" if any(m._off.values()) else "")
            xu = xx - UI.size("  ", big, 2)[0]
            ops.append(('t', u_, xu, y, FS, UI.DIM, 1))
            right = xu + UI.size(u_, FS)[0]
            y += 12
            ops.append(('l', y))
            y += LH + 2
            cur_ = dict(steps, exp=m.exposure, obj=m.objective, afs=af_auto(m)[0], afr=af_auto(m)[1])
            for ax_ in FIELDS:
                lab_ = {'exp': "Exposure", 'obj': "Objective", 'afs': "AF step", 'afr': "AF window ±"}.get(ax_, f"Step {ax_.upper()}")
                unit_ = {'exp': "ms", 'obj': "x"}.get(ax_, "um")
                hint_ = {'z': "Shift: x5", 'obj': "O: next", 'afs': f"{m.objective}x" + ("" if m.af_step else ", auto"),
                         'afr': f"{m.objective}x" + ("" if m.af_range else ", auto")}.get(ax_, "")
                act = bool(edit and edit[0] == ax_)
                val = (edit[1] + "_") if act else f"{cur_[ax_]:g}"
                bx = (x_l + lw, y - th0 - 6, x_l + lw + vw, y + 6)
                ops.append(('t', lab_, x_l, y, FS, UI.DIM, 1))
                ops.append(('b', bx, act))
                ops.append(('t', val, bx[2] - 8 - UI.size(val, FS)[0], y, FS, UI.ACCENT if act else UI.TEXT, 1))
                ops.append(('t', unit_, bx[2] + 8, y, FS, UI.DIM, 1))
                if hint_:
                    ops.append(('t', hint_, bx[2] + 14 + UI.size("um", FS)[0] + 10, y, FS * 0.9, (120, 120, 120), 1))
                rects[ax_] = (x_l - 6, bx[1], bx[2] + 40, bx[3])
                y += LH + 4
            # флажок: Z каждого кадра — по точкам фокуса
            cb = (x_l, y - th0 - 2, x_l + th0 + 4, y + 2)
            ops.append(('c', cb, m._use_fp))
            t_ = "Z follows the focus points"
            ops.append(('t', t_, cb[2] + 10, y, FS, UI.TEXT if m._use_fp else UI.DIM, 1))
            pass
            rects['usefp'] = (x_l - 6, cb[1] - 4, cb[2] + 40 + UI.size(t_, FS)[0], cb[3] + 4)
            right = max(right, rects['usefp'][2], rects['obj'][2] + 60)
            y += LH
            # флажок: люфт — подъезд всегда с одной стороны (помнится для прибора)
            os_ = bool(m.one_side)
            cb = (x_l, y - th0 - 2, x_l + th0 + 4, y + 2)
            ops.append(('c', cb, os_))
            t_ = f"One-side approach (backlash, overshoot {m.overshoot} steps)"
            ops.append(('t', t_, cb[2] + 10, y, FS, UI.TEXT if os_ else UI.DIM, 1))
            pass
            rects['oneside'] = (x_l - 6, cb[1] - 4, cb[2] + 40 + UI.size(t_, FS)[0], cb[3] + 4)
            right = max(right, rects['oneside'][2])
            y += LH
            xb_ = x_l                                      # редкие дела прибора — кнопками (их буквы — у просмотра)
            for nm_, kc_ in (("Prepare", 'a'), ("Distortion", 'd'), ("Zero here", '0'), ("Steps = frame", 'g'),
                             ("Check directions", 'o'), ("Arrows X", '['), ("Arrows Y", ']'), ("Map 90 deg", '\\'),
                             ("Clear fog", '#')):
                bw_ = UI.size(nm_, FS * 0.9)[0] + 14
                if xb_ + bw_ > max(right, x_l + 330) and xb_ > x_l:
                    xb_ = x_l
                    y += LH
                bx_ = (xb_, y - th0 - 4, xb_ + bw_, y + 5)
                ops.append(('b', bx_, False))
                ops.append(('t', nm_, xb_ + 7, y, FS * 0.9, UI.TEXT, 1))
                rects['act:' + kc_] = bx_
                xb_ += bw_ + 6
                right = max(right, xb_)
            y += LH + 2
            lt_ = "Light on"                               # свет и лазер: подписи одинаковые, состояние — цветом
            ops.append(('t', lt_, x_l, y, FS, UI.TEXT if m.light else UI.DIM, 1))
            lz_ = f"Laser on {m.laser:g} mW" if m.laser else "Laser on"
            xz_ = x_l + UI.size(lt_ + "      ", FS)[0]
            ops.append(('t', lz_, xz_, y, FS, (80, 80, 255) if m.laser else UI.DIM, 1 if not m.laser else 2))
            right = max(right, xz_ + UI.size(lz_, FS, 2)[0])
            y += LH
            fp = SM.numbered(m)
            ops.append(('l', y - LH // 2 + 4))
            y += 6
            if fp:
                ops.append(("t", "Focus points", x_l, y, FS, UI.DIM, 1))
                y += LH
                per = 4                                      # таблица: столбцы равной ширины, Z — по правому краю
                fz = FS * 0.95
                vals = [f"{f_['z'] - m._off['z']:.1f}" for _n, _k, f_ in fp[:12]]
                nw = max(UI.size(str(n_), fz)[0] for n_, _k, _f in fp[:12])
                vw_ = max(UI.size(v_, fz)[0] for v_ in vals)
                colw = 14 + nw + 10 + vw_ + 22
                for r0 in range(0, min(len(fp), 12), per):
                    for c0_, (n_, k_, f_) in enumerate(fp[r0:r0 + per]):
                        xx = x_l + c0_ * colw
                        c_ = SM.col(m, k_ if k_ >= 0 else len(m._regions))
                        ops.append(('d', (xx + 5, y - th0 // 2), c_))
                        ops.append(('t', str(n_), xx + 14, y, fz, UI.DIM, 1))
                        v_ = vals[r0 + c0_]
                        ops.append(('t', v_, xx + 14 + nw + 10 + vw_ - UI.size(v_, fz)[0], y, fz, UI.TEXT, 1))
                    right = max(right, x_l + min(per, len(fp)) * colw - 22)
                    y += LH - 2
                if len(fp) > 12:
                    ops.append(('t', f"... {len(fp) - 12} more", x_l, y, FS * 0.9, UI.DIM, 1))
                    y += LH - 2
            if zm['z'] > 1:
                ops.append(('t', f"Zoom x{zm['z']:.2f}", x_l, y, FS * 0.9, UI.ACCENT, 1))
                y += LH
            hidden_ = getattr(m, '_panel_hidden', False)
            hov_ = lambda r_: hover[0] is not None and r_[0] <= hover[0][0] <= r_[2] and r_[1] <= hover[0][1] <= r_[3]
            if hidden_:                                      # панель спрятана: «ручка» у левого края
                tw_ = int(26 * sc)
                th_ = int(72 * sc)
                rects.clear()                                # невидимые поля не нажимаются
                rects['panel'] = (0, 12, tw_, 12 + th_)
                nv['pbox'] = rects['panel']
                hv_ = hov_(rects['panel'])
                UI.panel(view, 0, 12, tw_, 12 + th_, 0.9)
                UI.round_rect(view, -8, 12, tw_, 12 + th_, 8, UI.ACCENT if hv_ else UI.LINE, 1)
                if hv_:
                    view[13:12 + th_, 0:tw_ - 1] = np.clip(view[13:12 + th_, 0:tw_ - 1].astype(int) + 14, 0, 255)
                UI.chevron(view, tw_ // 2 - 1, 12 + th_ // 2, int(7 * sc), +1, UI.ACCENT if hv_ else UI.TEXT, 2)
                if hv_:                                      # подсказка рядом с ручкой
                    UI.toast(view, "Show the panel (P)", tw_ + 90 * sc, 12 + th_ // 2 + 6, FS * 0.9, 'info')
                line = f"X {p.x:.1f}  Y {p.y:.1f}  Z {p.z:.1f} um    " + line
                ops = []
            else:
                UI.panel(view, 8, 8, right + 44, y - LH + 16, 0.82, UI.LINE)
                nv['pbox'] = (8, 8, right + 44, y - LH + 16)  # щелчки по панели — ей, не карте под ней
                bs_ = int(26 * sc)
                rects['panel'] = (right + 34 - bs_, 16, right + 34, 16 + bs_)    # кнопка «спрятать»
                hv_ = hov_(rects['panel'])
                x0b, y0b, x1b, y1b = rects['panel']
                UI.round_rect(view, x0b, y0b, x1b, y1b, 6, (78, 80, 90) if hv_ else (46, 47, 54), -1)
                UI.round_rect(view, x0b, y0b, x1b, y1b, 6, UI.ACCENT if hv_ else UI.LINE, 1)
                UI.chevron(view, (x0b + x1b) // 2 + 1, (y0b + y1b) // 2, int(6 * sc), -1,
                           UI.ACCENT if hv_ else UI.TEXT, 2)
            for op in ops:
                if op[0] == 't':
                    UI.put(view, op[1], op[2], op[3], op[4], op[5], op[6])
                elif op[0] == 'l':
                    cv2.line(view, (x_l, op[1]), (right, op[1]), UI.LINE, 1)
                elif op[0] == 'b':
                    (b0, b1, b2, b3), act = op[1], op[2]
                    cv2.rectangle(view, (b0, b1), (b2, b3), (14, 14, 16), -1)
                    cv2.rectangle(view, (b0, b1), (b2, b3), UI.ACCENT if act else UI.LINE, 1)
                elif op[0] == 'c':
                    (b0, b1, b2, b3), on = op[1], op[2]
                    cv2.rectangle(view, (b0, b1), (b2, b3), UI.ACCENT if on else UI.LINE, 1 if not on else -1)
                    if on:
                        cv2.line(view, (b0 + 3, (b1 + b3) // 2), ((b0 + b2) // 2 - 1, b3 - 4), (20, 20, 20), 2)
                        cv2.line(view, ((b0 + b2) // 2 - 1, b3 - 4), (b2 - 3, b1 + 3), (20, 20, 20), 2)
                elif op[0] == 'd':
                    cv2.circle(view, op[1], 5, op[2], -1, cv2.LINE_AA)
            Wl = W - SM.CAR_W
            prof.mark('panel')
            # низ: строка состояния; над ней — сообщение
            bh_ = th0 + 20
            UI.status_bar(view, 0, H - bh_, Wl, H, None,                   # «Ready» и факты; дело — в busy
                          [f"Objective {m.objective}x"] + [t_.strip() for t_ in line.split("    ") if t_.strip()]
                          + ["H: keys"], scale=FS)                 # объектив программы — всегда на виду
            qh_ = UI.queue_height(qq_, FS) if qq_ is not None else 0     # очередь — над строкой состояния
            qbar['rects'] = UI.queue_panel(view, 0, H - bh_, Wl, qq_, FS) if qh_ else []
            qbar['h'] = qh_
            prof.mark('bars')
            if not nav_early:
                draw_nav(view, Wl, bh_ + qh_, img.shape)
                prof.mark('map')
            lay['Hp'] = H - bh_ - qh_
            if split_:                                       # граница видов
                if lay['panes'] == 'map|camera':             # карта слева, камера справа
                    Hp_ = lay['Hp']
                    view[:Hp_, :Wl] = np.hstack([view[:Hp_, Wcam:Wl], view[:Hp_, :Wcam]])
                    xb_ = Wl - Wcam
                else:
                    xb_ = Wcam
                cv2.line(view, (xb_, 0), (xb_, lay['Hp']), (90, 92, 104), 2)
            if show_help:                                    # H: все клавиши — поверх всего (переносом, влезает в окно)
                kh_ = [h_.partition(": ") for h_ in HELP]
                UI.help_panel(view[:, :Wl], [(k_, d_) for k_, _s, d_ in kh_])
            m.__dict__['_last_msg'] = msg
            wf_ = m.__dict__['_queue'].waiting_for() if m.__dict__.get('_queue') is not None else {}
            if wf_:                                          # съёмка ждёт другой объектив: видно сразу, пока ждёт
                o_, n_ = sorted(wf_.items(), key=lambda t_: -t_[1])[0]
                UI.toast(view, f"{n_} scan{'s' if n_ > 1 else ''} in the queue wait{'' if n_ > 1 else 's'} for {o_}x, "
                               f"now {m.objective}x: put {o_}x in place and set it (O) - the scan starts by itself",
                         Wl / 2, 70, FS, 'info', Wl - 60)
            if msg and time.monotonic() - msg_t < 5:
                UI.toast(view, msg, Wl / 2, H - bh_ - qh_ - 20, FS, 'err' if msg.startswith('[E') else 'ok', Wl - 60)
            if prof.on:                                      # `: мс на кадр по этапам
                UI.prof_panel(view, prof, Wl - 10, 60)
            prof.mark('toasts')
            items = car_items()
            selk_ = len(m._regions) if m._pending else m._sel
            sel_ = car['order'].index(selk_) if selk_ in car['order'] else None
            hx_ = hover[0]
            hv_ = (hx_[0] - (W - SM.CAR_W), hx_[1]) if hx_ is not None and hx_[0] >= W - SM.CAR_W else None
            pan_, hits_, cont_ = SM.carousel(H, items, sel_, car['scroll'], title="Selections", hover=hv_,
                                             tools=car_tools())
            car['hits'], car['content'], car['W'] = hits_, cont_, W
            car['scroll'] = car_clamp(car['scroll'])
            dr_ = car.get('drag')
            tg_ = car_target(dr_['y']) if dr_ is not None else None
            if dr_ is not None and tg_ is not None and tg_[0] != dr_['src'] and SM.reorder_ok(m, dr_['src'], tg_[0]):
                # перетаскивание: список сразу такой, каким станет, если отпустить сейчас (карточка — в рамке)
                ranks_ = [r_.get('rank') for r_ in m._regions]
                order0_ = list(car['order'])
                SM.reorder(m, dr_['src'], tg_[0], tg_[1], save=False)
                items2_ = car_items()
                order2_ = list(car['order'])
                for r_, rk_ in zip(m._regions, ranks_):          # как было: решает отпускание
                    if rk_ is None:
                        r_.pop('rank', None)
                    else:
                        r_['rank'] = rk_
                car['order'] = order0_
                sel2_ = order2_.index(selk_) if selk_ in order2_ else None
                pan_, hits2_, _c = SM.carousel(H, items2_, sel2_, car['scroll'], title="Selections", tools=car_tools())
                sh_ = next((h for h in hits2_ if h[4] == 'sel' and h[5] < len(order2_) and order2_[h[5]] == dr_['src']), None)
                if sh_ is not None and sh_[3] > 39:
                    cv2.rectangle(pan_, (sh_[0], max(39, sh_[1])), (sh_[2] - 1, sh_[3] - 1), (90, 220, 90), 2)
                dr_ = None                                     # линия-указатель не нужна: видно и так
            if dr_ is not None:                                # перетаскивание: куда ляжет карточка
                src_h = next((h for h in hits_ if h[4] == 'sel' and h[5] < len(car['order'])
                              and car['order'][h[5]] == dr_['src']), None)
                if src_h is not None:
                    pan_[max(39, src_h[1]):max(39, src_h[3]), src_h[0]:src_h[2]] //= 2
                th_ = next((h for h in hits_ if h[4] == 'sel' and h[5] < len(car['order'])
                            and tg_ is not None and car['order'][h[5]] == tg_[0]), None)
                if th_ is not None:
                    ok_ = SM.reorder_ok(m, dr_['src'], tg_[0])
                    yl_ = th_[3] - 1 if tg_[1] else th_[1] + 1
                    cv2.line(pan_, (th_[0], yl_), (th_[2], yl_), (90, 220, 90) if ok_ else (80, 80, 230), 3)
            view[:, W - SM.CAR_W:] = pan_
            if wiz.get('title'):                           # мастер Prepare: условие шага и кнопки (окно живёт)
                wr_ = UI.confirm_panel(view, wiz['title'], wiz['lines'], wiz['buttons'], wiz.get('hover'),
                                       W=W - SM.CAR_W, low=True)
                oy_ = UI.MENU_H + UI.TOOL_H
                wiz['rects'] = {k2: (r2[0], r2[1] + oy_, r2[2], r2[3] + oy_) for k2, r2 in wr_.items()}
            lay['view0'] = view                             # до перестановки списка: подложка для долгих дел
            if lay['list'] == 'left':                       # список — слева
                view = np.ascontiguousarray(np.hstack([view[:, W - SM.CAR_W:], view[:, :W - SM.CAR_W]]))
            if pick.get('k') is not None:                  # список съёмок выделения
                pick['rects'] = UI.pick_panel(view, pick['title'], PICK_HEAD, pick['rows'], pick['star'], pick.get('hover'),
                                              pick.get('scroll', 0), W=W, note=PICK_NOTE)
            prof.mark('list')
            cv2.imshow(WIN, np.vstack([menu_img(W), view]))
            prof.mark('show')
            prof.end()
            if n == 1:                                    # окно уже на экране: значок программы (Windows)
                set_icon(WIN)
        key = next(keys, EXIT) if keys is not None else cv2.waitKeyEx(8 if live is not None else 30)
        if keys is not None:
            cv2.waitKey(1)
            if isinstance(key, tuple) and key[0] == 'mouse':   # тест: событие мыши как от окна (ev, x, y, flags)
                on_mouse(*key[1:]); key = -1
            elif isinstance(key, tuple) and key[0] == 'call':  # тест: сделать что-то, как кнопкой
                key[1](); key = -1
            elif isinstance(key, tuple) and key[0] == 'qwait':   # тест: дождаться очереди
                if '_queue' in m.__dict__:
                    m._queue.wait(key[1] if len(key) > 1 else None)
                key = -1
            elif isinstance(key, tuple) and key[0] == 'wiz':   # тест: кнопка мастера Prepare
                wiz['click'] = key[1]; key = -1
            elif isinstance(key, tuple):                  # тест: щелчок мышью (x, y)
                clicks.append(key); key = -1
        if wiz.get('click'):                              # кнопка мастера Prepare
            wiz_action(wiz.pop('click'))
            continue
        if pick.get('k') is not None:                     # список съёмок: щелчок, Esc; прочие клавиши — мимо
            c_ = pick.pop('click', None)
            if c_ is None and key in (27, 13):
                c_ = 'close' if key == 27 else ('star', pick.get('cur', pick['star']))
            if c_ is not None:
                r_ = pick_do(c_)
                if r_:
                    msg, msg_t = r_, time.monotonic()
            clicks.clear()
            continue
        while clicks:                                     # щелчок по полю — ввод значения; по картинке — направляющие
            c_ = clicks.pop(0)
            if c_[0] == 'queue':                                    # панель очереди: x, Stop all, остальное — список
                if c_[1] == 'stop':
                    j_ = next((j_ for j_ in m.queue.jobs if j_.id == c_[2]), None)
                    if j_ is not None and m.queue.stop(j_):
                        msg = f"Stopping: {j_.title}" + (" (after the current frame)" if j_.lane == 'inst' else "")
                elif c_[1] == 'pause':
                    j_ = next((j_ for j_ in m.queue.jobs if j_.id == c_[2]), None)
                    if j_ is not None and j_.paused:
                        m.queue.resume(j_)
                        msg = f"Resumed: {j_.title}"
                    elif j_ is not None and m.queue.pause(j_):
                        msg = f"Paused: {j_.title}" + (" (after the current frame)" if j_.lane == 'inst' and
                                                      j_.state == 'running' else "") + " - > resumes"
                elif c_[1] == 'all':
                    n_ = len(m.queue.active())
                    m.queue.stop_all()
                    msg = f"Stopping all {n_} job{'s' if n_ != 1 else ''} of the queue"
                elif c_[1] == 'menu':
                    menu_do("Queue")
                msg_t = time.monotonic()
                continue
            if c_[0] == 'scroll':                                   # тачпад: увеличенный кадр едет за пальцами
                if zm['z'] > 1 and zm['c'] is not None:
                    Wl_ = W - SM.CAR_W
                    k2 = min(Wl_ / img.shape[1], H / img.shape[0]) * zm['z']
                    zm['c'] = np.asarray(zm['c'], float) - np.array(c_[3:5], float) / k2
                continue
            if c_[0] == 'wheel':                                    # зум к курсору
                pix = v2i(c_[1:3])
                f3_ = c_[3] if 0.05 < c_[3] < 2 and c_[3] != 1 else (1.25 if c_[3] > 0 else 0.8)
                zm['z'] = min(8.0, max(1.0, zm['z'] * f3_))
                Wl_ = W - SM.CAR_W
                k2 = min(Wl_ / img.shape[1], H / img.shape[0]) * zm['z']
                zm['c'] = pix - (np.array(c_[1:3], float) - (Wl_ / 2, H / 2)) / k2
                continue
            if c_[0] == 'tbar':                                     # шаги и большие кнопки
                it_ = next((t_ for x0_, x1_, t_ in menu['thits'] if x0_ <= c_[1] <= x1_), None)
                if it_:
                    tbar_do(it_)
                continue
            if c_[0] == 'swap':                                     # карта <-> камера
                nv['big'] = m._nav_big = not nv['big']
                nav_here()
                continue
            if c_[0] in ('mapclick', 'maprect', 'mappan', 'mapwheel', 'mapdbl', 'mapscroll'):   # карта
                if nv['mr'] is None or nav.c is None:
                    continue
                if c_[0] in ('mapwheel', 'mapscroll', 'mappan'):     # карта едет: пока едет — грубее (быстро)
                    nv['fast_t'] = time.monotonic()
                if c_[0] == 'mapwheel':                             # c_[3]: множитель (или знак — из тестов)
                    k3_ = c_[3] if c_[3] > 0.05 and abs(c_[3] - 1) < 1 and c_[3] not in (1,) else (1.25 if c_[3] > 0 else 0.8)
                    nav.zoom(c_[1:3], k3_)
                elif c_[0] == 'mapscroll':                          # тачпад двумя пальцами: карта едет
                    nav.pan(nav.c.copy(), np.array(c_[1:3], float))
                elif c_[0] == 'mappan':
                    if nv['pan'] is not None and nv['pan'][2] is not None:
                        nav.pan(nv['pan'][2], np.array(c_[1:3], float) - nv['pan'][:2])
                elif c_[0] == 'mapclick':
                    S_ = nav.to_um(c_[1:3])
                    try:
                        msg = busy_live("Going there", lambda: map_go(S_))
                    except MicroControlError as e:
                        msg = str(e)
                elif c_[0] == 'maprect':                            # c_[5]: Ctrl — свободная рамка
                    msg = new_area(nav.to_um(c_[1:3]), nav.to_um(c_[3:5]), free=len(c_) > 5 and bool(c_[5]))
                elif c_[0] == 'mapdbl':                             # открыть снятый участок под точкой
                    S_ = nav.to_um(c_[1:3])
                    from .layers import contains as _contains
                    hit_ = [k_ for f_, k_ in latest_scans() if _contains(m, m._regions[k_], S_[0], S_[1])]
                    if hit_:
                        m._sel = max(hit_, key=lambda k2: SM.obj(m, m._regions[k2]))
                        m._view_pick = None
                        open_view()
                    else:
                        msg = "Nothing shot here yet"
                msg_t = time.monotonic()
                continue
            if c_[0] == 'menu':                                     # строка меню
                it_ = next((t_ for x0_, x1_, t_ in menu['hits'] if x0_ <= c_[1] <= x1_), None)
                if it_:
                    menu_do(it_)
                continue
            if c_[0] in ('mdrag0', 'mdrag', 'mdrag1'):             # Shift + тянуть: сдвинуть выделение
                if back_v is None:
                    continue
                q_ = np.array(back_v((c_[1] + 0.5, c_[2] + 0.5)), float)
                if c_[0] == 'mdrag0':
                    from .layers import contains
                    hit_ = [k_ for k_, rg_ in enumerate(m._regions)
                            if rg_.get('kind') != 'group' and not SM.hidden(m, k_) and contains(m, rg_, q_[0], q_[1])]
                    k_ = max(hit_, key=lambda k2: SM.obj(m, m._regions[k2])) if hit_ else None   # самое верхнее
                    if k_ is None:                                  # не над выделением — сдвиг картинки
                        mdr.update(k=None)
                        zm['pan'] = (np.array(c_[1:3], float), np.array(zm['c'], float))
                    else:
                        mdr.update(k=k_, p0=q_, d=np.zeros(2))
                elif mdr.get('k') is None:
                    if zm['pan']:
                        zm['c'] = zm['pan'][1] - (np.array(c_[1:3], float) - zm['pan'][0]) / kz
                    if c_[0] == 'mdrag1':
                        zm['pan'] = None
                else:
                    mdr['d'] = q_ - mdr['p0']
                    if c_[0] == 'mdrag1':
                        k_, d_ = mdr.pop('k'), mdr.pop('d')
                        if np.hypot(*d_) > 0.05:
                            from .layers import move
                            note_brush(m._regions[k_])
                            try:
                                move(m, k_, float(d_[0]), float(d_[1]))
                                msg = f"{SM.title(m, k_)} moved by {d_[0]:+.1f}, {d_[1]:+.1f} um (Ctrl+Z: back)"
                            except ValueError as e_:          # пирамидка: за родителя / на такое же — нельзя
                                msg = f"Not moved: {e_}"
                            noted_done()
                            SM.persist(m)
                            msg_t = time.monotonic()
                continue
            if c_[0] == 'carw':                                     # прокрутка карусели
                car['scroll'] = car_clamp(car['scroll'] - 60 * np.sign(c_[3]))
                continue
            if c_[0] in ('cmove', 'cdrop'):                         # тянут: полосу прокрутки или карточку
                pr_ = car.get('press')
                if pr_ is None:
                    continue
                if pr_['what'] == 'bar':
                    ct_ = max(car.get('content', 0), H + 1)
                    car['scroll'] = car_clamp(pr_['s0'] + (c_[2] - pr_['y']) * ct_ / H)
                elif pr_['what'] == 'card':
                    if abs(c_[2] - pr_['y']) + abs(c_[1] - pr_['x']) > 8:
                        car['drag'] = dict(src=pr_['k'], x=c_[1], y=c_[2])
                    if car.get('drag') and c_[0] == 'cmove':        # у края — прокрутить
                        if c_[2] < 60:
                            car['scroll'] = car_clamp(car['scroll'] - 20)
                        elif c_[2] > H - 30:
                            car['scroll'] = car_clamp(car['scroll'] + 20)
                if c_[0] == 'cdrop':
                    car['press'] = None
                    dr_ = car.pop('drag', None)
                    if dr_ is not None:
                        tg_ = car_target(c_[2])
                        if tg_ is not None and tg_[0] != dr_['src']:
                            r_ = SM.reorder(m, dr_['src'], tg_[0], tg_[1])
                            if r_:
                                msg, msg_t = r_, time.monotonic()
                continue
            if c_[0] == 'car':                                      # щелчок по карусели
                xc_ = c_[1] - (W - SM.CAR_W)
                if xc_ >= SM.CAR_W - 8 and car.get('content', 0) > H:   # полоса прокрутки
                    car['press'] = dict(what='bar', y=c_[2], s0=car['scroll'])
                    continue
                h_ = SM.hit(car['hits'], xc_, c_[2])
                if h_ and h_[0] == 'tool':                          # операция над отмеченными
                    if h_[1] < len(car['tools']):
                        msg, msg_t = layer_op(car['tools'][h_[1]]), time.monotonic()
                    continue
                ctrl_ = len(c_) > 3 and c_[3]
                if h_ and h_[0] == 'sel' and ctrl_ and h_[1] < len(car.get('order', [])) and \
                        car['order'][h_[1]] < len(m._regions):      # Ctrl+щелчок: отметить / снять
                    k_ = car['order'][h_[1]]
                    car['multi'] = [x_ for x_ in car['multi'] if x_ != k_] if k_ in car['multi'] else car['multi'] + [k_]
                    msg, msg_t = (f"Marked: {', '.join(SM.title(m, x_) for x_ in car['multi'])} (the first is the target)"
                                  if car['multi'] else "Nothing marked"), time.monotonic()
                    continue
                if h_ and h_[0] == 'sel' and h_[1] < len(car.get('order', [])) and \
                        car['order'][h_[1]] != len(m._regions):       # карточку можно перетащить
                    car['press'] = dict(what='card', k=car['order'][h_[1]], x=c_[1], y=c_[2])
                if h_:
                    act, k = h_
                    k = car['order'][k] if k < len(car.get('order', [])) else k
                    if act == 'chk' and k < len(m._regions):       # галочка: в экспорт или нет
                        rg_ = m._regions[k]
                        rg_['export'] = not rg_.get('export', False)
                        SM.persist(m)
                        msg = f"{SM.title(m, k)}: " + ("goes into the export" if rg_['export'] else "not in the export")
                        msg_t = time.monotonic()
                        car['press'] = None
                        continue
                    if act == 'eye' and k < len(m._regions):       # скрыть / показать на картах
                        rg_ = m._regions[k]
                        rg_['hidden'] = not rg_.get('hidden')
                        if not rg_['hidden']:
                            rg_.pop('hidden')
                        SM.persist(m)
                        msg, msg_t = (f"{SM.title(m, k)} " + ("hidden on the maps (the eye shows it again)"
                                                                if rg_.get('hidden') else "shown on the maps")), time.monotonic()
                        continue
                    if act == 'alpha' and k < len(m._regions):     # прозрачность картинки: 100 -> 75 -> ... -> 0 -> 100
                        lv_ = [1.0, 0.75, 0.5, 0.25, 0.0]
                        u_ = m._regions[k]['uid']
                        a0_ = map_alpha().get(u_, 1.0)
                        i_ = min(range(len(lv_)), key=lambda i2_: abs(lv_[i2_] - a0_))
                        a1_ = lv_[(i_ + 1) % len(lv_)]
                        map_alpha({u_: a1_})
                        car['shot_t'] = -9
                        msg, msg_t = (f"{SM.title(m, k)}: picture {100 * a1_:.0f} %" + (" (outline only)" if a1_ == 0 else "")
                                      + "   (click: less, [ ]: more / less)"), time.monotonic()
                        continue
                    if act == 'more' and k < len(m._regions):      # «...»: все съёмки, выбрать основную
                        car['press'] = None
                        msg, msg_t = pick_open(k), time.monotonic()
                        continue
                    if act == 'obj' and k < len(m._regions):       # плашка объектива: следующий (пока не снято)
                        msg, msg_t = cycle_objective(k), time.monotonic()
                        continue
                    if act == 'sel':
                        car['confirm'] = None
                        if k < len(m._regions):
                            m._sel = k
                            msg, msg_t = (f"{SM.title(m, k)} chosen ({SM.layer_name(m, k)}; Go or Enter: "
                                          "drive there; Ctrl+click: mark for Merge / Minus / Overlap)"), time.monotonic()
                    elif act == 'del' and k <= len(m._regions):     # сразу, без подтверждения (Ctrl+Z вернёт)
                        delete_region(k)
                        msg_t = time.monotonic()
                    elif act == 'go' and k < len(m._regions):
                        car['confirm'] = None
                        hold_ = live.hold() if live is not None else None
                        try:
                            if hold_:
                                hold_.__enter__()
                            go_region(k)
                        except MicroControlError as e:
                            msg = str(e)
                        finally:
                            if hold_:
                                hold_.__exit__()
                        msg_t = time.monotonic()
                continue
            if c_[0] in ('pan0', 'pan', 'pan1'):                    # сдвиг увеличенной картинки
                if c_[0] == 'pan0':
                    zm['pan'] = (np.array(c_[1:3], float), np.array(zm['c'], float))
                elif c_[0] == 'pan' and zm['pan']:
                    zm['c'] = zm['pan'][1] - (np.array(c_[1:3], float) - zm['pan'][0]) / kz
                else:
                    zm['pan'] = None
                continue
            if len(c_) == 3:
                if c_[0] == 'up':
                    drag = False
                continue
            x, y = c_
            hit = [ax for ax, (x0, y0, x1, y1) in rects.items() if x0 <= x <= x1 and y0 <= y <= y1]
            if rul['on'] and not hit and x < W - SM.CAR_W and back_v is not None:   # точка линейки
                q_ = back_v((x + 0.5, y + 0.5))
                rul['pts'] = [q_] if len(rul['pts']) != 1 else rul['pts'] + [q_]
                msg = ruler_text(*rul['pts']) if len(rul['pts']) == 2 else "Ruler: click the second point"
                msg_t = time.monotonic()
                continue
            if edit:
                commit(edit)
            if hit and hit[0] == 'usefp':
                key, edit = INT + ord('y'), None
                break
            if hit and hit[0] == 'oneside':
                key, edit = INT + ord('b'), None
                break
            if hit and hit[0] == 'panel':                # кнопка «спрятать / показать панель»
                key, edit = INT + ord('p'), None
                break
            if hit and hit[0].startswith('act:'):        # кнопки панели Settings: редкие дела прибора
                key, edit = INT + ord(hit[0][4:]), None
                break
            edit = (hit[0], "") if hit else None
        if key == -1:
            if keys is None and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            try:
                if not qbusy():
                    m._keepalive()           # простой: запрос «на связи», чтобы Windows не усыпляла USB
            except MicroControlError as e:
                msg, msg_t = str(e), time.monotonic()
            continue
        if key in PROF_KEYS:                               # ` / F12: сколько мс на что уходит (и в журнал)
            msg, msg_t = prof.toggle(), time.monotonic()
            continue
        if key == 16 or (key > 0 and (key & 0xFF) in (ord('p'), ord('P')) and (key >> 16) & 0x04):   # Ctrl+P:
            m._panel_hidden = not getattr(m, '_panel_hidden', False)                                  # левая панель
            msg, msg_t = ("" if not m._panel_hidden else "Settings panel hidden: Ctrl+P or the arrow on the left "
                          "shows it"), time.monotonic()
            continue
        if exit_key(key):                                 # Shift+Esc / Ctrl+Esc — выйти
            q_ = m.__dict__.get('_queue')
            if q_ is not None and q_.active():            # очередь не кончилась: ждать, остановить или пусть идёт
                if keys is not None:
                    q_.wait()
                else:
                    from . import dialogs as D
                    r_ = D.choose("The queue is not done", [j_.line() for j_ in q_.active()], None,
                                  ("Keep the window", "Stop all and exit", "Exit, the queue goes on"),
                                  extra="The queue goes on without the window too (m.queue): m.close() stops it")
                    if r_ is None or r_[0] == "Keep the window":
                        continue
                    if r_[0] == "Stop all and exit":
                        q_.stop_all()
                        q_.wait(15)
            break
        ud_ = undo_key(key)
        if ud_:                                           # Ctrl+Z — отменить, Ctrl+Y / Ctrl+Shift+Z — вернуть
            e_ = hist.undo() if ud_ == 'undo' else hist.redo()
            if e_ is None:
                msg = "Nothing to undo" if ud_ == 'undo' else "Nothing to redo"
            else:
                hold_ = live.hold() if live is not None else None
                try:
                    if hold_:
                        hold_.__enter__()
                    h_restore(e_['before'] if ud_ == 'undo' else e_['after'])
                    msg = (f"Undone: {e_['label']}  (Ctrl+Y: redo, {len(hist)} more to undo)" if ud_ == 'undo' else
                           f"Redone: {e_['label']}")
                    m._map_win = None
                except MicroControlError as ex_:
                    msg = f"Could not undo ({e_['label']}): {ex_}"
                finally:
                    if hold_:
                        hold_.__exit__()
                h_last = (h_sig(), h_snap())
            msg_t = time.monotonic()
            continue
        if key == 9 and not edit and lay['panes'] != 'one':   # два вида: Tab меняет их местами
            lay['panes'] = 'map|camera' if lay['panes'] == 'camera|map' else 'camera|map'
            set_ui_pref('panes', lay['panes'])
            nav_here()
            continue
        if key == 9 and not edit:                         # Tab: карта <-> камера крупно
            nv['big'] = m._nav_big = not nv['big']
            nav_here()
            continue
        if key == 9:                                      # Tab в поле ввода: следующее поле (X -> Y -> Z -> выход)
            if edit:
                commit(edit)
            order = [None, *FIELDS]
            nxt = order[(order.index(edit[0] if (edit and edit[0] in FIELDS) else None) + 1) % len(order)]
            edit = (nxt, "") if nxt else None
            continue
        if edit:                                          # режим ввода: цифры, Backspace, Enter, Esc
            if 48 <= key <= 57 and len(edit[1]) < (12 if edit[0] == 'lab' else 6):
                edit = (edit[0], edit[1] + chr(key))
            elif edit[0] == 'lab' and key in (32, 44, 45) and len(edit[1]) < 12:   # пробел, запятая, минус
                edit = (edit[0], edit[1] + (' ' if key in (32, 44) else '-'))
            elif key in (46, 44) and edit[0] in ('exp', 'afs') and '.' not in edit[1]:   # дробное: . или ,
                edit = (edit[0], edit[1] + '.')
            elif key == 8:
                edit = (edit[0], edit[1][:-1])
            elif key in (13, 10):
                commit(edit); edit = None
            elif key == 27:
                edit = None
            continue
        if isinstance(key, int) and INT <= key < INT + 256:   # кнопка панели: внутреннее дело как есть
            raw = chr(key - INT)
        else:                                              # одна буква — один смысл в обоих окнах
            raw = chr(key) if 0 <= key < 128 else ''
            if key in DEL_KEYS:
                raw = 'c'                                  # Del: удалить выбранное / отменить новое
            elif raw in VIEW_ONLY:
                msg, msg_t = f"{raw.upper()}: only in the view ({VIEW_ONLY[raw]}; V opens it)", time.monotonic()
                continue
            elif raw in ('x', 'X'):                        # X: экспорт
                export_drawing()
                continue
            elif raw == 'J':                               # Shift+J: точная сшивка (quality) выбранного
                stitch_quality()
                continue
            elif raw in ('[', ']'):                        # [ ]: прозрачность картинки выбранного на карте
                if m._sel is None or m._sel >= len(m._regions):
                    msg, msg_t = "[ ]: choose a selection (its card) first", time.monotonic()
                    continue
                lv_ = [1.0, 0.75, 0.5, 0.25, 0.0]
                u_ = m._regions[m._sel]['uid']
                a0_ = map_alpha().get(u_, 1.0)
                i_ = min(range(len(lv_)), key=lambda i2_: abs(lv_[i2_] - a0_))
                a1_ = lv_[max(0, min(len(lv_) - 1, i_ + (1 if raw == '[' else -1)))]
                map_alpha({u_: a1_})
                msg, msg_t = (f"{SM.title(m, m._sel)}: picture {100 * a1_:.0f} %" + (" (outline only)" if a1_ == 0 else "")
                              + "   [: more transparent, ]: less"), time.monotonic()
                continue
            elif raw in SETTINGS_ONLY:
                msg, msg_t = f"{raw.upper()}: {SETTINGS_ONLY[raw]} is a button on the Settings panel now", time.monotonic()
                continue
            else:
                raw = MAN_KEYS.get(raw, raw)
        ch = raw.lower()
        if qbusy() and (key in LEFT or key in RIGHT or key in UP or key in DOWN or key in PGUP or key in PGDN
                        or key in (13, 10) or (ch and ch in 'eqfkliwoadm 12') or raw in ('0', ')')):
            msg, msg_t = "The instrument is busy: " + QBUSY, time.monotonic()
            continue
        ax = m._screen_axes()
        (ah, sh), (av, sv) = ax['h'], ax['v']
        sh, sv = (-sh if m.arrows_flip_x else sh), (-sv if m.arrows_flip_y else sv)   # только клавиши
        hold = live.hold() if live is not None and not qbusy() else None   # очередь снимает: камера — её
        if hold:
            hold.__enter__()
        try:
            if key == 27 and rul['on']:
                rul['on'], rul['pts'] = False, []
                msg = "Ruler off"
            elif key == 27:
                msg = "Exit: Shift+Esc or close the window"
            elif key in LEFT:                              # движение — только стрелки
                m.move(**{'d' + ah: -sh * steps[ah]}); msg = SM.follow_z(m)
            elif key in RIGHT:
                m.move(**{'d' + ah: sh * steps[ah]}); msg = SM.follow_z(m)
            elif key in UP:
                m.move(**{'d' + av: -sv * steps[av]}); msg = SM.follow_z(m)
            elif key in DOWN:
                m.move(**{'d' + av: sv * steps[av]}); msg = SM.follow_z(m)
            elif ch == 's':                                # снять и сшить (S дважды)
                if hold:
                    hold.__exit__(); hold = None           # камера нужна скану из другого потока
                scan_here()
            elif ch == 'v':                                # открыть последний скан в просмотре
                if hold:
                    hold.__exit__(); hold = None
                open_view()
            elif ch == 'i':                                # лазер вкл. / выкл.
                if m.laser:
                    m._laser_last = m.laser
                    m.laser = 0
                    msg = "Laser off"
                else:
                    m.laser = getattr(m, '_laser_last', 0) or 1
                    msg = f"LASER ON: {m.laser:g} mW (I: off)"
                L.event('MAN-007', msg + ".")
            elif ch == 'p':                                # левая панель: спрятать / показать
                m._panel_hidden = not getattr(m, '_panel_hidden', False)
                msg = "Panel hidden: Ctrl+P or the arrow on the left shows it" if m._panel_hidden else ""
            elif ch == 't':                                # повернуть картинку камеры на 180° (помнится для прибора)
                m.turn_180 = not m.turn_180
                m._map_win = None
                msg = (f"Camera picture turned by 180 deg ({'on' if m.turn_180 else 'off'}, remembered). "
                       "Now aim at the crosses and press O to check the directions")
            elif raw in ('[', ']'):                        # стрелки (и карта вместе с ними) — в другую сторону
                k_ = 'arrows_flip_x' if raw == '[' else 'arrows_flip_y'
                setattr(m, k_, not getattr(m, k_))
                m._map_win = None
                msg = (f"Arrows {'left / right' if raw == '[' else 'up / down'} turned the other way, the mini map "
                       f"follows them (remembered; {raw} again: back)")
            elif raw == '#':                               # кнопка Clear fog: «туман войны» карты — заново
                fog_of(m).clear()
                m._map_win = None
                msg = "Fog of war cleared (the live frames on the map; the scans are not touched)"
            elif raw == '\\':                              # \: карта образца — ещё на 90° против часовой (помнится)
                m.map_turn = (int(m.map_turn) + 90) % 360
                nav.c = None                               # вид — заново, чтобы всё поместилось
                msg = f"Map turned: {m.map_turn} deg anticlockwise (remembered; \\ again: 90 deg more)"
            elif ch == 'b':                                # B: подъезд с одной стороны (люфт), помнится для прибора
                m.one_side = not m.one_side
                msg = (f"One-side approach on: X and Y always stop coming from the same side (a move back goes "
                       f"{m.overshoot} steps past and returns); scans go row by row. Remembered for this microscope"
                       if m.one_side else "One-side approach off: scans go as a snake (faster, backlash counted by the stitch)")
            elif ch == 'y':
                m._use_fp = not m._use_fp
                msg = ("Z follows the focus points: arrows inside an area and every scan frame" if m._use_fp else
                       "Z stays as it is: arrows do not change Z; the scan uses one Z")
            elif ch == 'z':
                zm['z'], zm['c'] = 1.0, None; msg = ''
            elif raw in ('+', '=', '-', '_'):              # зум с клавиатуры (к центру окна)
                zm['z'] = min(8.0, max(1.0, zm['z'] * (1.25 if raw in '+=' else 0.8)))
                msg = f"Zoom x{zm['z']:.2f}"
            elif ch == 'o':
                r = m.check_orientation(); m._orient_checked = True
                msg = r.get('Changed', r.get('Result', r.get('Reason', '')))
                msg = f"Orientation: {msg}" + (". If the arrow keys now go the wrong way: [ - left / right, "
                                               "] - up / down" if 'Changed' in r else "")
            elif ch in ('e', 'q') or key in PGUP or key in PGDN:
                up = ch == 'e' or key in PGUP
                dz = steps['z'] * (SHIFT if raw in ('E', 'Q') else 1)
                r = m.move(dz=dz if up else -dz)
                msg = r.get('Note', f"Stage {'up' if up else 'down'} {dz:g} um")
            elif ch == 'm':
                from .config import OBJECTIVES
                obs = sorted(OBJECTIVES)
                m.objective = obs[(obs.index(m.objective) + 1) % len(obs)]
                d0 = default_steps(m)
                steps['x'], steps['y'] = d0['x'], d0['y']
                msg = f"Objective {m.objective}x: put it in place. Steps X, Y = one frame: {steps['x']:g}, {steps['y']:g} um"
                wq_ = (m.__dict__.get('_queue').waiting_for() if m.__dict__.get('_queue') is not None else {}).get(m.objective)
                if wq_:
                    msg += f".  {wq_} scan{'s' if wq_ != 1 else ''} wait for {m.objective}x: S starts them"
            elif ch == 'g':
                d0 = default_steps(m)
                steps['x'], steps['y'] = d0['x'], d0['y']
                msg = f"Steps X, Y = one frame: {steps['x']:g}, {steps['y']:g} um"
            elif ch == 'f':                                # F: автофокус (A — мастер Prepare)
                r = busy_live("Autofocus", m.focus)
                msg = f"Focused at Z {r['Z']}" + ("  (weak peak)" if 'Note' in r else "")
            elif ch == 'k':                                # запомнить Z здесь как точку фокуса выделения
                msg = SM.add_point(m)
                L.event('MAN-006', msg + ".")
            elif ch == 'j':                                # J: сверка дополненной реальности с мозаикой
                arl['on'] = not arl['on']
                if not arl['on']:
                    m.__dict__['_ar_d'] = None
                arl['done'] = False
                msg, msg_t = ("AR lock on: when the stage stops, the camera picture is matched to the scan and "
                              "the outlines follow the real crosses" if arl['on'] else
                              "AR lock off: outlines by the stage model only"), time.monotonic()
            elif ch == 'l':                                # L: свет вкл. / выкл.
                m.light = 0 if m.light else 5
                msg = "Light on" if m.light else "Light off"
            elif raw in (',', '.') and m._regions:         # предыдущая / следующая (в порядке дерева слоёв)
                od_ = [k_ for k_, _d in SM.tree(m)]
                i_ = od_.index(m._sel) if m._sel in od_ else 0
                m._sel = od_[(i_ + (1 if raw == '.' else -1)) % len(od_)]
                msg = f"Selection {SM.num(m, m._sel)} chosen ({SM.layer_name(m, m._sel)}; Enter: drive there)"
            elif key in (13, 10) and m._sel is not None and not m._pending:
                go_region(m._sel)
            elif ch == ' ':
                busy_live("Taking a picture", m.photo)
                msg = f"Picture saved: {m.last_photo}"
            elif raw in ('0', ')'):                        # выделения хранятся в настоящих мкм: остаются на месте
                if raw == '0':
                    m.offset()
                    msg = "Zero is here. Real position " + repr(m.real_position)
                else:
                    m.offset_clear()
                    msg = "Offset removed: real coordinates"
            elif ch == 'a':                                # A: мастер подготовки (как кнопка Prepare)
                prepare_start()
            elif ch == 'd':                                # D: дисторсия объектива
                prepare_here('distortion')
            elif ch == 'w':                                # баланс белого (навести на пустое место)
                busy_live("White balance", m.white_balance)
                msg = "White balance done"
            elif ch == 'r':                                # линейка: две точки на картинке -> расстояние
                rul['on'], rul['pts'] = not rul['on'], []
                msg = ("Ruler: click two points on the picture (you may move the stage between them); "
                       "R or Esc: off") if rul['on'] else "Ruler off"
            elif ch == 'h':
                show_help = not show_help
            elif ch == 'c':                                # отменить новую область / удалить выбранную (дважды)
                if m._pending:
                    delete_region(len(m._regions))
                elif m._sel is not None:                   # сразу (Ctrl+Z вернёт)
                    delete_region(m._sel)
            elif ch in '12':                               # углы области: строго по очереди 1, потом 2
                msg = SM.corner(m, int(ch), img)
                L.event('MAN-003', msg + ".")
            else:
                continue
            msg_t = time.monotonic()
        except MicroControlError as e:
            msg, msg_t = str(e), time.monotonic()
        finally:
            if hold:
                hold.__exit__()
        if keys is None:                 # нажатия, накопленные за время движения (автоповтор при удержании
            t_end = time.monotonic() + 0.5   # клавиши), выбрасываются: отпустил клавишу — столик стоит
            while cv2.waitKeyEx(1) != -1 and time.monotonic() < t_end:
                pass
    if live is not None:
        live.stop()
    m._save_last_view()                                   # кадр здесь — сверка положения при следующем подключении
    SM.persist(m)
    try:
        fog_of(m).save()                                  # туман карты — у образца
    except Exception:
        pass
    if not fullscreen:
        window_size(WIN, remember=True)                                  # запомнить размер для следующего окна
    try:                                                   # окно уже закрыли крестиком — ничего
        cv2.destroyWindow(WIN)
    except cv2.error:
        pass
    cv2.waitKey(1)
    regs = m._regions
    real_ = [k for k, r in enumerate(regs) if r.get('kind') != 'group']
    k_area = m._sel if m._sel in real_ else (real_[0] if real_ else None)
    m.area = reg_area(regs[k_area]) if k_area is not None else None
    if m.area is not None:
        m.area._of_selection = True                         # зона выбранного выделения, не новая
    m._focus_pts = [dict(p_, tag=str(n_)) for n_, k_, p_ in SM.numbered(m) if k_ == (m._sel or 0)] if regs else []
    L.event('MAN-002', f"Manual mode closed at {m.position}. Selections: {len(regs)}.")
    if not regs:
        print("Manual mode closed. No selection: mark two corners with keys 1 and 2.")
        return None
    lines_ = []
    for k, depth in SM.tree(m):                                          # деревом слоёв: 10x > 50x > 100x
        rg = regs[k]
        o_ = SM.obj(m, rg)
        if rg.get('kind') == 'group':
            lines_.append(f"  {'    ' * depth}{'└ ' if depth else ''}{SM.title(m, k)}")
            continue
        nx, ny = SM.frames_of(m, k)
        fr_ = f"{nx * ny} frames (by the outline)" if rg.get('shape') else f"{nx} x {ny} = {nx * ny} frames"
        lines_.append(f"  {'    ' * depth}{'└ ' if depth else ''}{SM.title(m, k)}  {o_}x  {reg_extent(rg)}   "
                      f"{fr_}   focus points {first_num(k)}" + ("   (hidden)" if SM.hidden(m, k) else ""))
    print("Manual mode closed." + (" The new selection without corner 2 is kept for the next time." if m._pending else "")
          + "\n" + "\n".join(lines_) + f"\n  m.scan() shoots the selections of {m.objective}x, each into its own "
          "folder; the others wait for their objective.")
    return m.area
