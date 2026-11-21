"""Ручной режим: живая картинка на весь экран, движение по кадрам, отметка зоны (как «приват» в Minecraft).

Координаты знать не нужно: доезжаете до края нужного участка и отмечаете угол.
Мини-карта в углу показывает отмеченную зону и где вы сейчас.

  Стрелки             На шаг X / Y в сторону (буквы — для действий)
  E / Q (PgUp/PgDn)   Столик вверх / вниз на шаг Z (Shift — x5)
  A                   Автофокус;  K — точка фокуса (Z здесь)
  1 / 2               Углы выделения;  C — удалить выделение (дважды)
  S (дважды)          Снять выделения этого объектива и сшить, прямо в окне (Esc — стоп)
  V                   Открыть последний скан в просмотре, после — снова сюда
  L / I / W           Свет / лазер / баланс белого
  R                   Линейка: два щелчка — расстояние
  Пробел              Снимок в photos/
  0 / Shift+0         Ноль здесь / настоящие координаты
  Ctrl+Z / Ctrl+Y     Отменить / вернуть
  H                   Подсказка
  Shift+Esc           Выход
"""
import math
import time
import os
import copy
import json
import cv2
import numpy as np
from . import log as L
from .errors import MicroControlError
from .util import wheel_delta, open_window, window_size, exit_key, EXIT, set_icon
from .undo import History, undo_key
from . import focus as F_
import threading
from .scan import Area, focus_z
from .config import scan_overlap
from . import samples as SM
from . import timing as TM
from . import ui as UI

WIN = "MicroControl"
HELP = ["Shift+Esc: Exit (or close the window)", "P: Hide / show the left panel",
        "Arrows: Move by Step X / Step Y   E / Q (PgUp / PgDn): Stage up / down by Step Z (Shift: x5)",
        "Tab or click a field: steps (multiple of 5 um), exposure (ms), objective; Enter",
        "G: Steps X, Y = one frame", "M: Next objective (4x, 10x, 20x, 50x, 100x)",
        "1 / 2: Corners of a selection, one after another (their Z is kept as focus points); a higher objective may mark inside a lower one",
        "A: Autofocus   K: Keep Z here as a focus point of this selection",
        "Y (or the box): Z follows the focus points of the area: when moving with the arrows and in the scan",
        "Selections (right, as layers 10x > 50x > 100x): click = choose, Go = drive there, x = delete;  , / . = previous / next;  Enter = drive",
        "C: Cancel the new selection / delete the chosen one (press twice)",
        "S (twice): Scan the selections of this objective and stitch them, here in the window (Esc: stop)",
        "V: Open the last scan in the view (crosses, squares, export); closing it brings you back here",
        "L: Light on / off   I: Laser on / off   W: White balance (aim at an empty place)   Space: Picture",
        "R: Ruler: click two points (distance, um)   0: Zero here   Shift+0: Real coordinates",
        "Wheel: zoom   Middle button or Shift+drag: move the zoomed picture   Z: whole frame",
        "T: Turn the camera picture by 180 degrees (remembered)",
        "[ / ]: Arrows left-right / up-down the other way; the mini map always follows the arrows (remembered)",
        "O: check with the camera which way the picture moves",
        "Ctrl+Z: undo (selections, focus points, moves: the stage drives back)   Ctrl+Y: redo", "H: Hide this help"]
LEFT, UP, RIGHT, DOWN = {2424832, 65361}, {2490368, 65362}, {2555904, 65363}, {2621440, 65364}
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
        cv2.line(img, tuple((a + u * t).astype(int)), tuple((a + u * min(t + dash, n)).astype(int)), color, 1, cv2.LINE_AA)
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
            UI.plus(img, u + 0.5, v + 0.5, RED, r=8, gap=3)


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
            try:
                img = self.m.image(0)
            except Exception:
                time.sleep(0.05)
                continue
            self.img, self.n = img, self.n + 1

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
    P = np.array([m._p['x'], m._p['y']], float) * d
    c = np.array([iw / 2, ih / 2])
    to = lambda S: tuple(i2v(c + K @ ((P - np.asarray(S)) / d)))
    back = lambda q: P - d * np.linalg.solve(K, v2i(q) - c)
    return to, back, i2v, v2i, c, k, zc


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
    for (rx0, ry0, rx1, ry1), oc, olab in others:                        # другие области — своим цветом
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
    try:                                                                 # выбранные в просмотре квадраты всех сканов:
        from .selections import marked_all                               # тёмно-зелёные, своей формы и масштаба
        polys = marked_all(m.dirs['scans'])
    except Exception:
        polys = []
    if polys:
        ovm = sub.copy()
        gx1m, gy1m = gx0 + gw, gy0 + gh
        for q in polys:
            pts = np.int32([to(*f(x_ - m._off['x'], y_ - m._off['y'])) for x_, y_ in q])
            if pts[:, 0].max() < gx0 or pts[:, 0].min() > gx1m or pts[:, 1].max() < gy0 or pts[:, 1].min() > gy1m:
                continue
            cv2.fillPoly(ovm, [pts], (40, 105, 40))
            cv2.polylines(ovm, [pts], True, (70, 160, 70), 1, cv2.LINE_AA)
        sub[gy0:int(gy1m), gx0:int(gx1m)] = cv2.addWeighted(ovm, 0.75, sub, 0.25, 0)[gy0:int(gy1m), gx0:int(gx1m)]
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
        label(f"Frame: row {cj + 1}, column {ci + 1}    Key 1: corner of a new area", (W / 2, H - 12), (170, 170, 170), 0.45)
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


FIELDS = ('x', 'y', 'z', 'exp', 'obj')             # порядок полей (Tab)


def draw_fields(view, steps, edit, scale, top, m=None):
    """Поля в левом верхнем углу: шаги X, Y, Z, экспозиция, объектив. Возвращает прямоугольники полей."""
    rects = {}
    y = top
    cur = dict(steps)
    if m is not None:
        cur.update(exp=m.exposure, obj=m.objective)
    for ax in (FIELDS if m is not None else 'xyz'):
        val = (edit[1] + "_") if (edit and edit[0] == ax) else f"{cur[ax]:g}"
        if ax == 'exp':
            txt = f"Exposure [ {val:>6} ] ms"
        elif ax == 'obj':
            txt = f"Objective[ {val:>6} ] x   (4, 10, 20, 100; M: next)"
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
    m._need_cam()
    WH0 = open_window(WIN, fullscreen)
    if fullscreen:
        SW, SH = screen_size()
    L.event('MAN-001', f"Manual mode at {m.position}.")
    show_help, msg, msg_t, sharp, n = False, "H: Help.  Tab or click a field to edit it", time.monotonic(), 0.0, 0
    SM.init_areas(m)                                     # области образцов (настоящие мкм)
    car = dict(scroll=0, hits=[], confirm=None, t=0.0)   # карусель справа
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

    menu = dict(hover=None, hits=[])                     # строка меню сверху: проект, образец, Open, New ...

    def on_mouse(ev, x, y, flags, *_):
        # тянуть любой кнопкой по картинке (левой — не по полям) — сдвиг увеличенной картинки; колесо, + / - — зум
        if y < UI.MENU_H and not zm['pan']:                 # строка меню
            menu['hover'] = x if ev == cv2.EVENT_MOUSEMOVE else menu['hover']
            if ev == cv2.EVENT_LBUTTONDOWN:
                clicks.append(('menu', x))
            return
        menu['hover'] = None
        y -= UI.MENU_H
        if ev == cv2.EVENT_MOUSEMOVE:
            hover[0], hover[1] = (x, y), time.monotonic()
        if x >= W - SM.CAR_W and not zm['pan']:          # карусель образцов справа
            if ev == cv2.EVENT_LBUTTONDOWN:
                clicks.append(('car', x, y))
            elif ev == cv2.EVENT_MOUSEWHEEL:
                clicks.append(('carw', x, y, wheel_delta(flags)))
            return
        if ev == cv2.EVENT_MOUSEWHEEL:
            clicks.append(('wheel', x, y, wheel_delta(flags)))
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
            r = m.check_orientation()
            m._orient_checked = 'Reason' not in r
            msg, msg_t = "Orientation: " + r.get('Changed', r.get('Result', r.get('Reason', ''))), time.monotonic()
        except MicroControlError as e:
            msg, msg_t = str(e), time.monotonic()
    rects = {}
    live = _Live(m) if keys is None else None             # в тестах — синхронно
    last_n, last_t, img = -1, 0.0, None
    W, H = WH0 if WH0 else (1280, 900)

    def work(P):                                         # настоящие -> рабочие мкм
        return (P[0] - m._off['x'], P[1] - m._off['y'])

    def reg_rect(reg):
        fw_, fh_ = m.frame_size
        return SM.rect(reg, fw_, fh_)               # размер кадра — объектива самого выделения

    def reg_area(reg):
        return Area(*work(reg['A']), *work(reg['B']))

    def first_num(k):                                    # номера точек фокуса области k
        nums = [n_ for n_, kk, _p in SM.numbered(m) if kk == k]
        return (f"{nums[0]}-{nums[-1]}" if len(nums) > 1 else str(nums[0])) if nums else "-"

    def car_items():
        """Карточки деревом слоёв: карта, под ней (с отступом) её участки большего увеличения и т. д.
        car['order'][позиция] -> индекс выделения (len(m._regions) — новое)."""
        items, order = [], []
        for k, depth in SM.tree(m):
            reg = m._regions[k]
            o_ = SM.obj(m, reg)
            x0_, y0_, x1_, y1_ = reg_rect(reg)
            nx_, ny_ = SM.frames_of(m, k)                  # кадры и время — для объектива этого выделения
            fr_ = (f"{nx_} x {ny_} = {nx_ * ny_} frames, " +
                   TM.fmt(TM.estimate(m, nx_ * ny_, 'points' if m._use_fp and reg['pts'] else 'off', o_)[0]))
            items.append(dict(title=f"Selection {SM.num(m, k)}", obj=f"{o_}x", color=SM.col(m, k),
                              thumbs=reg.get('thumbs', []), confirm=car['confirm'] == k, depth=depth,
                              muted=o_ != m.objective,
                              lines=[fr_, f"{(x1_ - x0_) / 1000:.2f} x {(y1_ - y0_) / 1000:.2f} mm",
                                     f"focus points {first_num(k)}"]
                              + ([f"put {o_}x in place to scan it"] if o_ != m.objective else [])))
            order.append(k)
        if m._pending:
            k = len(m._regions)
            items.append(dict(title=f"Selection {SM.num(m, k)}", color=SM.col(m, k), thumbs=m._pending.get('thumbs', []),
                              go=False, confirm=car['confirm'] == k, obj=f"{SM.obj(m, m._pending)}x",
                              lines=["corner 1 set", "go to the other corner, press 2", f"focus points {first_num(-1)}"]))
            order.append(k)
        car['order'] = order
        return items

    MENU = ["Open", "New project", "New sample", "Info", "Scans", "View", "Help"]

    def menu_img(W_):
        sm_ = m.sample
        img_, menu['hits'] = UI.menu_bar(W_, f"{sm_.title}" + (f"  —  {sm_.description}" if sm_.description else ""),
                                         MENU, menu['hover'])
        return img_

    def menu_do(item):
        """Пункт меню: открыть / создать проект и образец, сведения, съёмки, просмотр, подсказка."""
        nonlocal msg, msg_t, show_help
        from . import dialogs as D
        from . import project as P_
        msg_t = time.monotonic()
        if live is not None:
            live.paused = True
        try:
            if item == "Open":
                ss = P_.all_samples(m.folder)
                rows = [f"{x.title:<44} {len(x.data.get('selections', [])):>3} sel  {len(P_.all_scans(x.folder)):>3} scans   "
                        f"{x.description}" for x in ss]
                cur = next((i for i, x in enumerate(ss) if x == m.sample), None)
                r_ = D.choose("Open a sample (project / sample)", rows, cur, ("Open", "New project...", "New sample..."))
                if r_ is None:
                    return
                if r_[0] == "Open":
                    m.sample = ss[r_[1]]
                    msg = f"Opened {m.sample.title}: {len(m._regions)} selections"
                else:
                    return menu_do(r_[0].rstrip('.'))
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
                lines = [f"Project    {sm_.project.name}", f"Sample     {sm_.name}",
                         "Selections " + (", ".join(f"{n} x {o}x" for o, n in sorted(objs.items())) or "none"),
                         f"Scans      {len(P_.all_scans(sm_.folder))}", f"Created    {sm_.data.get('created', '')}",
                         f"Folder     {sm_.folder}"]
                r_ = D.describe(sm_.title, lines, [("Project description", sm_.project.description),
                                                   ("Sample description", sm_.description)])
                if r_:
                    sm_.project.description, sm_.description = r_
                    msg = "Descriptions saved"
            elif item == "Scans":
                lst = m.list()
                if not lst:
                    msg = "No scans of this sample yet: mark a selection (1, 2) and press S"
                    return
                rows = [l_ for l_ in repr(lst).split("\n")[2:-1:2]]
                r_ = D.choose(f"Scans of {m.sample.title}", rows, len(rows) - 1, ("View",))
                if r_:
                    m._last_view = lst[r_[1]]['folder']
                    open_view()
            elif item == "View":
                open_view()
            elif item == "Help":
                show_help = not show_help
        except Exception as e:
            msg = f"{item}: {e}"
        finally:
            if live is not None:
                live.paused = False

    def scan_here():
        """S (дважды): снять выделения текущего объектива и сшить их, не выходя из окна. Полоса прогресса —
        в окне поверх кадра камеры; Esc — остановить после текущего кадра. Потом V — открыть в просмотре."""
        nonlocal msg, msg_t
        import threading
        from .util import Progress
        from .stitch import stitch as _stitch
        msg_t = time.monotonic()
        ks = [k_ for k_, rg in enumerate(m._regions) if SM.obj(m, rg) == m.objective]
        if not ks:
            msg = f"No selection for {m.objective}x: mark one with 1 and 2 (or switch the objective, M)"
            return
        nfr = sum(int(np.prod(SM.frames_of(m, k_))) for k_ in ks)
        fm_ = 'points' if m._use_fp else 'off'
        est = sum(TM.estimate(m, int(np.prod(SM.frames_of(m, k_))), fm_ if m._regions[k_]['pts'] else 'off')[0] for k_ in ks)
        if not (car.get('scan_ok') and time.monotonic() - car['scan_ok'] < 5):
            car['scan_ok'] = time.monotonic()
            msg = (f"Press S again: scan {len(ks)} selection{'s' if len(ks) > 1 else ''} of {m.objective}x, {nfr} frames, "
                   f"{TM.fmt(est)}, then stitch. Esc stops the scan")
            return
        car['scan_ok'] = None
        prog = dict(title="Scan", f=0.0, text="starting", eta=None)
        res, t0 = {}, time.monotonic()

        def hook(t, f, txt, el, eta):
            prog.update(title=t, f=f, text=txt, eta=eta)

        def work():
            try:
                folder = m.scan()
                res['folder'] = folder
                if res.get('stop'):
                    return
                subs_ = SM.samples(folder)
                for i_, sf in enumerate(subs_):
                    res['stitch'] = (i_ + 1, len(subs_))
                    _stitch(sf, verbose=True)
            except Exception as e:                          # показать в окне
                res['e'] = e

        m._abort = False
        if live is not None:
            live.paused = True                             # камера — скану
        Progress.hook = hook
        th = threading.Thread(target=work, daemon=True)
        th.start()
        try:
            while th.is_alive():
                fr_ = getattr(m, '_last_img', None)
                Wl_ = W - SM.CAR_W
                pic = render(fr_, Wl_, H, kz, zm['c'], 0.0) if fr_ is not None else np.zeros((H, Wl_, 3), np.uint8)
                pic = (pic.astype(np.float32) * 0.75).astype(np.uint8)
                sti = res.get('stitch')
                ttl = (f"Stitching {sti[0]} of {sti[1]}" if sti else f"Scanning {len(ks)} selection{'s' if len(ks) > 1 else ''}"
                       f" of {m.objective}x")
                stg = prog['title'].replace("Stitch: ", "").capitalize()
                UI.progress_panel(pic, ttl, stg, prog['f'], prog['text'], time.monotonic() - t0, prog['eta'],
                                  "stopping after this frame..." if res.get('stop') else ("Esc: stop" if not sti else ""))
                frame_ = np.zeros((H, W, 3), np.uint8)
                frame_[:, :Wl_] = pic
                frame_[:, Wl_:] = view[:, Wl_:] if view.shape[:2] == (H, W) else 30
                cv2.imshow(WIN, np.vstack([menu_img(W), frame_]))
                k__ = cv2.waitKeyEx(60) if keys is None else -1
                if (k__ == 27 or exit_key(k__)) and not res.get('stitch'):
                    m._abort, res['stop'] = True, True
        finally:
            Progress.hook = None
            th.join()
            if live is not None:
                live.paused = False
        msg_t = time.monotonic()
        if 'e' in res:
            msg = f"Scan / stitch failed: {res['e']}"
        elif res.get('stop'):
            msg = f"Scan stopped. Frames so far: {res.get('folder', '-')}"
        else:
            m._last_view = res['folder']
            msg = f"Scanned and stitched in {TM.fmt(time.monotonic() - t0)}: V opens it in the view"

    def open_view():
        """V: открыть последний скан в просмотре (кресты, квадраты, экспорт); закрыли — снова ручной режим."""
        nonlocal msg, msg_t
        from .viewer import view as _view
        f_ = getattr(m, '_last_view', None) or getattr(m, 'last_scan', None)
        msg_t = time.monotonic()
        if not f_ or not os.path.isdir(f_):
            msg = "Nothing scanned yet: mark a selection (1, 2) and press S"
            return
        if live is not None:
            live.paused = True
        try:
            _view(f_, fullscreen=fullscreen, _keys=keys)          # в тестах — те же нажатия
        finally:
            if live is not None:
                live.paused = False
        msg = "Back in manual mode"

    def go_region(k):
        nonlocal msg
        msg = SM.goto_area(m, k)

    def delete_region(k):
        nonlocal msg
        msg = SM.delete_area(m, k)
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
        v = float(int(float(buf)))
        lim = step_limit(m, ax)
        v5 = float(max(5, min(round(v / 5) * 5, lim // 5 * 5)))
        steps[ax] = v5
        msg = f"Step {ax.upper()} = {v5:g} um" + (f" (rounded to a multiple of 5)" if v5 != v and v5 < lim // 5 * 5 else "") + \
              (f" (the limit is {lim:g} um)" if v > lim else "")
        msg_t = time.monotonic()
        L.event('MAN-004', f"Manual step {ax.upper()} = {v5:g} um.")

    rul = m.__dict__.setdefault("_ruler", dict(on=False, pts=[]))   # линейка (L): точки на образце, настоящие мкм
    back_v = None

    def ruler_text(a, b):
        d = np.asarray(b, float) - np.asarray(a, float)
        return f"{np.hypot(*d):.2f} um   (dX {d[0]:+.2f}, dY {d[1]:+.2f})"

    # история (Ctrl+Z / Ctrl+Y): после каждого действия — сравнить состояние с прежним
    hist = m.__dict__.setdefault('_history', History())

    def h_sig():
        strip = lambda r: None if r is None else {k_: v_ for k_, v_ in r.items() if k_ != 'thumbs'}
        return json.dumps([[strip(r) for r in m._regions], strip(m._pending), m._sel, getattr(m, '_area_seq', 0),
                           m._p, m.objective, m._off], sort_keys=True, default=str)

    def h_snap():
        return dict(regions=copy.deepcopy(m._regions), pending=copy.deepcopy(m._pending), sel=m._sel,
                    seq=getattr(m, '_area_seq', 0), p=dict(m._p), obj=m.objective, off=dict(m._off))

    def h_label(a_, b_):
        if a_['p'] != b_['p'] and json.dumps(a_['regions'], default=str) == json.dumps(b_['regions'], default=str):
            s_ = m.motor_step
            return f"move to X {b_['p']['x'] * s_ - b_['off']['x']:.0f}, Y {b_['p']['y'] * s_ - b_['off']['y']:.0f}, " \
                   f"Z {b_['p']['z'] * s_ - b_['off']['z']:.0f} um"
        if a_['obj'] != b_['obj']:
            return f"objective {a_['obj']}x -> {b_['obj']}x"
        return (msg or "change").split(":")[0][:60]

    def h_restore(t):
        """Вернуть выделения, ноль, объектив и положение столика (переезд — тем же безопасным порядком,
        что Go: сначала вниз, потом X/Y отрезками, потом вверх)."""
        m._regions[:] = copy.deepcopy(t['regions'])
        m._pending, m._sel, m._area_seq = copy.deepcopy(t['pending']), t['sel'], t['seq']
        if t['obj'] != m.objective:
            m.objective = t['obj']
            d0 = default_steps(m)
            steps['x'], steps['y'] = d0['x'], d0['y']
        m._off.update(t['off'])
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

    h_last = (h_sig(), h_snap())
    while True:
        sg_ = h_sig()
        if sg_ != h_last[0]:                              # что-то сделано: в историю
            new_ = h_snap()
            hist.push(h_label(h_last[1], new_), h_last[1], new_)
            h_last = (sg_, new_)
        need = True
        if live is not None:                               # перерисовать: новый кадр, события мыши или раз в 0.3 с
            need = live.n != last_n or bool(clicks) or time.monotonic() - last_t > 0.3
        if need:
            img = live.img if live is not None else m.image(0)
            last_n, last_t = (live.n if live is not None else last_n), time.monotonic()
            if m._pending:                                 # карта и строка состояния — по новой или выбранной области
                A, B, zk = work(m._pending['A']), None, len(m._regions)
            elif m._sel is not None and SM.obj(m, m._regions[m._sel]) == m.objective:
                A, B, zk = work(m._regions[m._sel]['A']), work(m._regions[m._sel]['B']), m._sel
            else:                                          # выделение другого объектива — только контуром на карте
                A = B = zk = None
            if n % 4 == 0:                                     # резкость по тому же кадру (не дёргаем камеру ещё раз)
                sharp = round(F_.sharpness(img, m.focus_method), 1)
            n += 1
            W, H = (SW, SH) if fullscreen else window_size(WIN, (W, H + UI.MENU_H))
            H -= UI.MENU_H                                     # сверху — строка меню
            rot = 0.0
            Wl = W - SM.CAR_W                                  # картинка — слева от карусели, по центру
            to_v, back_v, i2v, v2i, cimg, kz, zm['c'] = live_map(m, img.shape, Wl, H, zm['z'], zm['c'], rot)
            base = render(img, Wl, H, kz, zm['c'], rot)
            view = np.zeros((H, W, 3), np.uint8)
            view[:, :Wl] = base
            try:                                           # выбранные в просмотре квадраты — тёмно-зелёным на кадре
                from .selections import marked_all
                polys_ = marked_all(m.dirs['scans'])
            except Exception:
                polys_ = []
            if polys_:
                rp_ = np.array([m.real_position.x, m.real_position.y])
                rad_ = 1.5 * max(m.frame_size) / max(1.0, zm['z']) + 2 * max(m.frame_size)
                ov_ = None
                for q in polys_:
                    if np.abs(q.mean(0) - rp_).max() > rad_:
                        continue
                    pts = np.int32([to_v(S_) for S_ in q])
                    if ov_ is None:
                        ov_ = view[:, :Wl].copy()
                    cv2.fillPoly(ov_, [pts], (30, 90, 30))
                    cv2.polylines(view[:, :Wl], [pts], True, (60, 170, 60), 1, cv2.LINE_AA)
                if ov_ is not None:
                    view[:, :Wl] = cv2.addWeighted(ov_, 0.3, view[:, :Wl], 0.7, 0)
            if rul['on'] and rul['pts']:                    # линейка (точки — на образце: едут вместе с ним)
                a_ = to_v(rul['pts'][0])
                if len(rul['pts']) == 2:
                    b_, q2 = to_v(rul['pts'][1]), rul['pts'][1]
                elif hover[0] is not None and hover[0][0] < Wl:
                    b_ = (hover[0][0] + 0.5, hover[0][1] + 0.5)
                    q2 = back_v(b_)
                else:
                    b_ = q2 = None
                UI.ruler(view, a_, b_, ruler_text(rul['pts'][0], q2) if q2 is not None else "")
            if m.laser:                                    # лазер включён — видно всегда
                t_ = f"LASER ON  {m.laser:g} mW   (I: off)"
                tw_ = UI.size(t_, 0.7, 2)[0]
                x_ = int((Wl - tw_) / 2)
                cv2.rectangle(view, (x_ - 14, 12), (x_ + tw_ + 14, 52), (20, 20, 200), -1)
                UI.put(view, t_, x_, 42, 0.7, (255, 255, 255), 2)
            p = m.position
            sc = max(0.7, H / 1100)
            line = f"Sharpness {sharp:.0f}"
            if A and B:
                r, c, ny, nx = cell_of(m, A, B)
                line += f"    Selection {SM.num(m, zk)} ({m.objective}x): {nx} x {ny} = {nx * ny} frames" + \
                        (f"    Frame: row {r + 1}/{ny}, column {c + 1}/{nx}" if r is not None else "    Outside the area")
            else:
                *_, (ci_, cj_), _z = _grid(m, A, B)
                b0c, b0r = (_z[0], _z[2]) if _z else (0, 0)
                line += f"    Frame: row {cj_ - b0r + 1}, column {ci_ - b0c + 1}" + \
                        (f"    Selection {SM.num(m, zk)}: corner 1 set, press 2 at the other corner" if A else
                         f"    Key 1: corner of a new selection ({m.objective}x)")
            q = i2v(cimg + np.array([gd['u'], gd['v']]))                 # точка направляющих в окне

            # левая панель: координаты крупно, поля ввода в один столбец, точки фокуса цветом своей области
            FS = 0.55 * sc
            th0 = UI.size("Xg", FS)[1]
            LH = th0 + 16
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
            cur_ = dict(steps, exp=m.exposure, obj=m.objective)
            for ax_ in FIELDS:
                lab_ = {'exp': "Exposure", 'obj': "Objective"}.get(ax_, f"Step {ax_.upper()}")
                unit_ = {'exp': "ms", 'obj': "x"}.get(ax_, "um")
                hint_ = {'z': "Shift: x5", 'obj': "M: next"}.get(ax_, "")
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
            ops.append(('t', "Y", cb[2] + 22 + UI.size(t_, FS)[0], y, FS, UI.ACCENT, 1))
            rects['usefp'] = (x_l - 6, cb[1] - 4, cb[2] + 40 + UI.size(t_, FS)[0], cb[3] + 4)
            right = max(right, rects['usefp'][2], rects['obj'][2] + 60)
            y += LH
            lt_ = f"Light {'on' if m.light else 'off'}"   # свет и лазер: состояние и клавиши
            ops.append(('t', lt_, x_l, y, FS, UI.TEXT if m.light else UI.DIM, 1))
            ops.append(('t', "L", x_l + UI.size(lt_ + "  ", FS)[0], y, FS, UI.ACCENT, 1))
            lz_ = f"Laser {m.laser:g} mW" if m.laser else "Laser off"
            xz_ = x_l + UI.size(lt_ + "  L      ", FS)[0]
            ops.append(('t', lz_, xz_, y, FS, (80, 80, 255) if m.laser else UI.DIM, 1 if not m.laser else 2))
            ops.append(('t', "I", xz_ + UI.size(lz_ + "  ", FS, 2)[0], y, FS, UI.ACCENT, 1))
            right = max(right, xz_ + UI.size(lz_ + "  I", FS, 2)[0])
            y += LH
            fp = SM.numbered(m)
            ops.append(('l', y - LH // 2 + 4))
            y += 6
            if fp:
                ops.append(("t", "Focus points   (K: add one here)", x_l, y, FS, UI.DIM, 1))
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
            else:
                ops.append(('t', "Focus points: keys 1, 2 keep Z of the corners, K keeps Z here", x_l, y, FS * 0.9,
                            UI.DIM, 1))
                right = max(right, x_l + UI.size("Focus points: keys 1, 2 keep Z of the corners, K keeps Z here",
                                                 FS * 0.9)[0])
                y += LH
            if zm['z'] > 1:
                ops.append(('t', f"Zoom x{zm['z']:.2f}   Z: whole frame", x_l, y, FS * 0.9, UI.ACCENT, 1))
                y += LH
            hidden_ = getattr(m, '_panel_hidden', False)
            hov_ = lambda r_: hover[0] is not None and r_[0] <= hover[0][0] <= r_[2] and r_[1] <= hover[0][1] <= r_[3]
            if hidden_:                                      # панель спрятана: «ручка» у левого края
                tw_ = int(26 * sc)
                th_ = int(72 * sc)
                rects.clear()                                # невидимые поля не нажимаются
                rects['panel'] = (0, 12, tw_, 12 + th_)
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
            if show_help:                                    # H: все клавиши — панель по центру
                fsh = 0.52 * sc
                lh = UI.size("Xg", fsh)[1] + 12
                ks = [h_.partition(": ") for h_ in HELP]
                wl = max(UI.size(k_, fsh)[0] for k_, _s, _d in ks) + 22
                wh = min(Wl - 40, wl + max(UI.size(d_, fsh)[0] for _k, _s, d_ in ks) + 40)
                hh = lh * len(HELP) + 54
                x0_, y0_ = int((Wl - wh) / 2), max(10, int((H - hh) / 2))
                UI.panel(view, x0_, y0_, x0_ + wh, y0_ + hh, 0.92, UI.LINE)
                UI.put(view, "Keys", x0_ + 18, y0_ + 30, 0.6 * sc, UI.ACCENT, 2)
                for k__, (k_, _s, d_) in enumerate(ks):
                    yy = y0_ + 30 + (k__ + 1) * lh
                    UI.put(view, k_, x0_ + 18, yy, fsh, UI.ACCENT)
                    UI.put(view, d_, x0_ + 18 + wl, yy, fsh, UI.TEXT)
            # низ: строка состояния; над ней — сообщение
            bh_ = th0 + 20
            UI.panel(view, 0, H - bh_, Wl, H, 0.82)
            UI.facts(view, [t_.strip() for t_ in line.split("    ") if t_.strip()] + ["H: keys"], 14, H - 10, FS)
            if msg and time.monotonic() - msg_t < 5:
                UI.toast(view, msg, Wl / 2, H - bh_ - 20, FS, 'err' if msg.startswith('[E') else 'ok', Wl - 60)
            others = [(tuple(np.r_[work(reg_rect(rg)[:2]), work(reg_rect(rg)[2:])]), SM.col(m, k_),
                       f"{SM.obj(m, rg)}x" if SM.obj(m, rg) != m.objective else "")
                      for k_, rg in sorted(enumerate(m._regions), key=lambda t: SM.obj(m, t[1])) if k_ != zk]
            minimap(view, m, A, B, zcolor=SM.col(m, zk) if zk is not None else None, others=others, right=SM.CAR_W)
            items = car_items()
            selk_ = len(m._regions) if m._pending else m._sel
            sel_ = car['order'].index(selk_) if selk_ in car['order'] else None
            hx_ = hover[0]
            hv_ = (hx_[0] - (W - SM.CAR_W), hx_[1]) if hx_ is not None and hx_[0] >= W - SM.CAR_W else None
            pan_, hits_, cont_ = SM.carousel(H, items, sel_, car['scroll'], title="Selections", hover=hv_)
            car['hits'], car['content'] = hits_, cont_
            view[:, W - SM.CAR_W:] = pan_
            cv2.imshow(WIN, np.vstack([menu_img(W), view]))
            if n == 1:                                    # окно уже на экране: значок программы (Windows)
                set_icon(WIN)
        key = next(keys, EXIT) if keys is not None else cv2.waitKeyEx(8 if live is not None else 30)
        if keys is not None:
            cv2.waitKey(1)
            if isinstance(key, tuple):                    # тест: щелчок мышью (x, y)
                clicks.append(key); key = -1
        while clicks:                                     # щелчок по полю — ввод значения; по картинке — направляющие
            c_ = clicks.pop(0)
            if c_[0] == 'wheel':                                    # зум к курсору
                pix = v2i(c_[1:3])
                zm['z'] = min(8.0, max(1.0, zm['z'] * (1.25 if c_[3] > 0 else 0.8)))
                Wl_ = W - SM.CAR_W
                k2 = min(Wl_ / img.shape[1], H / img.shape[0]) * zm['z']
                zm['c'] = pix - (np.array(c_[1:3], float) - (Wl_ / 2, H / 2)) / k2
                continue
            if c_[0] == 'menu':                                     # строка меню
                it_ = next((t_ for x0_, x1_, t_ in menu['hits'] if x0_ <= c_[1] <= x1_), None)
                if it_:
                    menu_do(it_)
                continue
            if c_[0] == 'carw':                                     # прокрутка карусели
                car['scroll'] = int(max(0, min(car.get('content', 0) - H + 20, car['scroll'] - 60 * np.sign(c_[3]))))
                continue
            if c_[0] == 'car':                                      # щелчок по карусели
                h_ = SM.hit(car['hits'], c_[1] - (W - SM.CAR_W), c_[2])
                if h_:
                    act, k = h_
                    k = car['order'][k] if k < len(car.get('order', [])) else k
                    if act == 'sel':
                        car['confirm'] = None
                        if k < len(m._regions):
                            m._sel = k
                            msg, msg_t = (f"Selection {SM.num(m, k)} chosen ({SM.layer_name(m, k)}; Go or Enter: "
                                          "drive there)"), time.monotonic()
                    elif act == 'del':
                        if car['confirm'] == k:
                            delete_region(k)
                        else:
                            car['confirm'] = k
                            msg = f"Click x again to delete Selection {SM.num(m, k)}"
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
                key, edit = ord('y'), None
                break
            if hit and hit[0] == 'panel':                # кнопка «спрятать / показать панель»
                key, edit = ord('p'), None
                break
            edit = (hit[0], "") if hit else None
        if key == -1:
            if keys is None and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            try:
                m._keepalive()               # простой: запрос «на связи», чтобы Windows не усыпляла USB
            except MicroControlError as e:
                msg, msg_t = str(e), time.monotonic()
            continue
        if exit_key(key):                                 # Shift+Esc / Ctrl+Esc — выйти
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
        if key == 9:                                      # Tab: следующее поле (X -> Y -> Z -> выход из ввода)
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
            elif key in (46, 44) and edit[0] == 'exp' and '.' not in edit[1]:   # дробная экспозиция: . или ,
                edit = (edit[0], edit[1] + '.')
            elif key == 8:
                edit = (edit[0], edit[1][:-1])
            elif key in (13, 10):
                commit(edit); edit = None
            elif key == 27:
                edit = None
            continue
        raw = chr(key) if 0 <= key < 128 else ''
        ch = raw.lower()
        ax = m._screen_axes()
        (ah, sh), (av, sv) = ax['h'], ax['v']
        sh, sv = (-sh if m.arrows_flip_x else sh), (-sv if m.arrows_flip_y else sv)   # только клавиши
        hold = live.hold() if live is not None else None
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
                msg = "Panel hidden: P or the button shows it" if m._panel_hidden else ""
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
                       "follows them (remembered)")
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
                msg = f"Orientation: {msg}"
            elif ch == 'e' or key in PGUP:
                dz = steps['z'] * (SHIFT if raw == 'E' else 1)
                m.move(dz=dz); msg = f"Stage up {dz:g} um"
            elif ch == 'q' or key in PGDN:
                dz = steps['z'] * (SHIFT if raw == 'Q' else 1)
                m.move(dz=-dz); msg = f"Stage down {dz:g} um"
            elif ch == 'm':
                from .config import OBJECTIVES
                obs = sorted(OBJECTIVES)
                m.objective = obs[(obs.index(m.objective) + 1) % len(obs)]
                d0 = default_steps(m)
                steps['x'], steps['y'] = d0['x'], d0['y']
                msg = f"Objective {m.objective}x: put it in place. Steps X, Y = one frame: {steps['x']:g}, {steps['y']:g} um"
            elif ch == 'g':
                d0 = default_steps(m)
                steps['x'], steps['y'] = d0['x'], d0['y']
                msg = f"Steps X, Y = one frame: {steps['x']:g}, {steps['y']:g} um"
            elif ch == 'a':                                # автофокус
                r = m.focus()
                msg = f"Focused at Z {r['Z']}" + ("  (weak peak)" if 'Note' in r else "")
            elif ch == 'k':                                # запомнить Z здесь как точку фокуса выделения
                msg = SM.add_point(m)
                L.event('MAN-006', msg + ".")
            elif ch in 'lf' and ch:                        # L (и F): свет вкл. / выкл.
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
                m.photo()
                msg = f"Picture saved: {m.last_photo}"
            elif raw in ('0', ')'):                        # выделения хранятся в настоящих мкм: остаются на месте
                if raw == '0':
                    m.offset()
                    msg = "Zero is here. Real position " + repr(m.real_position)
                else:
                    m.offset_clear()
                    msg = "Offset removed: real coordinates"
            elif ch == 'w':                                # баланс белого (навести на пустое место)
                m.white_balance()
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
                elif m._sel is not None:
                    if car['confirm'] == m._sel and time.monotonic() - car['t'] < 4:
                        delete_region(m._sel)
                    else:
                        car['confirm'], car['t'] = m._sel, time.monotonic()
                        msg = f"Press C again to delete Selection {SM.num(m, m._sel)}"
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
    if not fullscreen:
        window_size(WIN, remember=True)                                  # запомнить размер для следующего окна
    cv2.destroyWindow(WIN)
    cv2.waitKey(1)
    regs = m._regions
    m.area = reg_area(regs[m._sel if m._sel is not None else 0]) if regs else None
    m._focus_pts = [dict(p_, tag=str(n_)) for n_, k_, p_ in SM.numbered(m) if k_ == (m._sel or 0)] if regs else []
    L.event('MAN-002', f"Manual mode closed at {m.position}. Selections: {len(regs)}.")
    if not regs:
        print("Manual mode closed. No selection: mark two corners with keys 1 and 2.")
        return None
    lines_ = []
    for k, depth in SM.tree(m):                                          # деревом слоёв: 10x > 50x > 100x
        rg = regs[k]
        o_ = SM.obj(m, rg)
        nx, ny = SM.frames_of(m, k)
        lines_.append(f"  {'    ' * depth}{'└ ' if depth else ''}Selection {rg['num']}  {o_}x  {reg_area(rg)}   "
                      f"{nx} x {ny} = {nx * ny} frames   focus points {first_num(k)}")
    print("Manual mode closed." + (" The new selection without corner 2 is kept for the next time." if m._pending else "")
          + "\n" + "\n".join(lines_) + f"\n  m.scan() shoots the selections of {m.objective}x, each into its own "
          "folder; the others wait for their objective.")
    return m.area
