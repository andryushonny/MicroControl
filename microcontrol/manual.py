"""Ручной режим: живая картинка на весь экран, движение по кадрам, отметка зоны (как «приват» в Minecraft).

Координаты знать не нужно: доезжаете до края нужного участка и отмечаете угол.
Мини-карта в углу показывает отмеченную зону и где вы сейчас.

  Стрелки / W A S D   На один кадр в сторону (или на долю кадра, см. [ ])
  [  ]                Шаг мельче / крупнее: 1, 1/2, 1/5, 1/10 кадра
  E / Q (PgUp/PgDn)   Столик вверх / вниз на шаг фокуса
  F                   Автофокус
  1 / 2               Угол зоны A / B (центр текущего кадра)
  C                   Стереть зону
  Пробел              Снимок в photos/
  0                   Ноль здесь: текущая точка становится (0, 0, 0), как m.offset()
  R                   Убрать сдвиг нуля: снова настоящие координаты, как m.offset_clear()
  B                   Баланс белого (наведите на пустой участок), как m.white_balance()
  L                   Свет вкл / выкл
  H                   Подсказка
  Esc / Enter         Выход
"""
import time
import cv2
import numpy as np
from . import log as L
from .errors import MicroControlError
from .util import wheel_delta, open_window, window_size
from . import focus as F_
import threading
from .scan import Area
from .config import scan_overlap

WIN = "MicroControl"
HELP = ["Esc: Exit", "Arrows / WASD: Move by Step X / Step Y", "E / Q: Stage up / down by Step Z (Shift: x5)",
        "Tab or click a field: steps (multiple of 5 um), exposure (ms), objective; Enter",
        "G: Steps X, Y = one frame", "M: Next objective (4x, 10x, 20x, 100x)",
        "1 / 2: Area corner A / B (their Z is kept as focus points)", "F: Keep Z here as a focus point   Shift+F: Autofocus",
        "Y (or the box): Autofocus by corners during the scan", "C: Clear the area and focus points", "Space: Picture",
        "0: Zero here (offset)", "R: Real coordinates (remove the offset)", "B: White balance (aim at an empty place)",
        "Wheel: zoom   Middle button or Shift+drag: move the zoomed picture   Z: whole frame",
        "[ / ]: Turn the mini map over left-right / up-down (remembered)",
        "O: check with the camera which way the picture moves", "L: Light on / off", "H: Hide this help"]
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
    out = np.zeros((H, W, 3), np.uint8)
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
            cv2.circle(img, (int(u), int(v)), 4, RED, 1, cv2.LINE_AA)


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
        out = np.zeros((H, W, 3), np.uint8)
        if a1 <= a0 or b1 <= b0:
            return out
        M = np.float32([[k, 0, (a0 - x0) * k], [0, k, (b0 - y0) * k]])
        part = cv2.warpAffine(img[b0:b1, a0:a1], M, (W, H), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        return part if part.ndim == 3 else cv2.cvtColor(part, cv2.COLOR_GRAY2BGR)
    src, kk, c = img, k, np.asarray(zc, float)
    if k < 1:                                           # сначала уменьшить без ряби, потом сдвинуть и повернуть
        src = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        kk, c = 1.0, c * k
    t = np.radians(rot)
    R = kk * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    M = np.float32(np.c_[R, np.array([W / 2, H / 2]) - R @ c])
    out = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR)
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
    sgh = -sgh if getattr(m, 'map_flip_x', False) else sgh        # отражение мини-карты (помнится для прибора;
    sgv = -sgv if getattr(m, 'map_flip_y', False) else sgv        # в окне — клавиши [ и ])
    st = m._frame_step(1 - scan_overlap(m.objective))
    fw, fh = m.frame_size
    size = {'x': fw, 'y': fh}
    f = lambda x, y: (sgh * {'x': x, 'y': y}[ah], sgv * {'x': x, 'y': y}[av])
    return f, st[ah], st[av], size[ah], size[av]


def snap(m, A, pt):
    """Точка -> центр ближайшей клетки решётки с началом в A (угол B встаёт ровно на сетку). Оси столика."""
    st = m._frame_step(1 - scan_overlap(m.objective))
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


def minimap(img, m, A, B, size=MAP_W):
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
    X0, Y0 = img.shape[1] - W - 12, 12
    sub = img[Y0:Y0 + H, X0:X0 + W]
    under = sub.copy()                                                   # для полупрозрачности
    sub[:] = (sub * 0.4).astype(sub.dtype)
    gx0, gy0 = 46, 30
    gw, gh = MAP_COLS * sx * k, MAP_ROWS * sy * k

    def to(hh, vv):         # экранные мкм -> пиксели карты
        return int(gx0 + (hh - (ox + (i0 - 0.5) * sx)) * k), int(gy0 + (vv - (oy + (j0 - 0.5) * sy)) * k)

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
        fill, line = ((40, 95, 40), (80, 200, 80)) if fixed else ((40, 80, 60), (120, 220, 170))
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
    a_, b_ = to(px - w / 2, py - h / 2), to(px + w / 2, py + h / 2)      # жёлтое — ровно поле зрения
    ov_ = sub.copy()
    cv2.rectangle(ov_, (max(a_[0], gx0), max(a_[1], gy0)), (min(b_[0], gx0 + int(gw)), min(b_[1], gy0 + int(gh))),
                  (0, 200, 255), -1)
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
        label(f"{'Area' if fixed else 'Preview'} {nx_} x {ny_} = {nx_ * ny_}" + (f"    hidden {hid}" if hid else ""),
              (W / 2, H - 12), line, 0.45)
    else:
        label(f"Frame: row {cj + 1}, column {ci + 1}    Key 1: corner A", (W / 2, H - 12), (170, 170, 170), 0.45)
    for c, name in ((A, "A"), (B, "B")):
        if c:
            u, v = to(*f(*c))
            if gx0 <= u <= gx0 + gw and gy0 <= v <= gy0 + gh:
                cv2.putText(sub, name, (u - 4, v + 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cr = getattr(m, '_crosses', None)
    if cr is not None and cr.marks:                                      # решётка крестов (красная)
        offv = np.array([m._off['x'], m._off['y']])
        draw_lattice(sub, cr, lambda S: to(*f(*(np.asarray(S) - offv))), (gx0, gy0, int(gw), int(gh)), solid=True)
    for f_ in getattr(m, '_focus_pts', None) or []:                      # точки фокуса (голубые)
        u, v = to(*f(f_['x'] - m._off['x'], f_['y'] - m._off['y']))
        if gx0 <= u <= gx0 + gw and gy0 <= v <= gy0 + gh:
            cv2.circle(sub, (u, v), 4, (255, 220, 80), -1, cv2.LINE_AA)
            cv2.putText(sub, f_['tag'], (u + 5, v - 4), font, 0.4, (255, 220, 80), 1, cv2.LINE_AA)
    cv2.rectangle(sub, to(px - w / 2, py - h / 2), to(px + w / 2, py + h / 2), (255, 255, 255), 2)   # поле зрения
    cv2.rectangle(sub, (gx0, gy0), (gx0 + int(gw), gy0 + int(gh)), (120, 120, 120), 1)
    sub[:] = cv2.addWeighted(sub, 0.85, under, 0.15, 0)                 # карта слегка прозрачная
    if zone and (c1 - c0 >= MAP_COLS or r1 - r0 >= MAP_ROWS):          # обзор всей зоны
        nx_, ny_ = c1 - c0 + 1, r1 - r0 + 1
        ok = 120 / max(nx_ * sx, ny_ * sy)
        oW, oH = max(8, int(nx_ * sx * ok)), max(8, int(ny_ * sy * ok))
        oxp, oyp = img.shape[1] - 12 - oW - 4, Y0 + H + 6
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
    """Шаги ручного режима, мкм, кратные 5: X, Y — кадр (округлён вниз до 5 мкм и не больше max_move), Z — 100."""
    st = m._frame_step()
    r5 = lambda v, lim: float(max(5, min(lim, v) // 5 * 5))
    return {'x': r5(st['x'], m.max_move), 'y': r5(st['y'], m.max_move), 'z': r5(Z_STEP, m.max_move_z)}


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


def run(m, fullscreen=True, _keys=None):
    m._need_cam()
    WH0 = open_window(WIN, fullscreen)
    if fullscreen:
        SW, SH = screen_size()
    L.event('MAN-001', f"Manual mode at {m.position}.")
    show_help, msg, msg_t, sharp, n = False, "H: Help.  Tab or click a field to edit it", time.monotonic(), 0.0, 0
    A = B = None
    if m.area is not None:
        A, B = (m.area.x0, m.area.y0), (m.area.x1, m.area.y1)
    elif getattr(m, '_corner_a', None):                  # угол A из прошлого входа (хранится в настоящих координатах)
        A = (m._corner_a[0] - m._off['x'], m._corner_a[1] - m._off['y'])
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

    def on_mouse(ev, x, y, flags, *_):
        # тянуть любой кнопкой по картинке (левой — не по полям) — сдвиг увеличенной картинки; колесо, + / - — зум
        if ev == cv2.EVENT_MOUSEMOVE:
            hover[0], hover[1] = (x, y), time.monotonic()
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

    while True:
        need = True
        if live is not None:                               # перерисовать: новый кадр, события мыши или раз в 0.3 с
            need = live.n != last_n or bool(clicks) or time.monotonic() - last_t > 0.3
        if need:
            img = live.img if live is not None else m.image(0)
            last_n, last_t = (live.n if live is not None else last_n), time.monotonic()
            if n % 4 == 0:                                     # резкость по тому же кадру (не дёргаем камеру ещё раз)
                sharp = round(F_.sharpness(img, m.focus_method), 1)
            n += 1
            W, H = (SW, SH) if fullscreen else window_size(WIN, (W, H))
            rot = 0.0
            to_v, back_v, i2v, v2i, cimg, kz, zm['c'] = live_map(m, img.shape, W, H, zm['z'], zm['c'], rot)
            base = render(img, W, H, kz, zm['c'], rot)
            view = base.copy()
            p = m.position
            sc = max(0.7, H / 1100)
            line = f"X {p.x:.0f}  Y {p.y:.0f}  Z {p.z:.0f} um" + ("  (zero moved)" if any(m._off.values()) else "") + \
                   f"    Sharpness {sharp:.0f}"
            if A and B:
                r, c, ny, nx = cell_of(m, A, B)
                line += f"    Area {nx} x {ny} = {nx * ny} frames" + \
                        (f"    Frame: row {r + 1}/{ny}, column {c + 1}/{nx}" if r is not None else "    Outside the area")
            else:
                *_, (ci_, cj_), _z = _grid(m, A, B)
                b0c, b0r = (_z[0], _z[2]) if _z else (0, 0)
                line += f"    Frame: row {cj_ - b0r + 1}, column {ci_ - b0c + 1}" + \
                        (f"    Corner {'A' if A else 'B'} set, mark the other one" if (A or B) else "")
            q = i2v(cimg + np.array([gd['u'], gd['v']]))                 # точка направляющих в окне

            def lines(im):
                pass
            lines(view)
            # все надписи слева — один размер шрифта FS и один шаг строк LH
            FS = 0.55 * sc
            th0 = cv2.getTextSize("Xg", cv2.FONT_HERSHEY_SIMPLEX, FS, 1)[0][1]
            LH = th0 + 18
            y_ = [th0 + 12]

            def row(s_, color=(235, 235, 235), border=None, key=None):
                (tw, _), _ = cv2.getTextSize(s_, cv2.FONT_HERSHEY_SIMPLEX, FS, 1)
                r_ = (8, y_[0] - th0 - 7, 16 + tw, y_[0] + 7)
                cv2.rectangle(view, r_[:2], r_[2:], (0, 0, 0), -1)
                if border:
                    cv2.rectangle(view, r_[:2], r_[2:], border, 1)
                cv2.putText(view, s_, (12, y_[0]), cv2.FONT_HERSHEY_SIMPLEX, FS, color, 1, cv2.LINE_AA)
                if key:
                    rects[key] = r_
                y_[0] += LH

            rects = {}
            row(f"X {p.x:.1f}   Y {p.y:.1f}   Z {p.z:.1f} um", (255, 255, 255))
            cur_ = dict(steps, exp=m.exposure, obj=m.objective)
            for ax_ in FIELDS:
                val = (edit[1] + "_") if (edit and edit[0] == ax_) else f"{cur_[ax_]:g}"
                txt = {'exp': f"Exposure  [ {val:>6} ] ms", 'obj': f"Objective [ {val:>6} ] x   (4, 10, 20, 100; M: next)"}.get(
                    ax_, f"Step {ax_.upper()}    [ {val:>6} ] um" + ("   Shift: x5" if ax_ == 'z' else ""))
                act = edit and edit[0] == ax_
                row(txt, (0, 220, 255) if act else (235, 235, 235), (0, 200, 255) if act else (90, 90, 90), ax_)
            row(f"[{'x' if m._use_fp else '  '}] Autofocus by corners: Z of each frame from the focus points (weighted)   Y",
                (120, 230, 255) if m._use_fp else (235, 235, 235), (90, 90, 90), 'usefp')
            if zm['z'] > 1:
                row(f"Zoom x{zm['z']:.2f}   wheel or + / -: zoom, drag the picture: move, Z: whole frame", (200, 200, 200))
            fp = m._focus_pts
            row(("Focus points: " + "   ".join(f"{f_['tag']}: Z {f_['z'] - m._off['z']:.1f}" for f_ in fp)
                 + "   (scan takes Z from them)") if fp else
                "Focus points: none (keys 1, 2 keep Z of the corners; F keeps Z here)", (230, 220, 120))
            if msg and time.monotonic() - msg_t < 4:
                row(msg, (90, 90, 255) if msg.startswith('[E') else (120, 230, 120))
            if show_help:
                for s_ in HELP:
                    if y_[0] > H - LH - 10:
                        break
                    row(s_)
            y_[0] = H - 12                                               # строка состояния — тем же шрифтом
            row(line)
            minimap(view, m, A, B)
            cv2.imshow(WIN, view)
        key = next(keys, 27) if keys is not None else cv2.waitKeyEx(8 if live is not None else 30)
        if keys is not None:
            cv2.waitKey(1)
            if isinstance(key, tuple):                    # тест: щелчок мышью (x, y)
                clicks.append(key); key = -1
        while clicks:                                     # щелчок по полю — ввод значения; по картинке — направляющие
            c_ = clicks.pop(0)
            if c_[0] == 'wheel':                                    # зум к курсору
                pix = v2i(c_[1:3])
                zm['z'] = min(8.0, max(1.0, zm['z'] * (1.25 if c_[3] > 0 else 0.8)))
                k2 = min(W / img.shape[1], H / img.shape[0]) * zm['z']
                zm['c'] = pix - (np.array(c_[1:3], float) - (W / 2, H / 2)) / k2
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
            if edit:
                commit(edit)
            if hit and hit[0] == 'usefp':
                key, edit = ord('y'), None
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
            if key == 27:
                break
            elif key in LEFT or ch == 'a':
                m.move(**{'d' + ah: -sh * steps[ah]}); msg = ''
            elif key in RIGHT or ch == 'd':
                m.move(**{'d' + ah: sh * steps[ah]}); msg = ''
            elif key in UP or ch == 'w':
                m.move(**{'d' + av: -sv * steps[av]}); msg = ''
            elif key in DOWN or ch == 's':
                m.move(**{'d' + av: sv * steps[av]}); msg = ''
            elif raw in ('[', ']'):                        # отразить мини-карту по горизонтали / вертикали
                k_ = 'map_flip_x' if raw == '[' else 'map_flip_y'
                setattr(m, k_, not getattr(m, k_))
                msg = f"Mini map turned over {'left-right' if raw == '[' else 'up-down'} (remembered)"
            elif ch == 'y':
                m._use_fp = not m._use_fp
                msg = f"Autofocus by corners {'on' if m._use_fp else 'off'}"
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
            elif raw == 'F':                               # Shift+F: автофокус
                r = m.focus()
                msg = f"Focused at Z {r['Z']}" + ("  (weak peak)" if 'Note' in r else "")
            elif ch == 'f':                                # F: запомнить Z здесь как точку фокуса
                rp = m.real_position
                m._focus_pts.append(dict(x=rp.x, y=rp.y, z=rp.z, tag=str(len(m._focus_pts) + 1)))
                msg = f"Focus point {len(m._focus_pts)}: Z {p.z:.1f} um at X {p.x:.0f}, Y {p.y:.0f}"
                L.event('MAN-006', msg + ".")
            elif ch == ' ':
                m.photo()
                msg = f"Picture saved: {m.last_photo}"
            elif ch in '0r':
                old = dict(m._off)
                if ch == '0':
                    m.offset()
                    msg = "Zero is here. Real position " + repr(m.real_position)
                else:
                    m.offset_clear()
                    msg = "Offset removed: real coordinates"
                dx, dy = old['x'] - m._off['x'], old['y'] - m._off['y']   # углы зоны остаются на месте
                A = (A[0] + dx, A[1] + dy) if A else None
                B = (B[0] + dx, B[1] + dy) if B else None
            elif ch == 'b':
                m.white_balance()
                msg = "White balance done"
            elif ch == 'l':
                m.light = not m.light
                msg = "Light on" if m.light else "Light off"
            elif ch == 'h':
                show_help = not show_help
            elif ch == 'c':
                A = B = None
                m.area = None
                m._focus_pts.clear()
                msg = "Area and focus points cleared"
            elif ch in '12':
                pt = (p.x, p.y)
                if ch == '1':
                    A = pt
                    if B:
                        B = snap(m, A, B)                 # B — ровно на сетке кадров от A
                else:
                    B = pt = snap(m, A, pt) if A else pt
                rp = m.real_position                      # Z угла — точка фокуса (A, B)
                tag = 'A' if ch == '1' else 'B'
                m._focus_pts[:] = [f_ for f_ in m._focus_pts if f_['tag'] != tag]
                m._focus_pts.insert(0 if tag == 'A' else min(1, len(m._focus_pts)),
                                    dict(x=rp.x, y=rp.y, z=rp.z, tag=tag))
                msg = f"Corner {tag} at X {pt[0]:.0f}, Y {pt[1]:.0f} um, focus Z {p.z:.1f} um"
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
    if A and B:
        m.area = Area(*A, *B)
    m._corner_a = (A[0] + m._off['x'], A[1] + m._off['y']) if (A and not B) else None
    L.event('MAN-002', f"Manual mode closed at {m.position}. Area: {m.area}.")
    if m.area is None:
        print("Manual mode closed. The area is not set: mark two corners with keys 1 and 2.")
        return None
    nx, ny = m.area.size(m._frame_step(1 - scan_overlap(m.objective)))
    print(f"Manual mode closed.\n  Area     {m.area}\n  Frames   {nx} x {ny} = {nx * ny}")
    return m.area
