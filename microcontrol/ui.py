"""Общий вид окон: полупрозрачные панели, «чипы», строка подсказок с выделенными клавишами, всплывающее сообщение.
Один шрифт, одни цвета и отступы — в ручном режиме и в просмотре."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT = (238, 238, 238)
DIM = (165, 165, 165)
ACCENT = (0, 205, 255)          # жёлтый (BGR): активное, клавиши
OK = (110, 215, 110)
ERR = (90, 90, 255)
PANEL = (26, 27, 31)
LINE = (78, 80, 88)
PAD = 10


def size(s, scale, th=1):
    (w, h), b = cv2.getTextSize(s, FONT, scale, th)
    return w, h, b


def put(img, s, x, y, scale, color=TEXT, th=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, th, cv2.LINE_AA)


def panel(img, x0, y0, x1, y1, alpha=0.78, border=None):
    """Полупрозрачная тёмная панель (обрезается краями картинки)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    sub = img[y0:y1, x0:x1]
    sub[:] = (sub * (1 - alpha) + np.array(PANEL, np.float32) * alpha).astype(np.uint8)
    if border:
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), border, 1)


def chip(img, s, x, y, scale, color, fg=None):
    """Плашка режима: тонированный фон, рамка и текст цвета color (обычное начертание).
    (x, y) — левый край и базовая линия текста. Возвращает правый край."""
    w, h, _ = size(s, scale)
    x0, y0, x1, y1 = int(x), int(y - h - 7), int(x + w + 18), int(y + 7)
    tint = tuple(int(0.28 * c + 0.72 * p) for c, p in zip(color, PANEL))
    cv2.rectangle(img, (x0, y0), (x1, y1), tint, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
    light = tuple(min(255, int(c * 0.55 + 255 * 0.45)) for c in color)
    put(img, s, x + 9, y, scale, fg or light, 1)
    return x1


def keys_line(img, s, x, y, scale, max_x=None):
    """Строка подсказок «K: что делает  K2: ...»: клавиши (до двоеточия) — акцентом. Группы — через два пробела."""
    sep = size("   ", scale)[0]
    for grp in [g for g in s.split("  ") if g.strip()]:
        grp = grp.strip()
        k, _, d = grp.partition(": ")
        parts = [(k + ":", ACCENT), (" " + d, TEXT)] if d else [(grp, TEXT)]
        wgrp = sum(size(t, scale)[0] for t, _c in parts)
        if max_x and x + wgrp > max_x:
            break
        for t, c in parts:
            put(img, t, x, y, scale, c)
            x += size(t, scale)[0]
        x += sep
    return x


def toast(img, msg, cx, y, scale, kind='ok', max_w=None):
    """Сообщение по центру: тёмная плашка с цветной полосой слева (зелёная — готово, красная — ошибка)."""
    if not msg:
        return
    col = ERR if kind == 'err' else (ACCENT if kind == 'info' else OK)
    max_w = max_w or img.shape[1] - 40
    while size(msg, scale)[0] > max_w - 40 and len(msg) > 10:
        msg = msg[:-2]
    w, h, _ = size(msg, scale)
    x0 = int(cx - w / 2 - 18)
    panel(img, x0, y - h - 12, x0 + w + 36, y + 12, 0.88)
    cv2.rectangle(img, (x0, int(y - h - 12)), (x0 + 4, int(y + 11)), col, -1)
    put(img, msg, x0 + 20, y, scale, TEXT)


def facts(img, items, x, y, scale, color=TEXT):
    """Факты через точку-разделитель: «30 frames · 1.40 um/px · ...». Возвращает правый край."""
    dot = "  |  "
    for k, t in enumerate(items):
        if k:
            put(img, dot, x, y, scale, LINE)
            x += size(dot, scale)[0]
        put(img, t, x, y, scale, color)
        x += size(t, scale)[0]
    return x


def chevron(img, cx, cy, r, direction, color, th=2):
    """Стрелка-шеврон ‹ или › (direction = -1 / +1), сглаженная, по центру (cx, cy)."""
    d = int(direction)
    pts = np.array([[cx - d * r * 0.45, cy - r], [cx + d * r * 0.55, cy], [cx - d * r * 0.45, cy + r]], np.int32)
    cv2.polylines(img, [pts], False, color, th, cv2.LINE_AA)


def round_rect(img, x0, y0, x1, y1, r, color, th=-1):
    """Прямоугольник со скруглёнными углами (заливка th=-1 или контур)."""
    x0, y0, x1, y1, r = int(x0), int(y0), int(x1), int(y1), int(max(1, r))
    if th < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        for a, b in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)),
                     ((x0, y0 + r), (x0, y1 - r)), ((x1, y0 + r), (x1, y1 - r))):
            cv2.line(img, a, b, color, th, cv2.LINE_AA)
        for (cx, cy), a0 in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270), ((x0 + r, y1 - r), 90), ((x1 - r, y1 - r), 0)):
            cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a0 + 90, color, th, cv2.LINE_AA)


def plus(img, x, y, color, r=10, gap=4, th=1, halo=True):
    """Отметка креста «+» с точностью до доли пикселя: (x, y) — точка в координатах окна, где пиксель u
    занимает [u, u + 1). В центре — просвет gap, чтобы было видно, что лежит ровно под отметкой."""
    sh = 4                                                   # 1/16 пикселя
    cx, cy = (x - 0.5) * 16, (y - 0.5) * 16
    segs = [((cx - r * 16, cy), (cx - gap * 16, cy)), ((cx + gap * 16, cy), (cx + r * 16, cy)),
            ((cx, cy - r * 16), (cx, cy - gap * 16)), ((cx, cy + gap * 16), (cx, cy + r * 16))]
    for col, t in (((0, 0, 0), th + 1), (color, th)) if halo else ((color, th),):
        for a, b in segs:
            cv2.line(img, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), col, t,
                     cv2.LINE_AA, sh)


def ruler(img, a, b, label, color=(0, 220, 255), scale=0.5):
    """Линейка: отметки «+» в точках a и b (координаты окна, b может быть None), линия между ними и подпись
    с расстоянием у середины, на тёмной плашке."""
    if a is None:
        return
    if b is not None:
        p, q = (int(round(a[0] - 0.5)), int(round(a[1] - 0.5))), (int(round(b[0] - 0.5)), int(round(b[1] - 0.5)))
        cv2.line(img, p, q, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(img, p, q, color, 1, cv2.LINE_AA)
        plus(img, b[0], b[1], color)
    plus(img, a[0], a[1], color)
    if label:
        mx, my = (a if b is None else ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
        x0 = int(min(max(4, mx + 12), img.shape[1] - tw - 14))
        y0 = int(min(max(th + 10, my - 12), img.shape[0] - 6))
        panel(img, x0 - 6, y0 - th - 7, x0 + tw + 6, y0 + 6, 0.85, color)
        cv2.putText(img, label, (x0, y0), FONT, scale, (245, 245, 245), 1, cv2.LINE_AA)


def progress_panel(img, title, stage, frac, text="", passed=0.0, left=None, footer=""):
    """Полоса прогресса посреди окна (долгая работа прямо в окне: сшивка, скан): заголовок, этап, полоса
    в процентах 0.00 … 100.00 %, что делается сейчас, сколько прошло и осталось."""
    H, W = img.shape[:2]
    pw, ph = min(700, W - 60), 206
    x0, y0 = int((W - pw) / 2), int((H - ph) / 2)
    panel(img, x0, y0, x0 + pw, y0 + ph, 0.94, LINE)
    put(img, title, x0 + 22, y0 + 38, 0.7, ACCENT, 2)
    pct = f"{100 * frac:.2f}% / 100.00%"
    st_ = stage
    while st_ and size(st_, 0.55)[0] > pw - 60 - size(pct, 0.5)[0]:
        st_ = st_[:-2]
    put(img, st_, x0 + 22, y0 + 74, 0.55, TEXT, 1)
    put(img, pct, x0 + pw - 22 - size(pct, 0.5)[0], y0 + 74, 0.5, TEXT, 1)
    bx0, bx1, by = x0 + 22, x0 + pw - 22, y0 + 92
    cv2.rectangle(img, (bx0, by), (bx1, by + 18), (60, 60, 66), -1)
    cv2.rectangle(img, (bx0, by), (bx0 + int((bx1 - bx0) * max(0.0, min(1.0, frac))), by + 18), (90, 200, 90), -1)
    cv2.rectangle(img, (bx0, by), (bx1, by + 18), (110, 110, 118), 1)
    t = text or ""
    while t and size(t, 0.45)[0] > pw - 44:
        t = t[:-2]
    put(img, t, x0 + 22, y0 + 136, 0.45, DIM, 1)
    tm = lambda s: f"{int(s // 60):02d}:{int(s % 60):02d}"
    put(img, f"{tm(passed)} passed" + (f", this step {tm(left)} left" if left else "") + (f"   {footer}" if footer else ""),
        x0 + 22, y0 + 176, 0.45, DIM, 1)


MENU_H = 30


def menu_bar(W, title, items, hover=None, active=None):
    """Строка меню сверху окна: слева — открытый проект, дальше пункты (New, Open, ...). hover — x мыши
    в строке (подсветка). Возвращает (картинка MENU_H x W, [(x0, x1, пункт)])."""
    img = np.full((MENU_H, W, 3), (34, 34, 38), np.uint8)
    cv2.line(img, (0, MENU_H - 1), (W, MENU_H - 1), (62, 62, 70), 1)
    x = 10
    hits = []
    for it in items:
        tw = size(it, 0.5)[0]
        x0, x1 = x - 8, x + tw + 8
        if (hover is not None and x0 <= hover <= x1) or it == active:
            cv2.rectangle(img, (x0, 3), (x1, MENU_H - 4), (66, 66, 74), -1)
        put(img, it, x, MENU_H - 10, 0.5, TEXT, 1)
        hits.append((x0, x1, it))
        x = x1 + 12
    t = title
    while t and size(t, 0.5)[0] > W - x - 20:
        t = t[:-2]
    if t:
        put(img, t, W - 12 - size(t, 0.5)[0], MENU_H - 10, 0.5, ACCENT, 1)
    return img, hits
