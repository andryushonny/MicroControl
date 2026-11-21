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
