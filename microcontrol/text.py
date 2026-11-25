"""Шрифт окон: JetBrains Mono (OFL, microcontrol/fonts) вместо штрихового шрифта OpenCV.

Модули окон пишут `from .text import cv2`: это тот же cv2, только putText и getTextSize рисуют и меряют
JetBrains Mono (Regular; толщина >= 2 — Bold), с теми же аргументами и тем же масштабом, что у
FONT_HERSHEY_SIMPLEX (высота заглавной буквы та же). Код пользователя с обычным cv2 это не трогает.
Нет Pillow или файла шрифта — всё как раньше (штриховой шрифт OpenCV).

Надписи кэшируются (текст, размер, жирность -> маска): одна и та же надпись в каждом кадре не рисуется заново.
"""
import os
import types
from collections import OrderedDict
import numpy as np
import cv2 as _cv2

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
_FILES = {False: 'JetBrainsMono-Regular.ttf', True: 'JetBrainsMono-Bold.ttf'}
_FONTS = {}
_MASK = OrderedDict()
_CAP = None                     # высота заглавной OpenCV при масштабе 1 (px)
_PER = 1.0                      # кегль JetBrains Mono на единицу масштаба OpenCV
try:
    from PIL import Image, ImageDraw, ImageFont
    _OK = all(os.path.exists(os.path.join(_DIR, f)) for f in _FILES.values())
except ImportError:
    _OK = False


def _font(px, bold):
    key = (px, bold)
    f = _FONTS.get(key)
    if f is None:
        f = _FONTS[key] = ImageFont.truetype(os.path.join(_DIR, _FILES[bold]), px)
    return f


def _calib():
    """Кегль под масштаб OpenCV: заглавная H — той же высоты."""
    global _CAP, _PER
    if _CAP is None:
        _CAP = _cv2.getTextSize("H", _cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)[0][1]
        f = ImageFont.truetype(os.path.join(_DIR, _FILES[False]), 100)
        b = f.getbbox("H", anchor='ls')
        _PER = 0.82 * _CAP / (-b[1] / 100.0)            # моноширинный шире: чуть мельче, чтобы влезал
    return _PER


def _px(scale):
    return max(6, int(round(_calib() * float(scale))))


_MET, _LEN = {}, OrderedDict()


def _metrics(px, bold):
    m = _MET.get((px, bold))
    if m is None:
        f = _font(px, bold)
        m = _MET[(px, bold)] = (f, -f.getbbox("H", anchor='ls')[1], f.getbbox("gjpqy", anchor='ls')[3])
    return m


def getTextSize(text, fontFace, fontScale, thickness=1):
    """Как cv2.getTextSize: ((ширина, высота над базовой линией), под линией). Размеры — из кэша."""
    if not _OK:
        return _cv2.getTextSize(text, fontFace, fontScale, thickness)
    px, bold = _px(fontScale), thickness >= 2
    key = (str(text), px, bold)
    r = _LEN.get(key)
    if r is None:
        f, asc, desc = _metrics(px, bold)
        r = _LEN[key] = ((int(round(f.getlength(key[0]))), int(asc)), int(desc))
        if len(_LEN) > 20000:
            _LEN.popitem(last=False)
    return r


def _mask(text, px, bold):
    key = (text, px, bold)
    m = _MASK.get(key)
    if m is not None:
        _MASK.move_to_end(key)
        return m
    f, asc, desc = _metrics(px, bold)
    x0, y0, x1, y1 = f.getbbox(text, anchor='ls')
    w, h = max(1, x1 - min(0, x0) + 2), max(1, y1 - min(y0, -asc) + 2)
    im = Image.new('L', (w, h), 0)
    oy = -min(y0, -asc) + 1
    ImageDraw.Draw(im).text((-min(0, x0) + 1, oy), text, font=f, fill=255, anchor='ls')
    m = (np.asarray(im, np.uint8), -min(0, x0) + 1, oy)       # маска; где в ней начало строки (x, базовая линия)
    _MASK[key] = m
    if len(_MASK) > 6000:
        _MASK.popitem(last=False)
    return m


def putText(img, text, org, fontFace, fontScale, color, thickness=1, lineType=None, bottomLeftOrigin=False):
    """Как cv2.putText: org — начало базовой линии."""
    if not _OK or bottomLeftOrigin:
        return _cv2.putText(img, text, org, fontFace, fontScale, color, thickness,
                            _cv2.LINE_AA if lineType is None else lineType, bottomLeftOrigin)
    text = str(text)
    if not text:
        return img
    a, ox, oy = _mask(text, _px(fontScale), thickness >= 2)
    x0, y0 = int(org[0]) - ox, int(org[1]) - oy
    H, W = img.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + a.shape[1]), min(H, y0 + a.shape[0])
    if xb <= xa or yb <= ya:
        return img
    al = a[ya - y0:yb - y0, xa - x0:xb - x0]
    roi = img[ya:yb, xa:xb]
    if img.ndim == 2:
        c = float(color[0] if isinstance(color, (tuple, list)) else color)
        roi[:] = (roi * (1 - al / 255.0) + c * (al / 255.0)).astype(img.dtype)
        return img
    c = np.array(color[:img.shape[2]] if len(color) >= img.shape[2] else list(color) + [0] * (img.shape[2] - len(color)),
                 np.float32)
    k = (al.astype(np.float32) / 255.0)[..., None]
    roi[:] = (roi.astype(np.float32) * (1 - k) + c * k).astype(img.dtype)
    return img


class _Cv2(types.ModuleType):
    """cv2 со шрифтом JetBrains Mono в putText / getTextSize; всё остальное — настоящий cv2."""
    def __getattr__(self, name):
        return getattr(_cv2, name)


cv2 = _Cv2('cv2')
cv2.putText = putText
cv2.getTextSize = getTextSize
