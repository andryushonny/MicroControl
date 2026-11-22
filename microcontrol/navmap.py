"""Карта образца (навигатор): всё, что снято, — сшитые мозаики на своих местах в настоящих мкм столика
(10x под 50x под 100x), контуры выделений и рамка «где сейчас камера». Щелчок по карте — столик едет туда,
тянуть мышью — новый участок, колесо — зум, правая / средняя кнопка — сдвиг карты.

Мозаика -> столик строго по геометрии съёмки (snap.geometry): S = ms · Mo⁻¹ (q − c − кадр/2).
"""
import os
import numpy as np
import cv2

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class NavMap:
    def __init__(self):
        self.c = None                     # центр вида, настоящие мкм
        self.s = None                     # мкм на пиксель карты
        self.size = (0, 0)
        self._base, self._key = None, None
        self._prev = {}                   # папка -> (mtime, пирамида превью [(k, img)], геометрия)

    # ------------------------------------------------------------ координаты
    def to_px(self, S):
        w, h = self.size
        return (np.asarray(S, float) - self.c) / self.s + (w / 2, h / 2)

    def to_um(self, p):
        w, h = self.size
        return (np.asarray(p, float) - (w / 2, h / 2)) * self.s + self.c

    def fit(self, boxes, w, h, pad=0.12):
        """Вид — чтобы поместились прямоугольники [(x0, y0, x1, y1)] мкм."""
        if not boxes:
            return
        b = np.array(boxes, float)
        x0, y0 = b[:, 0].min(), b[:, 1].min()
        x1, y1 = b[:, 2].max(), b[:, 3].max()
        self.c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        self.s = max((x1 - x0) / max(1, w), (y1 - y0) / max(1, h), 1e-3) * (1 + 2 * pad)

    def zoom(self, p, k):
        """Зум к точке p (пиксели карты): k > 1 — ближе."""
        S = self.to_um(p)
        self.s = float(np.clip(self.s / k, 0.05, 500.0))
        self.c = S - (np.asarray(p, float) - (self.size[0] / 2, self.size[1] / 2)) * self.s

    # ------------------------------------------------------------ мозаики
    def _preview(self, folder):
        jp = os.path.join(folder, 'join', 'preview_all.jpg')
        if not os.path.exists(jp):
            return None
        mt = os.path.getmtime(jp)
        if folder in self._prev and self._prev[folder][0] == mt:
            return self._prev[folder][1:]
        from .snap import geometry
        g = geometry(folder)
        img = cv2.imread(jp)
        if img is None:
            return None
        size = g['pos'].max(0) + (g['W'], g['H'])
        k = min(1.0, 4000 / size.max())                    # как в stitch.make_preview
        ok = (img.max(2) > 6).astype(np.uint8) * 255        # где кадры (чёрное вокруг — прозрачно)
        ok = cv2.erode(ok, np.ones((3, 3), np.uint8))
        pyr = [(k, img, ok)]
        while max(pyr[-1][1].shape[:2]) > 300:
            pyr.append((pyr[-1][0] / 2, cv2.pyrDown(pyr[-1][1]), cv2.resize(pyr[-1][2], None, fx=0.5, fy=0.5,
                                                                               interpolation=cv2.INTER_NEAREST)))
        self._prev[folder] = (mt, pyr, g)
        return pyr, g

    def _warp(self, canvas, folder):
        got = self._preview(folder)
        if got is None:
            return
        pyr, g = got
        Mi = np.linalg.inv(g['Mo']) * g['ms']                # пиксели мозаики -> мкм столика
        off = Mi @ (g['c'] + (g['W'] / 2, g['H'] / 2))
        k, img, ok = pyr[0]
        for k_, im_, ok_ in pyr:                            # уровень пирамиды ~ масштаб карты
            if abs(np.linalg.det(Mi)) ** 0.5 / k_ / self.s <= 1.6:
                k, img, ok = k_, im_, ok_
        w, h = self.size
        A = Mi / k / self.s                                 # пиксели превью -> пиксели карты
        t = (-off - self.c) / self.s + (w / 2, h / 2)
        M = np.c_[A, t]
        pic = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        msk = cv2.warpAffine(ok, M, (w, h), flags=cv2.INTER_NEAREST) > 0
        canvas[msk] = pic[msk]

    def base(self, scans, w, h):
        """Подложка карты: мозаики scans (папки, сначала меньшее увеличение). Кэш, пока вид не менялся."""
        self.size = (w, h)
        key = (w, h, tuple(np.round(self.c, 3)), round(self.s, 6),
               tuple((f, os.path.getmtime(os.path.join(f, 'join', 'preview_all.jpg'))
                      if os.path.exists(os.path.join(f, 'join', 'preview_all.jpg')) else 0) for f in scans))
        if key != self._key:
            canvas = np.full((h, w, 3), 24, np.uint8)
            step = 10 ** np.floor(np.log10(max(1e-3, self.s * 80)))          # сетка мкм — для ориентира
            S0, S1 = self.to_um((0, 0)), self.to_um((w, h))
            for x in np.arange(np.floor(S0[0] / step) * step, S1[0], step):
                X = int(self.to_px((x, 0))[0])
                cv2.line(canvas, (X, 0), (X, h), (34, 34, 38), 1)
            for y in np.arange(np.floor(S0[1] / step) * step, S1[1], step):
                Y = int(self.to_px((0, y))[1])
                cv2.line(canvas, (0, Y), (w, Y), (34, 34, 38), 1)
            for f in scans:
                try:
                    self._warp(canvas, f)
                except Exception:
                    pass
            self._base, self._key = canvas, key
        return self._base.copy()

    # ------------------------------------------------------------ поверх
    def poly(self, img, ring, color, thick=1, fill=None):
        pts = np.int32(np.round(self.to_px(ring) * 16))
        if fill is not None:
            ov = img.copy()
            cv2.fillPoly(ov, [pts], fill, cv2.LINE_AA, 4)
            cv2.addWeighted(ov, 0.25, img, 0.75, 0, dst=img)
        cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA, 4)

    def label(self, img, text, S, color):
        x, y = self.to_px(S)
        cv2.putText(img, text, (int(x) + 4, int(y) + 14), _FONT, 0.42, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(img, text, (int(x) + 4, int(y) + 14), _FONT, 0.42, color, 1, cv2.LINE_AA)

    def scale_bar(self, img):
        w, h = self.size
        L = 10 ** np.floor(np.log10(self.s * 120))
        for k in (5, 2, 1):
            if k * L / self.s <= 160:
                L *= k
                break
        px = int(L / self.s)
        y = h - 14
        cv2.line(img, (12, y), (12 + px, y), (230, 230, 230), 2)
        t = f"{L / 1000:g} mm" if L >= 1000 else f"{L:g} um"
        cv2.putText(img, t, (18 + px, y + 5), _FONT, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
