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
        self.turn = 0                     # карта повёрнута на 0 / 90 / 180 / 270° против часовой (m.map_turn)

    # ------------------------------------------------------------ координаты
    @property
    def R(self):
        """Поворот карты: вектор столика -> вектор на экране (y вниз; +90° — против часовой)."""
        t = np.radians(self.turn)
        return np.round(np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]), 12)

    def to_px(self, S):
        w, h = self.size
        return ((np.asarray(S, float) - self.c) / self.s) @ self.R.T + (w / 2, h / 2)

    def to_um(self, p):
        w, h = self.size
        return ((np.asarray(p, float) - (w / 2, h / 2)) * self.s) @ self.R + self.c

    def pan(self, c0, dp):
        """Сдвиг карты мышью на dp пикселей от центра c0 (мкм)."""
        self.c = np.asarray(c0, float) - self.R.T @ np.asarray(dp, float) * self.s

    def fit(self, boxes, w, h, pad=0.12):
        """Вид — чтобы поместились прямоугольники [(x0, y0, x1, y1)] мкм."""
        if not boxes:
            return
        b = np.array(boxes, float)
        x0, y0 = b[:, 0].min(), b[:, 1].min()
        x1, y1 = b[:, 2].max(), b[:, 3].max()
        self.c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        if self.turn % 180:                                 # повёрнута на 90°: X столика — по вертикали
            w, h = h, w
        self.s = max((x1 - x0) / max(1, w), (y1 - y0) / max(1, h), 1e-3) * (1 + 2 * pad)

    def zoom(self, p, k):
        """Зум к точке p (пиксели карты): k > 1 — ближе."""
        S = self.to_um(p)
        self.s = float(np.clip(self.s / k, 0.05, 500.0))
        self.c = S - self.R.T @ (np.asarray(p, float) - (self.size[0] / 2, self.size[1] / 2)) * self.s

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

    def _warp(self, canvas, folder, clip=None):
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
        A = self.R @ Mi / k / self.s                        # пиксели превью -> пиксели карты
        t = self.R @ ((-off - self.c) / self.s) + (w / 2, h / 2)
        M = np.c_[A, t]
        pic = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        msk = cv2.warpAffine(ok, M, (w, h), flags=cv2.INTER_NEAREST) > 0
        if clip:                                            # только внутри контура выделения: лишнее (кадры за краем,
            cm = np.zeros((h, w), np.uint8)                 # до шага мотора) не путает
            cv2.fillPoly(cm, [np.int32(np.round(self.to_px(np.asarray(r_, float)))) for r_ in clip], 1, cv2.LINE_8)
            msk &= cm > 0
        canvas[msk] = pic[msk]

    def base(self, scans, w, h, clip=None):
        """Подложка карты: мозаики scans (папки, сначала меньшее увеличение), каждая — по контуру своего
        выделения (clip: папка -> кольца, мкм). Кэш, пока вид не менялся."""
        self.size = (w, h)
        clip = clip or {}
        ck_ = tuple((f, tuple(round(float(v), 2) for r_ in clip.get(f, []) for p_ in r_ for v in p_)) for f in scans)
        key = (w, h, tuple(np.round(self.c, 3)), round(self.s, 6), self.turn, ck_,
               tuple((f, os.path.getmtime(os.path.join(f, 'join', 'preview_all.jpg'))
                      if os.path.exists(os.path.join(f, 'join', 'preview_all.jpg')) else 0) for f in scans))
        if key != self._key:
            canvas = np.full((h, w, 3), 24, np.uint8)
            step = 10 ** np.floor(np.log10(max(1e-3, self.s * 80)))          # сетка мкм — для ориентира
            cs_ = self.to_um(np.array([(0, 0), (w, 0), (0, h), (w, h)], float))
            S0, S1 = cs_.min(0), cs_.max(0)
            for x in np.arange(np.floor(S0[0] / step) * step, S1[0], step):
                a_, b_ = self.to_px(np.array([(x, S0[1]), (x, S1[1])]))
                cv2.line(canvas, tuple(np.int32(a_)), tuple(np.int32(b_)), (34, 34, 38), 1)
            for y in np.arange(np.floor(S0[1] / step) * step, S1[1], step):
                a_, b_ = self.to_px(np.array([(S0[0], y), (S1[0], y)]))
                cv2.line(canvas, tuple(np.int32(a_)), tuple(np.int32(b_)), (34, 34, 38), 1)
            for f in scans:
                try:
                    self._warp(canvas, f, clip.get(f))
                except Exception:
                    pass
            self._base, self._key = canvas, key
        return self._base.copy()

    # ------------------------------------------------------------ поверх
    def poly(self, img, ring, color, thick=1, fill=None, dashed=False):
        pf = self.to_px(ring)
        pts = np.int32(np.round(pf))
        if fill is not None:
            ov = img.copy()
            cv2.fillPoly(ov, [pts], fill, cv2.LINE_8)
            cv2.addWeighted(ov, 0.25 if not dashed else 0.12, img, 0.75 if not dashed else 0.88, 0, dst=img)
        if dashed:
            from .ui import dashed_poly
            dashed_poly(img, pf, color, thick)
        else:
            cv2.polylines(img, [pts], True, color, thick, cv2.LINE_8)

    def labels(self, img, items, reserved=()):
        """Подписи выделений плашками, не наезжая друг на друга и на чужие выделения.
        items: [(текст, рамка на карте (x0, y0, x1, y1), цвет, [рамки, на которые не залезать])] — по важности."""
        from . import ui as UI
        H, W = img.shape[:2]
        placed = [(0, 0, W, 26), (0, H - 26, 220, H), *reserved]   # подсказка сверху, масштаб внизу, камера в углу
        inter = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        for text, (x0, y0, x1, y1), color, avoid in items:
            tw, th, _b = UI.size(text, 0.42)
            bw, bh = tw + 10, th + 8
            cands = [(x0, y0 - bh - 2), (x1 - bw, y0 - bh - 2), (x0 + 2, y0 + 2), (x0, y1 + 2), (x1 - bw, y1 + 2),
                     (x1 + 2, y0), (x0 - bw - 2, y0)]
            best = None
            for cx, cy in cands:
                r = (cx, cy, cx + bw, cy + bh)
                if r[0] < 0 or r[1] < 0 or r[2] > W or r[3] > H:
                    continue
                cost = sum(inter(r, q) for q in placed) * 4 + sum(inter(r, q) for q in avoid)
                if best is None or cost < best[0]:
                    best = (cost, r)
                if cost == 0:
                    break
            if best is None:
                continue
            r = tuple(int(v) for v in best[1])
            sub = img[r[1]:r[3], r[0]:r[2]]
            sub[:] = (sub * 0.25 + np.array([24, 24, 28], np.float32) * 0.75).astype(np.uint8)
            cv2.rectangle(img, (r[0], r[1]), (r[2] - 1, r[3] - 1), color, 1)
            UI.put(img, text, r[0] + 5, r[3] - 5, 0.42, color, 1)
            placed.append(r)

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
