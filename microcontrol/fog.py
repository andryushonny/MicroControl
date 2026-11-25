"""«Туман войны» карты: живые кадры камеры, пока столик ездит, ложатся на карту под снятым (грубо, ~2 мкм/px).
Где уже были — видно; запоминается у образца (<образец>/fog/: плитки JPEG и fog.json).

    Fog(папка | None).add(img, corners_um)   — кадр, его 4 угла на столике (manual.camera_box)
    Fog.draw(canvas, nav)                    — на подложку карты (под снимками выделений)
"""
import os
import json
import threading
import time
import numpy as np
import cv2

RES = 2.0                        # мкм на пиксель тумана
TILE = 512                       # плитка, px


class Fog:
    def __init__(self, folder=None, res=RES):
        self.folder = os.path.join(folder, 'fog') if folder else None
        self.res, self.tiles, self.dirty = float(res), {}, set()
        self.ver, self.lock, self.t_save = 0, threading.RLock(), time.monotonic()
        self.last = None                                 # центр последнего кадра, мкм
        self._cache, self._tver = {}, {}                 # уровни плиток для карты; версия каждой плитки
        if self.folder and os.path.isdir(self.folder):
            try:
                d = json.load(open(os.path.join(self.folder, 'fog.json'), encoding='utf-8'))
                if abs(float(d.get('res', res)) - self.res) < 1e-9:
                    for k in d.get('tiles', []):
                        p = os.path.join(self.folder, f"{k[0]}_{k[1]}.jpg")
                        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) if os.path.exists(p) else None
                        if img is not None and img.shape[:2] == (TILE, TILE):
                            self.tiles[(int(k[0]), int(k[1]))] = img
            except (OSError, ValueError):
                pass

    def add(self, img, corners, min_move=0.25):
        """Кадр img (BGR), углы (0,0), (w,0), (w,h), (0,h) на столике, мкм. Кадр почти там же, где прошлый
        (меньше min_move поля зрения), — пропускается. -> True, если лёг."""
        C = np.asarray(corners, float)
        c = C.mean(0)
        fov = max(np.ptp(C[:, 0]), np.ptp(C[:, 1]), 1e-6)
        if self.last is not None and np.hypot(*(c - self.last)) < min_move * fov:
            return False
        h, w = img.shape[:2]
        src = np.float32([(0, 0), (w, 0), (w, h)])
        A = cv2.getAffineTransform(src, np.float32(C[:3]))            # пиксель кадра -> мкм
        k = max(1, int(self.res / (fov / max(w, h))))                 # кадр -> ~res мкм на пиксель
        small = cv2.resize(img, (max(2, w // k), max(2, h // k)), interpolation=cv2.INTER_AREA) if k > 1 else img
        A = np.c_[A[:, :2] @ np.diag([w / small.shape[1], h / small.shape[0]]), A[:, 2]]
        lo, hi = C.min(0), C.max(0)
        t0 = np.floor(lo / (self.res * TILE)).astype(int)
        t1 = np.floor(hi / (self.res * TILE)).astype(int)
        Ai = cv2.invertAffineTransform(np.float32(A))
        inner = np.full(small.shape[:2], 255, np.uint8)               # кадр целиком: туман — без щелей
        with self.lock:
            for tx in range(t0[0], t1[0] + 1):
                for ty in range(t0[1], t1[1] + 1):
                    T0 = np.array([tx, ty], float) * self.res * TILE
                    # пиксель плитки q -> мкм T0 + res q -> пиксель кадра: A⁻¹
                    M = np.c_[Ai[:, :2] * self.res, Ai[:, :2] @ T0 + Ai[:, 2]]
                    pic = cv2.warpAffine(small, np.float32(M), (TILE, TILE), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_REPLICATE)        # край кадра — без тёмной каймы
                    msk = cv2.warpAffine(inner, np.float32(M), (TILE, TILE),
                                         flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)
                    msk = cv2.dilate(msk, np.ones((3, 3), np.uint8)) > 0   # +1 px: стык соседних кадров без щели
                    if not msk.any():
                        continue
                    t = self.tiles.get((tx, ty))
                    if t is None:
                        t = self.tiles[(tx, ty)] = np.zeros((TILE, TILE, 3), np.uint8)
                    t[msk] = np.maximum(pic[msk], 1)              # 0 — «здесь не были»
                    self.dirty.add((tx, ty))
                    self._tver[(tx, ty)] = self._tver.get((tx, ty), 0) + 1
            self.ver += 1
            self.last = c
        if time.monotonic() - self.t_save > 5:
            self.save()
        return True

    def covered(self, x, y):
        """Здесь (мкм столика) туман уже рассеян?"""
        q = np.array([x, y], float) / self.res
        k = (int(np.floor(q[0] / TILE)), int(np.floor(q[1] / TILE)))
        with self.lock:
            t = self.tiles.get(k)
            if t is None:
                return False
            i, j = int(q[1] - k[1] * TILE), int(q[0] - k[0] * TILE)
            return bool(t[min(i, TILE - 1), min(j, TILE - 1)].max() > 0)

    def save(self):
        """Записать изменённые плитки (у образца)."""
        self.t_save = time.monotonic()
        if not self.folder:
            return
        with self.lock:
            dirty, self.dirty = set(self.dirty), set()
            imgs = {k: self.tiles[k].copy() for k in dirty}
            keys = sorted(self.tiles)
        if not imgs:
            return
        try:
            os.makedirs(self.folder, exist_ok=True)
            for (tx, ty), im in imgs.items():
                ok, buf = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 85])
                buf.tofile(os.path.join(self.folder, f"{tx}_{ty}.jpg"))
            with open(os.path.join(self.folder, 'fog.json'), 'w', encoding='utf-8') as f:
                json.dump(dict(res=self.res, tile=TILE, tiles=[list(k) for k in keys]), f)
        except OSError:
            pass

    def clear(self):
        with self.lock:
            self.tiles.clear()
            self.dirty.clear()
            self.ver += 1
            self.last = None
        if self.folder and os.path.isdir(self.folder):
            for f in os.listdir(self.folder):
                try:
                    os.remove(os.path.join(self.folder, f))
                except OSError:
                    pass

    def draw(self, canvas, nav, dim=0.6):
        """Плитки, видные на карте, — на canvas (подложка карты), чуть темнее снятого."""
        h, w = canvas.shape[:2]
        cs = nav.to_um(np.array([(0, 0), (w, 0), (0, h), (w, h)], float))
        lo, hi = cs.min(0), cs.max(0)
        with self.lock:
            items = [(k, t) for k, t in self.tiles.items()
                     if (k[0] + 1) * self.res * TILE >= lo[0] and k[0] * self.res * TILE <= hi[0]
                     and (k[1] + 1) * self.res * TILE >= lo[1] and k[1] * self.res * TILE <= hi[1]]
        for (tx, ty), t in items:
            T0 = np.array([tx, ty], float) * self.res * TILE
            A = nav.R * (self.res / nav.s)                                # пиксель плитки -> пиксель карты
            b = nav.R @ ((T0 - nav.c) / nav.s) + (w / 2, h / 2)
            cr = np.array([(0, 0), (TILE, 0), (0, TILE), (TILE, TILE)], float) @ A.T + b
            x0, y0 = np.maximum(np.floor(cr.min(0)).astype(int), 0)       # только свой кусок окна, не всё окно
            x1, y1 = np.minimum(np.ceil(cr.max(0)).astype(int) + 1, (w, h))
            if x1 <= x0 or y1 <= y0:
                continue
            pyr = self._pyr((tx, ty), t)
            sc = self.res / nav.s                                         # пикселей карты на пиксель плитки
            lv = 0
            while sc * 2 ** lv < 0.5 and lv + 1 < len(pyr):               # далеко: уменьшенная плитка (быстро, без ряби)
                lv += 1
            src, sm = pyr[lv]
            M = np.float32(np.c_[A * 2 ** lv, b - (x0, y0)])
            pic = cv2.warpAffine(src, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            msk = cv2.warpAffine(sm, M, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST, borderValue=0)
            if dim < 1:
                pic = cv2.convertScaleAbs(pic, alpha=dim, beta=24 * (1 - dim))
            np.copyto(canvas[y0:y1, x0:x1], pic, where=msk[..., None] > 0)

    def _pyr(self, k, t):
        """Плитка и её маска «были здесь» по уровням (1, 1/2, 1/4 ...) — пока плитка не менялась."""
        c = self._cache.get(k)
        if c is None or c[0] is not t or c[1] != self._tver.get(k, 0):
            lv = [(t, (t.max(2) > 0).astype(np.uint8))]
            while lv[-1][0].shape[0] >= 16:
                a, m = lv[-1]
                lv.append((cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA),
                           cv2.resize(m, (m.shape[1] // 2, m.shape[0] // 2), interpolation=cv2.INTER_NEAREST)))
            c = self._cache[k] = (t, self._tver.get(k, 0), lv)
        return c[2]
