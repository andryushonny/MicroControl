"""Хранилище кадров скана, рассчитанное на тысячи снимков.

scans/<скан>/
  meta.json          — объектив, масштаб кадра, настройки
  index.csv          — по строке на кадр: позиция, Z, резкость, где лежит (batch, slot)
  frames/00001_r001_c001.jpg — кадры в JPG (по умолчанию: компактно, открываются чем угодно сразу)
  batches/b0000.h5   — или пачки по batch_size кадров (HDF5, без потерь; m.frame_format = 'h5')

Почему пачки: 3000 кадров — это ~30 файлов, а не 3000; сбой при записи портит
максимум одну пачку. Кадры внутри разбиты на блоки 4×4, поэтому центр кадра или
любой прямоугольник читается без распаковки всего кадра.
"""
import csv, json, math, os
from collections import OrderedDict
import numpy as np
from . import log as L
from .errors import StoreError

COLS = ['id', 'batch', 'slot', 'row', 'col', 'x_um', 'y_um', 'z_um',
        'x_p', 'y_p', 'z_p', 'sharpness', 'refocused', 'blurry', 't_s', 'file']
NUM = {'id', 'slot', 'x_p', 'y_p', 'z_p', 'refocused', 'blurry'}
THUMB = 16                       # превью кадра: 1/16 по стороне (для обзора карты в окне)
FLT = {'x_um', 'y_um', 'z_um', 'sharpness', 't_s'}


def meta_A(meta, W=None, H=None):
    """Матрица A (px на импульс) карты: из meta['A']; для старых карт — из калибровки или оценки."""
    if meta.get('A'):
        return np.array(meta['A'], float)
    if meta.get('calibration'):
        return np.array(meta['calibration']['A'], float)
    upp = meta['um_per_pulse']['x']
    k = upp / (meta.get('um_per_px') or (meta['fov_um']['x'] / (W or 2592)))
    return -np.diag([k, k])


def _thumb(img):
    import cv2
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, w // THUMB), max(1, h // THUMB)), interpolation=cv2.INTER_AREA).reshape(
        (max(1, h // THUMB), max(1, w // THUMB)) + img.shape[2:])


class ScanData:
    """Кадры скана: st = mc.ScanData(папка); len(st); st.frame(i); st.frame(i, crop=0.5) — центр;
    st.query(center=0.3) — кадры центральной части; st.export() — в PNG."""

    def __init__(self, folder, _mode='r'):
        import h5py
        self.h5py, self.folder, self.mode = h5py, folder, _mode
        self._open = OrderedDict()
        import threading
        self.lock = threading.RLock()                       # h5py и кэш файлов — по одному потоку
        self.fix, self._fixed = None, OrderedDict()
        self.cache_n = 12                                   # сколько раскрытых кадров держать (stitch увеличивает)
        if _mode == 'w':
            return
        try:
            with open(os.path.join(folder, 'meta.json'), encoding='utf-8') as f:
                self.meta = json.load(f)
            with open(os.path.join(folder, 'index.csv'), encoding='utf-8') as f:
                self.index = [self._parse(r) for r in csv.DictReader(f)]
        except OSError as e:
            raise StoreError('E801', f"{folder}") from e
        if not self.index:
            raise StoreError('E801', f"{folder}: no tiles")

        self.x = np.array([r['x_um'] for r in self.index])
        self.y = np.array([r['y_um'] for r in self.index])
        self._cache = OrderedDict()                         # последние раскрытые JPG
        r0 = self.index[0]
        self.shape = self._jpg(r0['file']).shape if r0.get('file') else \
            tuple(self._file(r0['batch'])['tiles'].shape[1:])
        self.fix, self._fixed = None, OrderedDict()         # оптические поправки (optics.attach)
        self.gains = None
        try:
            from .optics import attach
            attach(self, folder)
        except Exception:
            self.fix = None

    def _jpg(self, name):
        img = self._cache.get(name)
        if img is None:
            import cv2
            img = cv2.imdecode(np.fromfile(os.path.join(self.folder, 'frames', name), np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise StoreError('E801', f"cannot read frames/{name}")
            self._cache[name] = img
            if len(self._cache) > self.cache_n:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(name)
        return img

    @staticmethod
    def _parse(r):
        out = {}
        for k, v in r.items():
            if v in ('', None):
                out[k] = 0 if k in NUM else ''
            else:
                out[k] = int(v) if k in NUM else float(v) if k in FLT else (int(v) if k in ('row', 'col') else v)
        return out

    # ---------- запись ----------
    @classmethod
    def create(cls, folder, meta, batch_size=100, compression='gzip', fmt='h5', quality=90):
        """Новая карта для записи. fmt='jpg' — кадры файлами frames/*.jpg (качество quality), 'h5' — пачки HDF5."""
        st = cls(folder, 'w')
        os.makedirs(os.path.join(folder, 'frames' if fmt == 'jpg' else 'batches'), exist_ok=True)
        st.meta, st.batch_size, st.comp = meta, int(batch_size), compression
        st.fmt, st.quality = fmt, int(quality)
        st.n, st._h5, st._csv_f = 0, None, open(os.path.join(folder, 'index.csv'), 'w', newline='', encoding='utf-8')
        st._csv = csv.writer(st._csv_f)
        st._csv.writerow(COLS)
        st.save_meta()
        return st

    def save_meta(self):
        with open(os.path.join(self.folder, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=1, default=str)

    def add(self, img, **row):
        """Дописать кадр. Возвращает его id."""
        if self.fmt == 'jpg':
            import cv2
            r_, c_ = row.get('row', ''), row.get('col', '')
            name = f"{self.n + 1:05d}" + (f"_r{int(r_) + 1:03d}_c{int(c_) + 1:03d}" if r_ != '' else '') + ".jpg"
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            buf.tofile(os.path.join(self.folder, 'frames', name))
            r = dict(id=self.n, batch='', slot=0, file=name, **row)
            self._csv.writerow([r.get(c, '') for c in COLS])
            self._csv_f.flush()
            self.n += 1
            return r['id']
        b, s = divmod(self.n, self.batch_size)
        name = f"b{b:04d}.h5"
        if s == 0:
            self._close_batch()
            self._h5 = self.h5py.File(os.path.join(self.folder, 'batches', name), 'w')
            ch = (1, math.ceil(img.shape[0] / 4), math.ceil(img.shape[1] / 4)) + img.shape[2:]
            self._ds = self._h5.create_dataset('tiles', shape=(0,) + img.shape, maxshape=(self.batch_size,) + img.shape,
                                               dtype=img.dtype, chunks=ch, compression=self.comp)
            th = _thumb(img)
            self._th = self._h5.create_dataset('thumbs', shape=(0,) + th.shape, maxshape=(self.batch_size,) + th.shape,
                                               dtype=img.dtype, chunks=(1,) + th.shape)
        self._ds.resize(s + 1, axis=0)
        self._ds[s] = img
        self._th.resize(s + 1, axis=0)
        self._th[s] = _thumb(img)
        self._h5.flush()
        r = dict(id=self.n, batch=name, slot=s, **row)
        self._csv.writerow([r.get(c, '') for c in COLS])
        self._csv_f.flush()
        self.n += 1
        return r['id']

    def _close_batch(self):
        if self._h5 is not None:
            n = self._ds.shape[0]
            self._h5.close()
            L.event('STO-001', f"batch closed: {n} tiles, total {self.n}")
            self._h5 = None

    # ---------- чтение ----------
    def _file(self, name):
        f = self._open.get(name)
        if f is None:
            f = self.h5py.File(os.path.join(self.folder, 'batches', name), 'r')
            self._open[name] = f
            if len(self._open) > 32:                       # держим открытыми ≤32 пачки
                self._open.popitem(last=False)[1].close()
        else:
            self._open.move_to_end(name)
        return f

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return self.frame(i)

    def info(self):
        """Сводка карты (JSON)."""
        from .util import Result
        mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(self.folder) for f in fs) / 2 ** 20
        blurry = sum(r.get('blurry', 0) or 0 for r in self.index)
        return Result(f"Scan {os.path.basename(self.folder)}", Frames=len(self), Blurry=blurry,
                      **{"Frame, px": f"{self.shape[1]} x {self.shape[0]}",
                         "X, um": (round(self.x.min()), round(self.x.max())),
                         "Y, um": (round(self.y.min()), round(self.y.max())),
                         "Objective": f"{self.meta.get('objective')}x", "Status": self.meta.get('status'),
                         "On disk, MB": round(mb, 1)})

    def __repr__(self):
        return repr(self.info())

    def _repr_pretty_(self, p, cycle):
        p.text(repr(self))

    def _box(self, crop):
        H, W = self.shape[:2]
        if crop is None:
            return 0, 0, W, H
        if isinstance(crop, (int, float)):                 # доля от центра
            cw, chh = int(W * crop), int(H * crop)
            x0, y0 = (W - cw) // 2, (H - chh) // 2
            return x0, y0, x0 + cw, y0 + chh
        return tuple(int(v) for v in crop)                 # (x0, y0, x1, y1) px

    def tile(self, *a, **k):
        raise AttributeError("'tile' is now called 'frame': st.frame(i)")

    def tiles(self, *a, **k):
        raise AttributeError("'tiles' is now called 'frames': st.frames()")

    def frame(self, i, crop=None):
        """Кадр i. crop=0.5 — только центральная половина; crop=(x0,y0,x1,y1) — прямоугольник, px."""
        r = self.index[i]
        x0, y0, x1, y1 = self._box(crop)
        if self.fix is not None:                            # поправленный кадр целиком (кэш), затем вырез
            with self.lock:
                img = self._fixed.get(i)
                if img is None:
                    raw = self._jpg(r['file']) if r.get('file') else self._file(r['batch'])['tiles'][r['slot']]
                    img = self.fix(raw)
                    g_ = getattr(self, 'gains', None)
                    if g_ is not None:                      # яркость кадра, выровненная по соседям
                        img = np.clip(img.astype(np.float32) * (g_[i] if img.ndim == 3 else g_[i].mean()),
                                      0, 255).astype(np.uint8)
                    self._fixed[i] = img
                    if len(self._fixed) > self.cache_n:
                        self._fixed.popitem(last=False)
                else:
                    self._fixed.move_to_end(i)
            return img[y0:y1, x0:x1]
        with self.lock:
            if r.get('file'):
                return self._jpg(r['file'])[y0:y1, x0:x1]
            return self._file(r['batch'])['tiles'][r['slot'], y0:y1, x0:x1]

    def frames(self, ids=None, crop=None):
        """Перебор (id, кадр) пачками — каждая пачка открывается один раз."""
        ids = range(len(self)) if ids is None else ids
        for i in sorted(ids, key=lambda k: (self.index[k]['batch'], self.index[k]['slot'])):
            yield i, self.frame(i, crop)

    def thumbs(self):
        """Все превью (1/16) списком — читаются из пачек; для старых карт строятся и кэшируются."""
        out = [None] * len(self)
        cache = os.path.join(self.folder, 'viewer', 'thumbs.npz')
        for i, r in enumerate(self.index):
            if r.get('file'):
                break
            with self.lock:
                f = self._file(r['batch'])
                if 'thumbs' not in f:
                    break
                out[i] = f['thumbs'][r['slot']]
        else:
            return out
        if os.path.exists(cache):
            with np.load(cache) as z:                       # файл не держится открытым (Windows: его перезапишет сшивка)
                return [z[f"t{i}"] for i in range(len(self))]
        for i, img in self.frames():                        # по одному кадру: память не растёт
            out[i] = _thumb(img)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **{f"t{i}": a for i, a in enumerate(out)})
        return out

    def footprint(self):
        """Размер кадра на образце, мкм (x, y)."""
        f = self.meta.get('fov_um') or {}
        return f.get('x', 0), f.get('y', 0)

    def query(self, x=None, y=None, center=None, rows=None, cols=None):
        """Выбрать кадры (id): x=(x0,x1), y=(y0,y1) мкм — задевающие прямоугольник;
        center=0.3 — центральные 30% карты; rows/cols=(a,b) — диапазон строк/столбцов."""
        fw, fh = self.footprint()
        m = np.ones(len(self), bool)
        if center is not None:
            cx, cy = (self.x.min() + self.x.max()) / 2, (self.y.min() + self.y.max()) / 2
            hw = (self.x.max() - self.x.min() + fw) * center / 2
            hh = (self.y.max() - self.y.min() + fh) * center / 2
            x, y = (cx - hw, cx + hw), (cy - hh, cy + hh)
        if x is not None:
            m &= (self.x + fw / 2 >= x[0]) & (self.x - fw / 2 <= x[1])
        if y is not None:
            m &= (self.y + fh / 2 >= y[0]) & (self.y - fh / 2 <= y[1])
        for key, rng in (('row', rows), ('col', cols)):
            if rng is not None:
                v = np.array([r[key] if isinstance(r[key], int) else -1 for r in self.index])
                m &= (v >= rng[0]) & (v <= rng[1])
        return np.flatnonzero(m)

    def export(self, out=None, ids=None, crop=None, fmt='png'):
        """Выгрузить кадры в обычные картинки (+ TileConfiguration.txt для Fiji)."""
        from .core import imwrite
        out = out or os.path.join(self.folder, 'export')
        os.makedirs(out, exist_ok=True)
        lines = ["dim = 2"]
        A = meta_A(self.meta, self.shape[1], self.shape[0])
        from .util import Progress
        pr = Progress(len(self) if ids is None else len(ids), "Export frames", unit='frames')
        for i, img in self.frames(ids, crop):
            r = self.index[i]
            fn = f"t{i:05d}." + fmt
            imwrite(os.path.join(out, fn), img)
            pr.update(1, fn)
            if A is not None:                              # позиция в px по масштабу кадра
                px, py = -(A @ (r['x_p'], r['y_p']))
                lines.append(f"{fn}; ; ({px:.2f}, {py:.2f})")
        pr.close()
        if A is not None:
            with open(os.path.join(out, 'TileConfiguration.txt'), 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + "\n")
        return out

    def close(self):
        if self.mode == 'w':
            self._close_batch()
            self._csv_f.close()
        for f in self._open.values():
            f.close()
        self._open.clear()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
