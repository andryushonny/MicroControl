"""Живая quick-сшивка: кадры сшиваются, пока скан ещё идёт (разделяй и властвуй).

Каждый новый кадр сравнивается только с уже снятыми соседями (по столику) — на уменьшенных кадрах, как у quick;
положения всех кадров решаются заново по всем измеренным швам (доли секунды и на тысячах кадров). Итог после
каждого шага — join/positions_all.csv (просмотр показывает скан по нему сразу) и join/live_pairs.json — швы:
quick-сшивка после скана берёт их готовыми и не считает заново.

    python -m microcontrol.live <папка>      — следить за папкой, пока не появится join/live.stop
"""
import os
import sys
import csv
import json
import time
import numpy as np
import cv2

from .storage import ScanData, meta_A

STOP = 'live.stop'
PAIRS = 'live_pairs.json'
FLAGS = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}


def reduce_of(W):
    """Во сколько раз уменьшать кадры (как у quick: ~300 px и больше по ширине)."""
    return max([k for k in (1, 2, 4, 8) if W // k >= 300] or [1])


def neighbours(pred, W, H, new, armax=0.0):
    """Пары (a, b), a < b, где b — новый кадр, а a — любой уже снятый, перекрывающийся по столику (как в
    stitch._builtin: прямоугольники перекрываются, уголки по диагонали — нет). -> (пары, наибольшее перекрытие)."""
    out = []
    for b in new:
        d = np.abs(pred[:b] - pred[b])
        k = np.flatnonzero((d[:, 0] < W - 16) & (d[:, 1] < H - 16))
        for a in k:
            out.append((int(a), b, float((W - d[a, 0]) * (H - d[a, 1]))))
    armax = max([armax] + [o[2] for o in out])
    return [(a, b) for a, b, ar in out if ar >= 0.3 * armax], armax


class Live:
    def __init__(self, folder):
        self.folder = folder
        self.img = {}                                   # i -> уменьшенный кадр
        self.pairs = {}                                 # (a, b) -> (dx, dy, err) в px уменьшенного кадра
        self.n = 0
        self.r = None
        self.pos = None
        self.armax = 0.0

    def _frame(self, st, i):
        row = st.index[i]
        p = os.path.join(self.folder, 'frames', row['file'])
        a = cv2.imdecode(np.fromfile(p, np.uint8), FLAGS.get(self.r, cv2.IMREAD_COLOR))
        if a is None:
            return None
        if a.shape[1] != self.W or a.shape[0] != self.H:
            a = cv2.resize(a, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return a

    def update(self):
        """Дошить то, что снято с прошлого раза. -> сколько кадров теперь (0 — ещё ни одного)."""
        from .stitch import _pair, _solve_graph
        try:
            st = ScanData(self.folder)
        except Exception:
            return 0
        try:
            if not st.index or not st.index[0].get('file'):
                return 0                                # HDF5: читать на ходу нельзя — сошьётся после скана
            H0, W0 = st.shape[:2]
            if self.r is None:
                self.r = reduce_of(W0)
                self.W, self.H = W0 // self.r, H0 // self.r
            n = len(st.index)
            new = list(range(self.n, n))
            for i in new:
                a = self._frame(st, i)
                if a is None:                           # кадр ещё пишется: в следующий раз
                    n = i
                    break
                self.img[i] = a
            new = [i for i in new if i < n]
            if not new and self.pos is not None:
                return self.n
            p = np.array([[r['x_p'], r['y_p']] for r in st.index[:n]], float)
            A = meta_A(st.meta, W0, H0)
            pred = -(p @ A.T) / self.r
            px = st.meta.get('pixel_size_um') or 1.0
            ms = min(40.0 / (px * self.r), 0.5 * min(self.W, self.H))

            class _R:
                pass
            red = _R()
            red.frame = lambda i, crop=None: self.img[i] if crop is None else \
                self.img[i][crop[1]:crop[3], crop[0]:crop[2]]
            todo, self.armax = neighbours(pred, self.W, self.H, new, self.armax)
            if todo:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) - 1))) as ex:
                    res = list(ex.map(lambda ab: _pair(red, ab[0], ab[1], pred[ab[1]] - pred[ab[0]], self.W, self.H,
                                                       int(ms), 0.0, ms), todo))
                for (a, b), (dt, e) in zip(todo, res):
                    self.pairs[(a, b)] = (None, None, 2.0) if dt is None else (float(dt[0]), float(dt[1]), float(e))
            self.n = n
            self.pos = self._solve(pred)
            self._write(st)
            return n
        finally:
            st.close()

    def _solve(self, pred):
        """Положения по принятым швам (ошибка < 0.5) + слабая привязка к столику; кадр без швов — по столику."""
        from .stitch import _solve_graph
        ok = [(a, b, v) for (a, b), v in self.pairs.items() if v[0] is not None and v[2] < 0.5 and b < len(pred)]
        if not ok:
            return pred.copy()
        I = np.array([o[0] for o in ok]); J = np.array([o[1] for o in ok])
        meas = np.array([[o[2][0], o[2][1]] for o in ok], float)
        w = 1.0 / (0.1 + 10.0 * np.array([o[2][2] for o in ok]))
        x0 = pred.copy()
        if self.pos is not None:
            k = min(len(self.pos), len(pred))
            x0[:k] = self.pos[:k]
        return _solve_graph(I, J, meas, w, pred, 0.01, x0=x0, rel=0.2)

    def _write(self, st):
        jd = os.path.join(self.folder, 'join')
        os.makedirs(jd, exist_ok=True)
        P = self.pos * self.r
        tmp = os.path.join(jd, 'positions_all.csv.tmp')
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(['id', 'x_px', 'y_px', 'scale'])
            for i, (x, y) in enumerate(P):
                wr.writerow([i, f"{x:.2f}", f"{y:.2f}", "1.0000"])
        os.replace(tmp, os.path.join(jd, 'positions_all.csv'))
        tmp = os.path.join(jd, PAIRS + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(dict(r=self.r, frames=self.n, pairs={f"{a},{b}": list(v) for (a, b), v in self.pairs.items()}), f)
        os.replace(tmp, os.path.join(jd, PAIRS))
        with open(os.path.join(jd, 'live.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(frames=self.n, t=time.time(), running=True), f)


def cached_pairs(folder, r):
    """Швы, измеренные на ходу (тем же уменьшением r): {(a, b): (dt | None, err)}."""
    try:
        d = json.load(open(os.path.join(folder, 'join', PAIRS), encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    if int(d.get('r') or 0) != int(r):
        return {}
    out = {}
    for k, v in d.get('pairs', {}).items():
        a, b = map(int, k.split(','))
        out[(a, b)] = (None if v[0] is None else np.array(v[:2], float), float(v[2]))
    return out


def running(folder):
    """Скан этой папки сейчас шьётся на ходу?"""
    try:
        d = json.load(open(os.path.join(folder, 'join', 'live.json'), encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return bool(d.get('running')) and time.time() - float(d.get('t') or 0) < 120


def watch(folder, every=0.4, parent=None):
    """Следить за папкой и дошивать, пока не попросят остановиться (join/live.stop) или не умрёт родитель."""
    lv = Live(folder)
    stop = os.path.join(folder, 'join', STOP)
    while True:
        done = os.path.exists(stop) or (parent is not None and not _alive(parent))
        try:
            lv.update()
        except Exception as e:                          # шов не вышел — не беда: quick после скана сошьёт всё
            try:
                from . import log as L
                L.debug('LIV-001', f"{folder}: {type(e).__name__}: {e}")
            except Exception:
                pass
        if done:
            break
        time.sleep(every)
    try:
        with open(os.path.join(folder, 'join', 'live.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(frames=lv.n, t=time.time(), running=False), f)
        if os.path.exists(stop):
            os.remove(stop)
    except OSError:
        pass


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class Runner:
    """Живая сшивка папки в своём процессе (ядра, не поток окна). stop() — дошить последнее и выйти."""
    def __init__(self, folder):
        import subprocess
        self.folder = folder
        os.makedirs(os.path.join(folder, 'join'), exist_ok=True)
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([here, os.environ.get('PYTHONPATH', '')]))
        kw = {'creationflags': 0x08000000 | 0x00004000} if os.name == 'nt' else {}
        self.p = subprocess.Popen([sys.executable, '-m', 'microcontrol.live', folder, str(os.getpid())], env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)

    def stop(self, timeout=60.0):
        try:
            open(os.path.join(self.folder, 'join', STOP), 'w').close()
        except OSError:
            pass
        try:
            self.p.wait(timeout)
        except Exception:
            self.p.kill()


if __name__ == '__main__':
    watch(sys.argv[1], parent=int(sys.argv[2]) if len(sys.argv) > 2 else None)
