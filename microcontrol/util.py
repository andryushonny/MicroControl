"""Текстовые ответы (Result) и полоса прогресса (Progress)."""
import os, sys, time, threading, json
import numpy as np


def fmt(v):
    """Значение → короткий текст."""
    if isinstance(v, dict):
        return "   ".join(f"{str(k).upper() if len(str(k)) == 1 else k} {fmt(x)}" for k, x in v.items())
    if isinstance(v, tuple) and type(v) is not tuple:      # Position и т.п. печатают себя сами
        return repr(v)
    if isinstance(v, (list, tuple)):
        return ", ".join(fmt(x) for x in v)
    if isinstance(v, float):
        return f"{v:.1f}" if abs(v) >= 10 or v == int(v) else f"{v:.3g}"
    if v is None:
        return "-"
    return str(v)


_WIN = {}                                                  # последний размер окна (общий для manual и view)


def open_window(name, fullscreen=False):
    """Окно OpenCV одного размера во всех режимах: полный экран или последний размер (по умолчанию WINDOW_SIZE)."""
    import cv2
    from .config import WINDOW_SIZE
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | getattr(cv2, 'WINDOW_FREERATIO', 0x100))   # тянут край — кадр на всё окно,
                                                                                      # без серых полос; следующий — по размеру
    set_icon(name)
    if fullscreen:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return None
    W, H = _WIN.get('size', WINDOW_SIZE)
    cv2.resizeWindow(name, W, H)
    return W, H


def window_size(name, default=None, remember=False):
    """Текущий размер окна; remember=True (при закрытии) — запомнить его для следующего окна."""
    import cv2
    from .config import WINDOW_SIZE
    try:
        _, _, W, H = cv2.getWindowImageRect(name)
    except cv2.error:
        W = H = 0
    if W >= 100 and H >= 100:
        if remember:
            _WIN['size'] = (W, H)
        return W, H
    return default or _WIN.get('size', WINDOW_SIZE)


class Result(dict):
    """Ответ метода: заголовок + поля «ключ  значение». Работает как обычный dict."""
    def __init__(self, title, **fields):
        super().__init__(**fields)
        self.title = title

    def __repr__(self):
        w = max((len(k) for k in self), default=0)
        return "\n".join([self.title] + [f"  {k:<{w}}   {fmt(v)}" for k, v in self.items()])

    __str__ = __repr__

    def _repr_pretty_(self, p, cycle):
        p.text(repr(self))


# ------------------------------------------------------------------ остановка долгого дела из окна
_STOP = {'tid': None, 'on': False}


def stop_begin():
    """Поток долгого дела (из окна): с этой минуты его можно остановить (stop_request)."""
    _STOP.update(tid=threading.get_ident(), on=False)


def stop_end():
    _STOP.update(tid=None, on=False)


def stop_request():
    """Esc / Stop в окне: дело остановится на ближайшей полосе прогресса (или check_stop)."""
    _STOP['on'] = True


_JOBS = {}                                                 # потоки задач очереди: tid -> {'on', 'hold'}


def job_begin(tid):
    """Поток задачи очереди: его останавливают отдельно от окна (job_stop), Esc окна его не трогает."""
    _JOBS[tid] = {'on': False, 'hold': 0}


def job_end(tid):
    _JOBS.pop(tid, None)
    _ProgressMeta._opens.pop(tid, None)


def job_stop(tid):
    if tid in _JOBS:
        _JOBS[tid]['on'] = True


class no_stop:
    """with no_stop(): ...  — внутри дело само знает, как остановиться мягко (скан: после кадра, снятое
    сохраняется), полоса прогресса его не прерывает и силой не останавливают."""
    def __enter__(self):
        j = _JOBS.get(threading.get_ident())
        if j is not None:
            j['hold'] += 1
            self.tid = 'job'
            return self
        self.tid, _STOP['tid'] = _STOP['tid'], None
        return self

    def __exit__(self, *a):
        if self.tid == 'job':
            j = _JOBS.get(threading.get_ident())
            if j is not None:
                j['hold'] -= 1
            return
        _STOP['tid'] = self.tid


def stop_requested():
    return _STOP['on']


def stop_active(tid):
    """Поток tid можно остановить силой (не внутри no_stop)."""
    return _STOP['tid'] == tid


def check_stop():
    """В потоке долгого дела: остановку попросили — Stopped. В других потоках (окно, камера) — ничего."""
    tid = threading.get_ident()
    j = _JOBS.get(tid)
    if (_STOP['on'] and _STOP['tid'] == tid) or (j is not None and j['on'] and not j['hold']):
        from .errors import Stopped
        raise Stopped()


class _ProgressMeta(type):
    """Progress._open — своя у каждого потока (задачи очереди идут одновременно с делом окна)."""
    _opens = {}

    @property
    def _open(cls):
        t = threading.get_ident()
        return cls._opens.get(t if t in _JOBS else 0)

    @_open.setter
    def _open(cls, v):
        t = threading.get_ident()
        cls._opens[t if t in _JOBS else 0] = v


def _hook_now():
    """Куда полосе сообщать о себе: поток задачи очереди — своей задаче (или никуда), остальные — окну."""
    tid = threading.get_ident()
    if tid in Progress.hooks:
        return Progress.hooks[tid]
    return None if tid in _JOBS else Progress.hook


class Progress(metaclass=_ProgressMeta):
    """Полоса прогресса в одну строку:
    Scan             [############............]  48.21% / 100.00%   27 of 56 frames   00:41 passed, 00:44 left   Z 102 um
    unit — что считается (frames, seams, squares ...); text — что делается сейчас. Вложенная полоса (фокус
    внутри скана) молчит, пока открыта внешняя. Можно как with Progress(...) as pr: ..."""
    hooks = {}                                            # tid -> hook: полосы задач очереди
    hook = None                                           # f(title, доля 0..1, подпись, прошло с, осталось с | None)
                                                          # — полоса ещё и в окне (сшивка из mc.view)

    def __init__(self, total, title, expect=None, unit=''):
        """expect — ожидаемое время всего, с (из прошлых сканов): в начале «left» берётся из него, потом всё
        больше — из того, как идёт на самом деле."""
        check_stop()
        self.total, self.title, self.done, self.t0, self._t = max(1, total), title, 0, time.monotonic(), 0.0
        self.expect, self.unit, self.text, self.sub, self._w = expect, unit, '', '', 0
        self.step, self.sub_frac = 1, 0.0                   # вес текущего шага и сколько его сделала вложенная полоса
        o = Progress._open
        self.quiet = o is not None and not o.closed and time.monotonic() - max(o._t, o.t0) < 120
        self.parent = o if self.quiet else None             # вложенная: пишет о себе в подписи внешней
        self.closed = False
        Progress._open = self                                # самая глубокая открытая полоса
        if not self.quiet:
            self._show(time.monotonic())

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def extend(self, n):
        """Работы стало больше на n (ещё один раунд): доля не прыгает к 100 %, пока она идёт."""
        self.total += max(0, int(n))
        self.update(0, force=True)

    def update(self, n=1, text=None, force=False):
        """Сделано ещё n (n=0 — только сменить подпись text). Здесь же — остановка из окна (Stopped)."""
        check_stop()
        self.done = min(self.total, self.done + n)
        if n:
            self.sub_frac = 0.0
        if text is not None:
            self.text = text
        now = time.monotonic()
        if self.quiet:                                        # подпись внешней: «Focus plane 25% › Focus 33% › Z ...»
            self._t = now
            if self.parent is not None and not self.parent.closed:
                self.parent._set_sub(f"{self.title} {100 * self.done / self.total:.0f}%" + self._label(" › "),
                                     self._frac())
            return
        if now - self._t < (0.1 if _hook_now() is not None else 0.25) and self.done < self.total and not force:
            return
        self._show(now)

    def _label(self, lead=""):
        t = " › ".join(x for x in (self.text, self.sub) if x)
        return lead + t if t else ""

    def _frac(self):
        return min(1.0, (self.done + self.sub_frac * self.step) / self.total)

    def _set_sub(self, sub, frac=0.0):
        """Подпись от вложенной полосы (фокус внутри скана) идёт после своей; доля текущего шага (вес step)
        растёт вместе с ней."""
        self.sub, self.sub_frac = sub, (frac if sub else 0.0)
        self.update(0)

    def write(self, msg):
        """Напечатать строки (отчёт) так, чтобы полоса не смешалась с ними: полоса продолжится ниже."""
        top = self.parent if self.quiet else self
        if top is None or top.closed:
            print(msg)
            return
        sys.stdout.write("\r" + " " * top._w + "\r")
        print(msg)
        top._w = 0
        top._show(time.monotonic())

    def stage(self, text, n=0):
        """Новый этап: подпись сразу на экран (перед долгим шагом без своих долей — запись большого файла)."""
        self.update(n, text, force=True)

    def _show(self, now):
        self._t = now
        el = now - self.t0
        f = self._frac()
        eta = el / f * (1 - f) if f > 0 else None
        if self.done >= self.total:
            f = 1.0
        if self.expect:                                   # смесь: ожидание по прошлым сканам и текущий темп
            w = f / (f + 0.15)
            eta = w * (eta or 0.0) + (1 - w) * max(0.0, self.expect - el)
        t = lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}"
        bar = int(round(24 * f))
        cnt = f"{self.done} of {self.total} {self.unit}".rstrip() + "   " if self.unit is not None else ""
        line = (f"{self.title:<18} [{'#' * bar}{'.' * (24 - bar)}] {100 * f:6.2f}% / 100.00%   {cnt}"
                f"{t(el)} passed" + (f", {t(eta)} left" if eta is not None and self.done < self.total else "")
                + self._label("   "))
        h_ = _hook_now()
        if h_ is not None:
            try:
                h_(self.title, f, self._label(), el, eta if self.done < self.total else 0.0)
            except Exception:
                pass
        pad = max(0, self._w - len(line))
        self._w = len(line)
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if Progress._open is self:
            Progress._open = self.parent
        if self.quiet:
            if self.parent is not None and not self.parent.closed:
                self.parent._set_sub('', 0.0)
            return
        self._show(time.monotonic())
        sys.stdout.write("\n")
        sys.stdout.flush()


def save_image(path, img, params=None, title="Saving", pr=None):
    from .stats import timed
    with timed("Saving large pictures"):
        return _save_image(path, img, params, title, pr)


def _save_image(path, img, params=None, title="Saving", pr=None):
    """Записать (большую) картинку с полосой прогресса. Кодирование PNG/JPG идёт одним вызовом OpenCV,
    поэтому доля — оценка: сначала кодируется полоса ~1/50 высоты и по её времени считается всё; полоса
    дойдёт до 100 % только когда файл записан. pr — уже открытая полоса: тогда пишется в её подпись."""
    import cv2
    import threading
    ext = os.path.splitext(path)[1].lower() or '.png'
    if params is None:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if ext == '.png' else [cv2.IMWRITE_JPEG_QUALITY, 92]
    h, w = img.shape[:2]
    mb = img.nbytes / 2 ** 20
    name = os.path.basename(path)
    if mb < 8:                                             # маленький — сразу
        ok, buf = cv2.imencode(ext, img, params)
        if not ok:
            raise IOError(f"Cannot encode {path}.")
        buf.tofile(path)
        return path
    t0 = time.monotonic()
    rows = max(16, h // 50)
    cv2.imencode(ext, np.ascontiguousarray(img[:rows]), params)
    est = max(0.5, (time.monotonic() - t0) * h / rows * 1.1) + mb / 150   # кодирование + запись на диск
    res = {}

    def work():
        try:
            ok, buf = cv2.imencode(ext, img, params)
            if not ok:
                raise IOError(f"Cannot encode {path}.")
            res['enc'] = time.monotonic()
            buf.tofile(path)
            res['size'] = buf.nbytes
        except Exception as e:                             # ошибка — в основной поток
            res['err'] = e

    th = threading.Thread(target=work, daemon=True)
    own = pr is None
    bar = Progress(1000, title, unit=None) if own else pr
    base = bar.done
    th.start()
    try:
        while th.is_alive():
            th.join(0.3)
            el = time.monotonic() - t0
            what = "writing to disk" if 'enc' in res else "encoding"
            frac = min(0.99, el / est)
            if own:
                bar.done = int(1000 * frac)
                bar.update(0, f"{name}: {what}, {w} x {h} px, {mb:.0f} MB in memory (estimated)", force=True)
            else:
                bar.update(0, f"{name}: {what} ~{100 * frac:.0f}%, {w} x {h} px, {mb:.0f} MB", force=True)
        if 'err' in res:
            raise res['err']
        if own:
            bar.done = bar.total
            bar.update(0, f"{name}: saved, {res['size'] / 2 ** 20:.0f} MB on disk", force=True)
        else:
            bar.done = base
    finally:
        if own:
            bar.close()
    return path


def warp_clean(img, M, size, alpha=None, inverse=False):
    """Картинку — аффинно на холст size (w, h) без тёмной каймы на краях и стыках: дробный сдвиг смешивает край с
    чёрным «вне кадров», поэтому картинка берётся с весом своей маски (alpha; None — всё ярче 6) и делится на
    перенесённый вес. -> (картинка uint8, маска bool: веса больше половины)."""
    import cv2
    fl = cv2.INTER_LINEAR | (cv2.WARP_INVERSE_MAP if inverse else 0)
    M = np.asarray(M, np.float32)
    g = img.max(2) if img.ndim == 3 else img
    if alpha is not None and np.asarray(alpha).shape[:2] != g.shape[:2]:   # маска уровнем пирамиды на пиксель меньше
        alpha = cv2.resize(np.asarray(alpha), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    a = ((g > 6) if alpha is None else (np.asarray(alpha) > 0)).astype(np.float32)
    pm = img.astype(np.float32) * (a[..., None] if img.ndim == 3 else a)
    w = cv2.warpAffine(pm, M, tuple(int(v) for v in size), flags=fl, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wa = cv2.warpAffine(a, M, tuple(int(v) for v in size), flags=fl, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = wa > 0.5
    out = np.zeros(w.shape, np.uint8)
    out[m] = np.clip(w[m] / (wa[m][:, None] if w.ndim == 3 else wa[m]) + 0.5, 0, 255).astype(np.uint8)
    return out, m


def warp_clean_roi(img, M, size, alpha=None):
    """warp_clean, но только там, куда картинка попадает: видимый кусок исходника -> свой прямоугольник холста.
    Невидимое не считается (карта / просмотр с тысячами кадров). -> (картинка, маска, (x0, y0)) или None."""
    import cv2
    w, h = int(size[0]), int(size[1])
    M = np.asarray(M, np.float64)
    ih, iw = img.shape[:2]
    cr = np.array([(0, 0), (iw, 0), (0, ih), (iw, ih)], float) @ M[:, :2].T + M[:, 2]
    x0, y0 = np.maximum(np.floor(cr.min(0)).astype(int) - 1, 0)
    x1, y1 = np.minimum(np.ceil(cr.max(0)).astype(int) + 2, (w, h))
    if x1 <= x0 or y1 <= y0:
        return None                                         # не видно — не рисуется
    A, t = M[:, :2], M[:, 2]
    Ai = np.linalg.inv(A)
    rc = (np.array([(x0, y0), (x1, y0), (x0, y1), (x1, y1)], float) - t) @ Ai.T   # видимое окно -> исходник
    sx0, sy0 = np.maximum(np.floor(rc.min(0)).astype(int) - 2, 0)
    sx1, sy1 = np.minimum(np.ceil(rc.max(0)).astype(int) + 3, (iw, ih))
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    sub = img[sy0:sy1, sx0:sx1]
    al = None
    if alpha is not None:
        al = np.asarray(alpha)
        if al.shape[:2] != (ih, iw):
            al = cv2.resize(al, (iw, ih), interpolation=cv2.INTER_NEAREST)
        al = al[sy0:sy1, sx0:sx1]
    M2 = np.c_[A, t + A @ (sx0, sy0) - (x0, y0)]
    pic, msk = warp_clean(sub, M2, (x1 - x0, y1 - y0), alpha=al)
    return pic, msk, (x0, y0)


def wheel_delta(flags):
    """Направление колеса мыши (>0 — от себя). В части сборок OpenCV нет cv2.getMouseWheelDelta:
    тогда берётся старшее 16-битное слово flags со знаком (так его кладёт OpenCV)."""
    import cv2
    f = getattr(cv2, 'getMouseWheelDelta', None)
    if f is not None:
        try:
            return f(flags)
        except Exception:
            pass
    d = (int(flags) >> 16) & 0xFFFF
    return d - 0x10000 if d >= 0x8000 else d


def wheel_gesture(ev, flags):
    """Колесо мыши или тачпад -> ('zoom', k) или ('pan', dx, dy) в пикселях окна.
    Колесо мыши (шаги по 120) — масштаб. Тачпад: два пальца (мелкие шаги, горизонтальная прокрутка) — двигать
    картинку; щипок (приходит как Ctrl + колесо) — масштаб. Shift + колесо — двигать вбок."""
    import cv2
    d = wheel_delta(flags)
    if not d:
        return ('pan', 0.0, 0.0)
    ctrl = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
    shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
    px = 60.0 * d / 120.0 if abs(d) >= 8 else 60.0 * np.sign(d)
    if ev == getattr(cv2, 'EVENT_MOUSEHWHEEL', -1):
        return ('pan', -px, 0.0)
    if ctrl:                                              # щипок на тачпаде / Ctrl + колесо
        return ('zoom', 1.25 ** (d / 120.0) if abs(d) >= 8 else (1.25 if d > 0 else 0.8))
    if shift:
        return ('pan', px, 0.0)
    if abs(d) >= 8 and d % 120:                           # плавная прокрутка тачпада двумя пальцами
        return ('pan', 0.0, px)
    return ('zoom', 1.25 if d > 0 else 0.8)


EXIT = -27          # «выйти» (от тестов и когда нажатия кончились)


def exit_key(key):
    """Выход из окна: Shift+Esc или Ctrl+Esc (одиночный Esc только отменяет). Windows: какие клавиши
    зажаты, спрашивается у системы; Linux / GTK: модификаторы — в старших битах кода клавиши."""
    if key == EXIT:
        return True
    if key is None or key < 0 or key & 0xFF != 27:
        return False
    if key >> 16:
        return True
    try:
        import ctypes
        u = ctypes.windll.user32
        return bool((u.GetAsyncKeyState(0x10) | u.GetAsyncKeyState(0x11)) & 0x8000)
    except Exception:
        return False


def shift_held(key=None):
    """Зажат ли Shift (для стрелок): Windows — спросить систему; Linux / GTK — старшие биты кода клавиши."""
    if key is not None and key > 0 and (key >> 16) & 0x1:
        return True
    try:
        import ctypes
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000)
    except Exception:
        return False


def icon_path():
    """Значок программы: microcontrol/icon.ico или ~/.microcontrol/icon.ico (если файл положили)."""
    for p in (os.path.join(os.path.dirname(__file__), 'icon.ico'),
              os.path.join(os.path.expanduser('~'), '.microcontrol', 'icon.ico')):
        if os.path.exists(p):
            return p
    return None


def _app_id():
    """Своя группа на панели задач Windows (иначе значок берётся у python.exe). Ставится один раз,
    до создания окон — поэтому вызывается при импорте библиотеки."""
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MicroControl.Microscope")
    except Exception:
        pass


def _windows_of_process(title):
    """Окна верхнего уровня этого процесса с таким заголовком (Windows)."""
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    out, pid = [], os.getpid()
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(h, _):
        p_ = wintypes.DWORD()
        u.GetWindowThreadProcessId(h, ctypes.byref(p_))
        if p_.value == pid:
            n = u.GetWindowTextLengthW(h)
            buf = ctypes.create_unicode_buffer(n + 1)
            u.GetWindowTextW(h, buf, n + 1)
            if buf.value == title:
                out.append(h)
        return True
    u.EnumWindows(proto(cb), 0)
    return out


def set_icon(title):
    """Значок окна OpenCV (заголовок, Alt+Tab, панель задач) — только Windows. Возвращает, что получилось
    (для проверки: mc.util.set_icon('MicroControl') при открытом окне)."""
    p = icon_path()
    if not p:
        return "no icon file (microcontrol/icon.ico or ~/.microcontrol/icon.ico)"
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
    except Exception:
        return "not Windows: nothing to do"
    try:
        u.LoadImageW.restype = wintypes.HANDLE
        u.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT, ctypes.c_int, ctypes.c_int,
                                 wintypes.UINT]
        u.SendMessageW.restype = ctypes.c_ssize_t
        u.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        u.FindWindowW.restype = wintypes.HWND
        set_cls = getattr(u, 'SetClassLongPtrW', None) or u.SetClassLongW
        set_cls.restype = ctypes.c_size_t
        set_cls.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        hw = _windows_of_process(title) or [u.FindWindowW(None, title)]
        hw = [h for h in hw if h]
        if not hw:
            return f"no window {title!r} yet"
        LR_LOADFROMFILE = 0x10
        big = u.LoadImageW(None, p, 1, u.GetSystemMetrics(11), u.GetSystemMetrics(12), LR_LOADFROMFILE)
        small = u.LoadImageW(None, p, 1, u.GetSystemMetrics(49), u.GetSystemMetrics(50), LR_LOADFROMFILE)
        if not big or not small:
            return f"could not load {p}"
        for h in hw:
            u.SendMessageW(h, 0x80, 1, big)                   # WM_SETICON, ICON_BIG: Alt+Tab, панель задач
            u.SendMessageW(h, 0x80, 0, small)                 # ICON_SMALL: заголовок окна
            set_cls(h, -14, big)                              # GCLP_HICON
            set_cls(h, -34, small)                            # GCLP_HICONSM
        return f"icon set on {len(hw)} window(s)"
    except Exception as e:
        return f"failed: {e}"


def named_folder(scans, objective, name, reuse=False):
    """Папка для скана с именем name: новая «<дата>_<время>_<объектив>x_<имя>»; reuse=True и такая уже есть
    (то же имя и объектив) — та же папка, старое содержимое заменяется. Возвращает путь (папка не создаётся)."""
    import re as _re
    import shutil as _sh
    from datetime import datetime as _dt
    safe = _re.sub(r'[^\w\-]+', '_', name or '').strip('_') or 'scan'
    if reuse and os.path.isdir(scans):
        pat = _re.compile(r"^\d{4}-\d\d-\d\d_\d{6}_" + _re.escape(f"{objective}x_{safe}") + "$")
        old = sorted(d for d in os.listdir(scans) if pat.match(d))
        if old:
            path = os.path.join(scans, old[-1])
            _sh.rmtree(path)
            print(f"Writing into the existing folder {old[-1]} (the previous scan with this name is replaced)")
            return path
    return os.path.join(scans, _dt.now().strftime("%Y-%m-%d_%H%M%S") + f"_{objective}x_{safe}")


# ---------------------------------------------------------------- раскладка окон (у пользователя, для обоих окон)
_UI_P = os.path.join(os.path.expanduser('~'), '.microcontrol', 'ui.json')
UI_DEFAULTS = dict(panes='one', list='right')            # panes: one | camera|map | map|camera; list: right | left


def ui_pref(key):
    """Настройка раскладки (View -> Layout): panes, list."""
    if os.environ.get('MC_TEST') and not os.environ.get('MC_UI'):   # проверки — всегда раскладка по умолчанию
        return UI_DEFAULTS.get(key)
    try:
        with open(_UI_P, encoding='utf-8') as f:
            v = json.load(f).get(key)
    except (OSError, ValueError):
        v = None
    return v if v is not None else UI_DEFAULTS.get(key)


def set_ui_pref(key, val):
    try:
        with open(_UI_P, encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {}
    d[key] = val
    os.makedirs(os.path.dirname(_UI_P), exist_ok=True)
    with open(_UI_P, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1)
