"""Текстовые ответы (Result) и полоса прогресса (Progress)."""
import os, sys, time
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
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
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


class Progress:
    """Полоса прогресса в одну строку:
    Scan             [############............]  48.21% / 100.00%   27 of 56 frames   00:41 passed, 00:44 left   Z 102 um
    unit — что считается (frames, seams, squares ...); text — что делается сейчас. Вложенная полоса (фокус
    внутри скана) молчит, пока открыта внешняя. Можно как with Progress(...) as pr: ..."""
    _open = None                                          # открытая сейчас полоса (вложенные молчат)

    def __init__(self, total, title, expect=None, unit=''):
        """expect — ожидаемое время всего, с (из прошлых сканов): в начале «left» берётся из него, потом всё
        больше — из того, как идёт на самом деле."""
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

    def update(self, n=1, text=None, force=False):
        """Сделано ещё n (n=0 — только сменить подпись text)."""
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
        if now - self._t < 0.25 and self.done < self.total and not force:
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
