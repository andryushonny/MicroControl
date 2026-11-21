"""Текстовые ответы (Result) и полоса прогресса (Progress)."""
import os, sys, time


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
    """Полоса прогресса в одну строку: Scan [######......] 12/40  Left 01:20."""
    def __init__(self, total, title, expect=None):
        """expect — ожидаемое время всего, с (из прошлых сканов): в начале «Left» берётся из него, потом всё
        больше — из того, как идёт на самом деле."""
        self.total, self.title, self.done, self.t0, self._t = max(1, total), title, 0, time.monotonic(), 0
        self.expect = expect

    def update(self, n=1, text=''):
        self.done = min(self.total, self.done + n)
        now = time.monotonic()
        if now - self._t < 0.3 and self.done < self.total:
            return
        self._t = now
        el = now - self.t0
        eta = el / self.done * (self.total - self.done)
        if self.expect:                                   # смесь: ожидание по прошлым сканам и текущий темп
            w = self.done / (self.done + 0.15 * self.total)
            eta = w * eta + (1 - w) * max(0.0, self.expect - el)
        bar = int(24 * self.done / self.total)
        sys.stdout.write(f"\r{self.title} [{'#' * bar}{'.' * (24 - bar)}] {self.done}/{self.total}"
                         f"  Left {int(eta // 60):02d}:{int(eta % 60):02d}  {text}   ")
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n")


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
