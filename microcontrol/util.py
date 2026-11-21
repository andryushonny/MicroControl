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
    def __init__(self, total, title):
        self.total, self.title, self.done, self.t0, self._t = max(1, total), title, 0, time.monotonic(), 0

    def update(self, n=1, text=''):
        self.done = min(self.total, self.done + n)
        now = time.monotonic()
        if now - self._t < 0.3 and self.done < self.total:
            return
        self._t = now
        el = now - self.t0
        eta = el / self.done * (self.total - self.done)
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


def set_icon(title):
    """Значок окна OpenCV и панели задач (Windows); на других системах — ничего."""
    p = icon_path()
    if not p:
        return
    try:
        import ctypes
        u = ctypes.windll.user32
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MicroControl")   # своя кнопка на панели задач
        hwnd = u.FindWindowW(None, title)
        if not hwnd:
            return
        u.LoadImageW.restype = ctypes.c_void_p
        big = u.LoadImageW(None, p, 1, 32, 32, 0x10)          # IMAGE_ICON, LR_LOADFROMFILE
        small = u.LoadImageW(None, p, 1, 16, 16, 0x10)
        u.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
        u.SendMessageW(hwnd, 0x80, 1, big)                    # WM_SETICON, ICON_BIG
        u.SendMessageW(hwnd, 0x80, 0, small)                  # ICON_SMALL
    except Exception:
        pass
