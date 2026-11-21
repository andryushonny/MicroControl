"""Microscope — управление ATR8300 как 3D-принтером: движение, свет, лазер, камера, фокус, G-код, скан."""
import atexit, math, os, time, warnings
from datetime import datetime
import numpy as np
import cv2
from . import protocol as P, focus as F, log as L
from .config import (OBJECTIVES, Z_RANGE, z_limits, VID, PID, UM_PER_STEP, SENSOR_PIXEL_UM, CHUNK_STEPS, MIN_INTERVAL_S,
                     REPLY_TIMEOUT_S, FRAME_FLUSH)
from .errors import SafetyError, LimitError, CommError, CameraError, FocusError, StopError
from .control import ControlLock, stop_requested_after
from .device import DeviceMemory
from .util import Result, fmt

AXES = ('x', 'y', 'z')
ORIENTATION = ('map_flip_x', 'map_flip_y', 'arrows_flip_x', 'arrows_flip_y', 'turn_180', 'swap_xy', 'mirror_x', 'mirror_y', 'reverse_x', 'reverse_y', 'reverse_z')   # помнится для прибора
WEAK_FOCUS = 1.3      # пик резкости ниже 1.3 × типичной — фокус ненадёжен (мало деталей)

# старые имена (до 0.7.0) -> новые: старое имя даёт ошибку с подсказкой, а не молчит
RENAMED = dict(dwell='photo_pause', settle='move_pause', light_warmup='light_pause', travel='xy_limit',
               laser_limit='laser_max', flip_x='mirror_x', flip_y='mirror_y', invert_x='reverse_x',
               invert_y='reverse_y', invert_z='reverse_z', focus_metric='focus_method', focus_spacing='focus_every',
               machine_position='real_position', field='frame_size', snap='photo', last_capture='last_photo',
               dark='dark_frame', flat='empty_frame', identify='info', workdir='folder', sdk_dir='sdk_folder',
               speed='motor_speed', home='find_limits')


def ports():
    """Список COM-портов. Контроллер ATR8300 отмечен звёздочкой; 'sim' — симулятор."""
    try:
        from serial.tools import list_ports
        found = list_ports.comports()
    except ImportError:
        found = []
    r = Result("Ports")
    for p in found:
        r[p.device] = ("* ATR8300  " if (p.vid, p.pid) == (VID, PID) else "") + (p.description or '')
    r['sim'] = "Simulator"
    return r


def imwrite(path, img):
    """Сохранить картинку (работает с русскими путями)."""
    ext = os.path.splitext(path)[1].lower() or '.png'
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_PNG_COMPRESSION, 3] if ext == '.png' else [])
    if not ok:
        raise IOError(f"Cannot encode {path}.")
    buf.tofile(path)
    return path


def imread(path):
    """Прочитать картинку (работает с русскими путями)."""
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)


class Position(tuple):
    """Положение столика (x, y, z) в мкм. Печатается как: X 100.0   Y -50.0   Z 205.6 um."""
    x = property(lambda s: s[0])
    y = property(lambda s: s[1])
    z = property(lambda s: s[2])

    def __repr__(self):
        return f"X {self[0]:.1f}   Y {self[1]:.1f}   Z {self[2]:.1f} um"


class Setting:
    """Параметр микроскопа: читается и меняется как обычное свойство, значение проверяется.
    m.photo_pause -> 0.3;  m.photo_pause = 0.5;  m.photo_pause = 100 -> [E208] ... allowed 0 .. 10."""

    def __init__(self, kind, lo=None, hi=None, unit='', doc=''):
        self.kind, self.lo, self.hi, self.unit, self.doc = kind, lo, hi, unit, doc
        self.__doc__ = f"{doc} ({self.allowed()}{' ' + unit if unit else ''})"

    def __set_name__(self, owner, name):
        self.name = name

    def allowed(self, m=None):
        if self.kind == 'bool':
            return "True / False"
        if self.kind == 'level':
            return f"{self.lo} .. {self.hi}, True / False"
        if self.kind == 'choice':
            return " / ".join(str(v) for v in self.hi)
        lo, hi = self.lo, self.hi
        if m is not None and callable(lo):
            lo = lo(m)
        if m is not None and callable(hi):
            hi = hi(m)
        if callable(lo) or callable(hi):
            return "see m.settings"
        return f"{lo:g} .. {hi:g}"

    def __get__(self, m, cls):
        if m is None:
            return self
        get = getattr(m, '_get_' + self.name, None)
        return get() if get else m._cfg[self.name]

    def __set__(self, m, v):
        v = self.check(m, v)
        old = m._cfg.get(self.name)
        apply = getattr(m, '_apply_' + self.name, None)
        if apply:
            apply(v)
        m._cfg[self.name] = v
        if old is not None and old != v and not apply:   # у «железных» параметров своя запись
            L.event('SET-001', f"{self.name} = {fmt(v)}{' ' + self.unit if self.unit else ''}.")
        if self.name in ORIENTATION and getattr(m, '_mem', None) is not None and old != v:
            m._mem.data.setdefault('orientation', {})[self.name] = v      # ориентация помнится для этого прибора
            m._mem.save()

    def check(self, m, v):
        bad = lambda: SafetyError('E208', f"{self.name} = {v!r}. Allowed: {self.allowed(m)}{' ' + self.unit if self.unit else ''}.")
        if self.kind == 'bool':
            if not isinstance(v, bool):
                raise bad()
            return v
        if self.kind == 'level':                     # True = 5, False = 0, или 0..5
            if isinstance(v, bool):
                return 5 if v else 0
            if not isinstance(v, int) or not self.lo <= v <= self.hi:
                raise bad()
            return v
        if self.kind == 'choice':
            if v not in self.hi:
                raise bad()
            return v
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise bad()
        lo = self.lo(m) if callable(self.lo) else self.lo
        hi = self.hi(m) if callable(self.hi) else self.hi
        if not lo <= v <= hi:
            raise bad()
        if self.kind == 'int' and v != int(v):         # контроллер принимает только целые: дробь — ошибка, не округление
            raise SafetyError('E208', f"{self.name} = {v!r}. Only whole numbers{' of ' + self.unit if self.unit else ''}.")
        return int(v) if self.kind == 'int' else float(v)


def _timed(fn, seconds, err):
    """Вызвать fn() не дольше seconds. Вызовы DLL камеры и драйвера порта могут зависнуть навсегда
    (камера занята другой программой, USB завис) — тогда ошибка вместо вечного ожидания."""
    import threading
    box = {}

    def run():
        try:
            box['v'] = fn()
        except BaseException as e:
            box['e'] = e
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise err
    if 'e' in box:
        raise box['e']
    return box.get('v')


def _open_serial(port):
    """Открыть порт как .NET SerialPort в штатном ПО: линии DTR и RTS выключены (pyserial по умолчанию их включает)."""
    import serial
    ser = serial.Serial(baudrate=115200, timeout=REPLY_TIMEOUT_S, write_timeout=2)
    ser.dtr, ser.rts = False, False
    ser.port = port
    ser.open()
    return ser


def _step_msg(text):
    """Строка хода подключения — сразу на экран (видно, где остановилось)."""
    print(text, flush=True)


def _sim0(m, lo):
    return 0.0 if m.port == 'sim' else lo


class Microscope:
    """Микроскоп Optosky ATR8300.

    Подключение:  m = Microscope()            порт ищется сам
                  m = Microscope('COM3')      порт вручную
                  m = Microscope('sim')       симулятор, без прибора
    Сценарий:     m.manual()                  посмотреть образец, отметить зону клавишами 1 и 2
                  folder = m.scan()           снять зону
                  m.close()                   отключиться
    Как принтер:  m.gcode("G0 X500 Y200")     команды G-кода (единицы — мкм)
    Параметры:    m.settings                  все параметры; m.photo_pause = 0.5 — изменить
    Справка:      mc.help()

    Координаты в мкм от точки подключения. +Z — столик вверх, к объективу.
    """

    # ---- параметры (читаются и меняются как свойства, значения проверяются) ----
    objective = Setting('choice', None, tuple(OBJECTIVES), 'x', "Objective magnification")
    light = Setting('level', 0, 5, '', "Light brightness: 0 = off, 5 = full")
    laser = Setting('int', 0, lambda m: m.laser_max, 'mW', "Laser power in whole mW, 0 = off (needs allow_laser=True)")
    laser_max = Setting('float', 0, 100, 'mW', "Highest allowed laser power")
    exposure = Setting('float', 0.05, 1000, 'ms', "Camera exposure time")
    pixel_size = Setting('float', 0.001, 10, 'um', "Size of one camera pixel on the sample")
    map_flip_x = Setting('bool', doc="Manual: mini map turned over left-right (only the picture of the map)")
    map_flip_y = Setting('bool', doc="Manual: mini map turned over up-down (only the picture of the map)")
    arrows_flip_x = Setting('bool', doc="Manual: left / right arrows the other way (only the keys; pictures and joining do not change)")
    arrows_flip_y = Setting('bool', doc="Manual: up / down arrows the other way (only the keys)")
    turn_180 = Setting('bool', doc="Turn the camera picture by 180 degrees (on the ATR8300 it comes upside down)")
    swap_xy = Setting('bool', doc="Camera turned by 90 degrees: stage X goes up-down on the screen, Y left-right")
    mirror_x = Setting('bool', doc="Moving the stage along X moves the picture the other way")
    mirror_y = Setting('bool', doc="Moving the stage along Y moves the picture the other way")
    reverse_x = Setting('bool', doc="X motor turns the other way")
    reverse_y = Setting('bool', doc="Y motor turns the other way")
    reverse_z = Setting('bool', doc="Z motor turns the other way")
    motor_step = Setting('float', 0.5, 50, 'um', "Stage shift for one motor step")
    max_move = Setting('float', 1, 2000, 'um', "Largest single move in X and Y")
    max_move_z = Setting('float', 1, 1000, 'um', "Largest single move of the stage up, towards the objective")
    working_distance = Setting('float', 100, 40000, 'um', "Objective working distance (on its barrel, WD): sets the Z limits")
    max_move_down = Setting('float', 1, 20000, 'um', "Largest single move of the stage down, away from the objective")
    xy_limit = Setting('float', 100, 50000, 'um', "How far the stage may go in X and Y from the start point, each way")
    z_max = Setting('float', lambda m: m.position.z, lambda m: m.position.z + 100000, 'um', "Ceiling: highest stage position")
    z_min = Setting('float', lambda m: m.position.z - 100000, lambda m: m.position.z, 'um', "Floor: lowest stage position")
    motor_speed = Setting('float', 50, 20000, 'um/s', "Real motor speed (set by the controller, not by us); MicroControl waits distance / motor_speed after each move")
    max_jump = Setting('float', 100, 25000, 'um', "Longest single move by goto or G-code; a longer one is refused as suspicious")
    move_pause = Setting('float', lambda m: _sim0(m, 0.05), 10, 's', "Pause after every move (the stage stops shaking)")
    photo_pause = Setting('float', 0, 10, 's', "Scan: pause before each picture")
    light_pause = Setting('float', 0, 120, 's', "Pause after switching the light on (brightness becomes stable)")
    focus_method = Setting('choice', None, tuple(F.METHODS), '', "Sharpness measure for autofocus")
    focus_range = Setting('float', 1, 2000, 'um', "Autofocus search window, each way")
    focus_step = Setting('float', 1, 500, 'um', "Autofocus step")
    focus_every = Setting('int', 1, 100, 'frames', "Scan: focus points every N frames (focus plane)")
    refocus = Setting('bool', doc="Scan: focus again and re-take a blurry frame")
    scan_check = Setting('bool', doc="Scan: first check with the camera which way the picture moves (m.check_orientation())")
    still_check = Setting('bool', doc="Scan: before each picture check that the stage stopped shaking (2 frames compared)")
    frame_format = Setting('choice', None, ('jpg', 'h5'), '', "Scan: frames as JPG files (small) or HDF5 batches (lossless)")
    frame_scale = Setting('float', 0.1, 1, '', "Scan: stored frame size, 0.5 = half (0.5 keeps all optical detail up to 20x)")
    jpg_quality = Setting('int', 50, 100, '', "Scan: JPG quality of the stored frames")
    batch_size = Setting('int', 1, 1000, 'frames', "Scan: frames per HDF5 file (a file can be opened when it is full)")
    save_jpg = Setting('bool', doc="Scan: also save every frame at once as JPG in <scan>/jpg/ to look at it during the scan")
    jpg_scale = Setting('float', 0.05, 1, '', "Scan: size of those JPG, 1 = full size, 0.5 = half")
    log_bytes = Setting('bool', doc="Write raw controller bytes to the log")

    def __init__(self, port='auto', objective=None, camera=True, allow_laser=False, allow_homing=False, cooling=True,
                 folder='.', sdk_folder=None, force=False, **settings):
        object.__setattr__(self, '_cfg', {})
        self._lock = None
        self.port = port
        self.folder = os.path.abspath(folder)
        self.dirs = {d: os.path.join(self.folder, d) for d in ('logs', 'photos', 'scans', 'references')}
        if objective not in OBJECTIVES:                   # потолок Z считается от объектива — без догадок
            raise SafetyError('E207', f"{objective!r}. Tell which objective is in place: "
                                      f"Microscope(objective=10). Known: {', '.join(map(str, OBJECTIVES))}.")
        self.log_file = L.start(self.dirs['logs'])
        L.header(Port=port, Objective=f"{objective}x", Laser="allowed" if allow_laser else "locked")
        self.allow_laser, self.allow_homing = allow_laser, allow_homing
        self.area = None                       # зона скана (Area): m.manual() или mc.Area(...)
        self.last_scan = None                  # папка последнего скана
        self.references = {}                   # тёмный кадр и кадр пустого места: m.dark_frame(), m.empty_frame()
        self._p = dict.fromkeys(AXES, 0)       # положение, шаги мотора
        self._blocked, self._uncertain = {}, set()
        self._last = 0.0
        self._ser = self.cam = self._world = self._mem = None
        self._quiet = False
        self._frame_px = (2592, 1944)
        self._plane = None                     # плоскость фокуса (G29): z = a + b*x + c*y, мкм (машинные)
        self._off = dict.fromkeys(AXES, 0.0)   # смещение нуля (offset), мкм: рабочие = машинные − смещение
        self._t_connect = time.time()
        self._motors_on = False                # штатное ПО включает моторы X/Y (0x24 01) перед первым движением
        self._cooling = cooling
        self._stopped = False
        sim = port == 'sim'
        self._fast = sim
        self._cfg.update(objective=objective, light=0, laser=0, laser_max=10.0, exposure=20.0,
                         map_flip_x=False, map_flip_y=False, arrows_flip_x=False, arrows_flip_y=False, turn_180=False, swap_xy=False, mirror_x=False, mirror_y=False, reverse_x=False, reverse_y=False, reverse_z=False,
                         motor_step=UM_PER_STEP, max_move=2000.0, max_move_down=2000.0, xy_limit=25000.0, max_jump=3000.0,
                         motor_speed=3000.0,
                         move_pause=0.0 if sim else 0.3, photo_pause=0.0 if sim else 0.3, light_pause=0.0 if sim else 2.0,
                         focus_method='laplacian', focus_every=5, refocus=False, batch_size=100, save_jpg=True, jpg_scale=0.5,
                         frame_format='jpg', frame_scale=0.5, jpg_quality=90, scan_check=True, still_check=True,
                         log_bytes=True)
        self._open_port(port, force)
        try:
            self._handshake()
            _step_msg(f"  Controller   {self.device['Model']}, serial {self.device['Serial']}, type {self.device['Type']}")
            mem_dir = os.path.join(self.folder, '.microcontrol', 'devices') if sim else None
            self._mem = DeviceMemory(*self._ident, folder=mem_dir)
            restored = self._restore()
            orient = dict(self._mem.data.get('orientation', {}))       # ориентация этого прибора (помнится)
            if not sim:
                orient.setdefault('swap_xy', True)     # ATR8300: камера повёрнута на 90° к осям столика (проверено)
                orient.setdefault('map_flip_x', True)  # ATR8300: так мини-карта совпадает с положением на подложке
                orient.setdefault('map_flip_y', True)  # (по словам пользователя); [ и ] в ручном режиме — поменять
                orient.setdefault('turn_180', True)    # ATR8300: кадр камеры перевёрнут (отражён по X и Y) — проверено
            self._cfg.update(orient)
            t, p = self._u2p(self.xy_limit), self._p          # пределы — от точки подключения
            self._lim = {'x': (p['x'] - t, p['x'] + t), 'y': (p['y'] - t, p['y'] + t),
                         'z': (p['z'] - self._u2p(Z_RANGE), p['z'] + self._u2p(Z_RANGE))}
            self.objective = objective
            if camera:
                if self._world is not None:
                    from .sim import SimCamera
                    self.cam = SimCamera(self._world)
                else:
                    from .camera import MVCamera
                    _step_msg("  Camera       opening ...")
                    self.cam = _timed(lambda: MVCamera(sdk_folder), 20, CameraError(
                        'E505', "No answer in 20 s while opening. Close the Optosky software (check Task Manager "
                                "for its background process), unplug and plug in the camera USB, restart the kernel. "
                                "To work without the camera: Microscope(..., camera=False)."))
                self.exposure = 20.0
                _timed(lambda: self.image(0), 15, CameraError(
                    'E505', "No picture in 15 s. Unplug and plug in the camera USB, restart the kernel."))
                _step_msg(f"  Camera       {self.cam.info()['model']}, frame {self._frame_px[0]} x {self._frame_px[1]} px")
                L.event('CON-001', f"Camera {self.cam.info()['model']}, frame {self._frame_px[0]} x {self._frame_px[1]} px.")
                was = getattr(self.cam, 'was', {})
                if any(was.values()):                          # другая программа оставила камеру перевёрнутой
                    _step_msg(f"  Camera       its own mirror / rotation was on ({was}): switched off")
                    L.event('CON-006', f"Camera mirror / rotation was on: {was}; switched off.", 30)
            o_ = {k: bool(self._cfg.get(k)) for k in ('turn_180', 'swap_xy', 'mirror_x', 'mirror_y', 'map_flip_x', 'map_flip_y')}
            _step_msg("  Orientation  " + (", ".join(k for k, v in o_.items() if v) or "default") + "   (manual: T, O, [ and ])")
            for k, v in settings.items():                      # те же проверки, что при m.<имя> = ...
                setattr(self, k, v)
            self.light = 5                                     # как штатное ПО при подключении
            if cooling:                                        # и охлаждение спектрометра (слышен вентилятор)
                self._cmd(P.SET_TEC, P.tec_data(P.COOL_C))
                L.event('CON-005', f"Spectrometer cooling {P.COOL_C} C (as the Optosky software).")
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)
        r = Result("Connected", Port=self.port, Objective=f"{objective}x",
                   Camera="Yes" if self.cam else "No", Log=self.log_file)
        if self._mem.home:
            r['Home'] = f"set {self._mem.home['time']}"
        if restored:
            r['Position'] = f"{self.position} (from the last session, not confirmed)"
            r['Note'] = restored
        elif not self._mem.home:
            r['Home'] = "not set: coordinates start here. Set once: m.set_home()"
        print(r)

    _PUBLIC = {'port', 'folder', 'dirs', 'log_file', 'allow_laser', 'allow_homing', 'area', 'last_scan',
               'last_photo', 'references', 'cam', 'device'}

    def __setattr__(self, k, v):
        """Только известные имена: опечатка (m.photo_paus = 1) — сразу ошибка, а не новая переменная."""
        if k.startswith('_') or isinstance(getattr(type(self), k, None), Setting):
            return object.__setattr__(self, k, v)
        if k in self._PUBLIC:
            if k == 'area' and v is not None and type(v).__name__ != 'Area':
                raise SafetyError('E208', f"area = {v!r}. Use mc.Area(x0, y0, x1, y1) or None.")
            return object.__setattr__(self, k, v)
        if isinstance(getattr(type(self), k, None), property):
            raise SafetyError('E208', f"{k} is read-only.")
        if k in RENAMED:
            raise SafetyError('E105', f"{k!r} is now called {RENAMED[k]!r}.")
        import difflib
        names = [n for n, s in type(self).__dict__.items() if isinstance(s, Setting)]
        near = difflib.get_close_matches(k, names, 1)
        raise SafetyError('E105', f"{k!r}." + (f" Did you mean {near[0]!r}?" if near else "") + " See m.settings.")

    def __getattr__(self, k):
        if k in RENAMED:
            raise AttributeError(f"{k!r} is now called {RENAMED[k]!r}: m.{RENAMED[k]}")
        import difflib
        names = [n for n in dir(type(self)) if not n.startswith('_')]
        near = difflib.get_close_matches(k, names, 1)
        raise AttributeError(f"Microscope has no {k!r}." + (f" Did you mean {near[0]!r}?" if near else ""))

    # ---------------- применение параметров к прибору ----------------
    def _apply_working_distance(self, v, log=True):
        """Пределы по Z из рабочего расстояния: подъём за команду и потолок (от текущего положения)."""
        step = z_limits(v)
        self._cfg.update(working_distance=float(v), max_move_z=float(step))
        if log:
            L.event('MOV-005', f"Working distance {v:.0f} um: step up {step:.0f} um.")

    def _apply_objective(self, v):
        prof = OBJECTIVES[v]
        self._cfg.update(pixel_size=prof['pixel_size'], focus_range=float(prof['focus_range']),
                         focus_step=float(prof['focus_step']))
        self._cfg['objective'] = v
        self._apply_working_distance(float(prof['wd']), log=False)
        self._sync_limits()
        if self._world is not None:
            self._world.set_objective(v)             # настоящий объектив симулятора
        L.event('MOV-005', f"Objective {v}x. Pixel {prof['pixel_size']} um, ceiling Z {self.z_max:.0f} um.")

    def _apply_light(self, v):
        self._cmd(P.LIGHT, int(v).to_bytes(2, 'big'))
        L.event('LGT-001', f"Light level {v}." if v else "Light off.")
        if v and not self._cfg.get('light') and self._cfg.get('light_pause'):
            time.sleep(self.light_pause)

    def _apply_laser(self, v):
        if v and not self.allow_laser:
            raise SafetyError('E401', "Connect with Microscope(..., allow_laser=True).")
        if not self.allow_laser:
            return
        if v:
            warnings.warn("Laser is on. Wear safety goggles.")
        self._cmd(P.LASER, int(round(v)).to_bytes(2, 'big'))
        L.event('LSR-001', f"Laser {v:g} mW." if v else "Laser off.", 30 if v else 20)

    def _apply_exposure(self, v):
        if self.cam:
            self.cam.set(exposure_ms=v)
            L.event('CAM-002', f"Exposure {v:g} ms.")

    def _apply_motor_step(self, v):
        self._cfg['motor_step'] = v

    def _apply_xy_limit(self, v):
        t = self._u2p(v)
        self._lim['x'], self._lim['y'] = (-t, t), (-t, t)

    def _apply_z_max(self, v):
        self._lim['z'] = (self._lim['z'][0], self._u2p(v))

    def _apply_z_min(self, v):
        self._lim['z'] = (self._u2p(v), self._lim['z'][1])

    def _apply_log_bytes(self, v):
        for h in L.logger.handlers:
            h.setLevel('DEBUG' if v else 'INFO')

    def _sync_limits(self):
        self._cfg['z_max'] = self._get_z_max()
        self._cfg['z_min'] = self._get_z_min()

    def _get_z_max(self):
        return round(self._lim['z'][1] * self.motor_step, 1)

    def _get_z_min(self):
        return round(self._lim['z'][0] * self.motor_step, 1)

    @property
    def settings(self):
        """Все параметры: значение, допустимые пределы, смысл. Изменить: m.<имя> = значение."""
        self._sync_limits()
        rows = []
        for k, s in type(self).__dict__.items():
            if isinstance(s, Setting):
                rows.append((k, fmt(self._cfg[k]) + (' ' + s.unit if s.unit else ''), s.allowed(self), s.doc))
        wv, wr = max(len(r[1]) for r in rows), max(len(r[2]) for r in rows)
        r = Result("Settings" + " " * (max(len(x[0]) for x in rows) - 3) +
                   f"{'Value':<{wv}}   {'Allowed':<{wr}}   Meaning")
        for k, v, a, d in rows:
            r[k] = f"{v:<{wv}}   {a:<{wr}}   {d}"
        return r

    # ---------------- подключение ----------------
    def _take_lock(self, port):
        try:
            self._lock = ControlLock(port)
        except OSError:
            own = ControlLock.owner(port) or {}
            raise SafetyError('E106', f"Port {port}: PID {own.get('pid', '?')} since {own.get('since', '?')}, "
                                      f"{own.get('program', '')}. Close it there with m.close(), "
                                      "or stop it from a command line: python -m microcontrol stop") from None
        L.event('LCK-001', f"Control of {port} taken by PID {os.getpid()}.")

    def _handshake(self):
        """Как штатное ПО при подключении: серийный номер, модель, лазер, калибровка, тип прибора (только чтение)."""
        info = {}
        for attempt in range(3):           # контроллер мог остаться «посреди кадра» после прошлого сбоя:
            try:                           # повторные запросы выравнивают его чтение по заголовку AA 55
                info['GET_SN'] = self._cmd(P.GET_SN)
                break
            except CommError as e:
                L.event('CON-003', f"No answer to GET_SN, attempt {attempt + 1} of 3 ({e.code}).", 30)
                self._ser.reset_input_buffer()
                time.sleep(0.3)
        else:
            raise CommError('E301', "The controller does not answer. Run: python -m microcontrol probe. "
                                    "If the Optosky software is open, close it; replug the controller USB.")
        for cmd, data in P.HANDSHAKE[1:]:              # серийный номер уже прочитан — один раз, как штатное ПО
            try:
                info[P.NAMES[cmd]] = self._cmd(cmd, data, timeout=5.0)   # как штатное ПО: ждать до 5 с
            except CommError as e:
                if cmd != P.GET_CALIBRATION:           # калибровка нам не нужна — только повторяем порядок штатного ПО
                    raise
                info[P.NAMES[cmd]] = b""
                L.event('CON-003', f"No answer to the calibration request ({e.code}); continuing.", 30)
        txt = lambda d: d.decode('ascii', 'replace').strip('\x00 ') if d and all(32 <= b < 127 or b == 0 for b in d) else d.hex(' ').upper()
        sn, pn = txt(info['GET_SN']), txt(info['GET_PN'])
        if not sn and not pn:
            raise SafetyError('E107', "Serial number and model are empty.")
        dt = info['GET_TYPE']
        kind = next((P.DEVICE_TYPES[b] for b in dt if b in P.DEVICE_TYPES), dt.hex(' ').upper() or '-')
        ml = int.from_bytes(info['GET_MAX_LASER'][-2:], 'big') if len(info['GET_MAX_LASER']) >= 2 else None
        self._ident = (pn, sn)
        ml_txt = f"{ml} mW" if ml and ml <= 2000 else (f"not set ({info['GET_MAX_LASER'].hex(' ').upper()})" if ml else "-")
        self.device = Result("Device", Model=pn, Serial=sn, Type=kind, **{"Laser maximum": ml_txt})
        L.event('CON-003', f"Model {pn}, serial {sn}, type {kind}, laser maximum {ml_txt}. "
                           f"Calibration block {len(info['GET_CALIB'])} bytes.")

    def _open_port(self, port, force):
        if port == 'sim':
            self._take_lock('sim')
            from .sim import SimWorld, SimPort
            self._world = SimWorld()
            self._ser = SimPort(self._world)
            L.event('CON-001', "Simulator connected.")
            return
        import serial
        from serial.tools import list_ports
        found = list_ports.comports()
        if port == 'auto':
            atr = [p.device for p in found if (p.vid, p.pid) == (VID, PID)]
            if len(atr) != 1:
                raise SafetyError('E101', f"Found {atr or 'none'}. Give the port: Microscope('COM3').")
            port = atr[0]
            self.port = port
        elif not force and not any(p.device == port and (p.vid, p.pid) == (VID, PID) for p in found):
            raise SafetyError('E103', f"{port}. If you are sure, use Microscope('{port}', force=True).")
        self._take_lock(port)
        _step_msg(f"  Port         {port} opening ...")
        try:
            self._ser = _timed(lambda: _open_serial(port), 10,
                               SafetyError('E108', f"{port}: no answer from the driver in 10 s. Unplug and plug in "
                                                   "the controller USB, restart the kernel."))
        except serial.SerialException as e:
            raise SafetyError('E102', f"{port}. Close the Optosky software. ({e})") from e
        self.port = port
        L.event('CON-001', f"Port {port}, 115200 baud.")

    def close(self):
        """Отключиться: лазер и свет выключаются, порт и камера закрываются."""
        if self._ser is not None and self._ser.is_open and not self._stopped:
            for name, f in (('laser', lambda: self._raw(P.LASER, b"\x00\x00")),          # как штатное ПО:
                            ('motors', lambda: self._motors_on and self._raw(P.TURN_MOTOR, b"\x00")),  # лазер, моторы,
                            ('light', lambda: self._raw(P.LIGHT, b"\x00\x00")),          # свет,
                            ('cooling', lambda: self._cooling and self._raw(P.SET_TEC, P.tec_data(P.WARM_C)))):  # охлаждение
                try:
                    f()
                except Exception as e:
                    warnings.warn(f"Could not switch the {name} off: {e}")
            self._remember()
            self._ser.close()
            L.event('CON-002', f"Disconnected at {self.position}.")
            L.event('SYS-002')
            print(Result("Disconnected", Position=self.position))
        if self.cam:
            self.cam.close()
            self.cam = None
        self._ser = None
        if self._lock:
            self._lock.release()
            self._lock = None
            L.event('LCK-001', "Control released.")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __repr__(self):
        r = Result("Microscope", Port=self.port if self._ser else f"{self.port} (closed)",
                   Objective=f"{self.objective}x", Position=self.position,
                   **({"Real position": self.real_position} if any(self._off.values()) else {}),
                   **{"Frame size": f"{self.frame_size[0]:.0f} x {self.frame_size[1]:.0f} um"},
                   Light=self.light, Laser=f"{self.laser:g} mW", Area=self.area,
                   **{"Focus plane": "On" if self._plane is not None else "Off"})
        if self._mem is not None:
            r["Home"] = f"set {self._mem.home['time']}" if self._mem.home else "not set (m.set_home())"
        if self._uncertain:
            r["Uncertain axes"] = ", ".join(a.upper() for a in sorted(self._uncertain))
        return repr(r)

    def _repr_pretty_(self, p, cycle):
        p.text(repr(self))

    def info(self):
        """Модель, серийный номер, тип прибора (прочитаны при подключении)."""
        return self.device

    # ---------------- низкий уровень ----------------
    def _u2p(self, um):
        return int(round(um / self._cfg.get('motor_step', UM_PER_STEP)))

    def _fmt(self):
        """Положение для журнала: рабочее и (если есть смещение) настоящее."""
        p = self.position
        if any(self._off.values()):
            return f"{p} (real {self.real_position})"
        return str(p)

    def _check_stop(self):
        """Аварийная остановка: выключить лазер и свет и прекратить работу."""
        if self._stopped or stop_requested_after(self._t_connect):
            if not self._stopped:
                self._stopped = True
                for cmd in (P.LASER, P.LIGHT):
                    try:
                        self._raw(cmd, b"\x00\x00")
                    except Exception:
                        pass
                self._cfg['laser'], self._cfg['light'] = 0.0, 0
                L.event('STP-001', f"Emergency stop at {self._fmt()}. Laser and light off. Port released.", 40)
                try:                                   # отпустить порт сразу: аварийной команде не нужно
                    self._ser.close()                  # завершать ноутбук принудительно
                except Exception:
                    pass
                if self._lock:
                    self._lock.release()
                    self._lock = None
            raise StopError('E110', "Stopped by 'python -m microcontrol stop'. Laser and light are off. "
                                    "Close with m.close() and connect again.")

    def _sleep(self, s):
        """Пауза, которая прерывается аварийной остановкой."""
        t_end = time.monotonic() + s
        while True:
            self._check_stop()
            left = t_end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(0.05, left))

    def _cmd(self, cmd, data=b"", timeout=None):
        """Отправить разрешённую команду и вернуть данные ответа (с проверкой аварийной остановки)."""
        if cmd not in P.ALLOWED:
            raise SafetyError('E305', f"0x{cmd:02X}.")
        if cmd == P.SET_TEC and data not in (P.tec_data(P.COOL_C), P.tec_data(P.WARM_C)):
            raise SafetyError('E305', f"Cooling only {P.COOL_C} or {P.WARM_C} C, as the Optosky software.")
        if self._ser is None:
            raise SafetyError('E104')
        self._check_stop()
        return self._raw(cmd, data, timeout)

    def _raw(self, cmd, data=b"", timeout=None, _retry=True):
        """Одна команда и ответ. Если порт пропал (USB переподключился или был усыплён Windows),
        порт открывается заново; безопасная команда повторяется, движение — нет (положение под вопросом)."""
        import serial
        try:
            return self._raw1(cmd, data, timeout)
        except (serial.SerialException, OSError) as e:
            if not _retry or self._world is not None:
                raise CommError('E307', f"{P.NAMES.get(cmd, hex(cmd))}: {e}") from None
            self._reopen(e)
            axis = {c: a for a, c in P.MOTOR.items()}.get(cmd)
            if axis:                                   # движение могло выполниться, а могло и нет — не повторяем
                self._uncertain.add(axis)
                raise CommError('E307', f"The connection was lost during a move on {axis.upper()} and restored. "
                                        "The move may or may not have been done: check the stage, then "
                                        "m.confirm_position().") from None
            return self._raw(cmd, data, timeout, _retry=False)

    def _reopen(self, why):
        """Порт пропал: закрыть и открыть заново (до 3 попыток), восстановить свет и охлаждение."""
        L.event('CON-006', f"Connection lost ({why}). Reopening {self.port}.", 30)
        try:
            self._ser.close()
        except Exception:
            pass
        last = why
        for attempt in range(3):
            time.sleep(1.0)
            try:
                self._ser = _timed(lambda: _open_serial(self.port), 10, CommError('E307', "the driver does not answer"))
                break
            except Exception as e:
                last = e
        else:
            raise CommError('E307', f"Could not reopen {self.port} ({last}). Unplug and plug in the controller USB, "
                                    "then reconnect.") from None
        self._motors_on = False                        # если контроллер перезапустился — моторы снова включатся
        L.event('CON-006', f"Port {self.port} reopened.")
        try:
            self._raw1(P.LIGHT, int(self._cfg.get('light', 0)).to_bytes(2, 'big'))
            if self._cooling:
                self._raw1(P.SET_TEC, P.tec_data(P.COOL_C))
        except Exception:
            pass

    def _keepalive(self, every=2.0):
        """Простой без обмена: безобидный запрос на чтение, чтобы Windows не усыпляла USB-порт."""
        if self._ser is None or self._world is not None or time.monotonic() - self._last < every:
            return
        self._quiet = True                             # не засорять журнал запросами «на связи»
        try:
            self._raw(P.GET_DEVICE_TYPE)
        finally:
            self._quiet = False

    def _raw1(self, cmd, data=b"", timeout=None):
        wait = self._last + (0 if self._fast else MIN_INTERVAL_S) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        frame = P.pack(cmd, data)
        try:
            self._ser.reset_input_buffer()
            import serial
            try:
                self._ser.write(frame)
            except serial.SerialTimeoutException as e:  # драйвер не принимает данные (write_timeout)
                raise CommError('E301', f"Could not send {P.NAMES[cmd]}: {e}. Unplug and plug in the controller USB.") from None
            if not self._quiet:
                L.debug('HW-TX', f"{P.NAMES[cmd]:<10} {frame.hex(' ').upper()}")
            rcmd, rdata, raw = P.read_frame(self._ser, timeout or REPLY_TIMEOUT_S)
            if not self._quiet:
                L.debug('HW-RX', f"{P.NAMES[cmd]:<10} {raw.hex(' ').upper()}")
        finally:
            self._last = time.monotonic()
        if rcmd != cmd:
            raise CommError('E304', f"Sent 0x{cmd:02X}, got 0x{rcmd:02X}.")
        return rdata

    def _step(self, ax, n, stop_at_limit=True):
        """Один кусок движения с проверкой концевика и паузой. Возвращает True на концевике."""
        pos = (n > 0) != (ax == 'z')               # Z: 0xFF = вверх
        if self._cfg['reverse_' + ax]:
            pos = not pos
        data = bytes([0x00 if pos else 0xFF]) + abs(n).to_bytes(2, 'big')
        self._check_stop()
        if not self._motors_on:                    # как штатное ПО (TurnOnMotorLegacy): моторы включаются
            self._cmd(P.TURN_MOTOR, b"\x01")       # перед первым движением по любой оси
            self._motors_on = True
            L.event('MOV-011', "Motors switched on (as the Optosky software before the first move).")
        self._remember(moving=True)                # если программу убьют сейчас — положение будет «под вопросом»
        t_end = time.monotonic() + 5.0
        while True:
            try:
                st = self._cmd(P.MOTOR[ax], data)
            except BaseException:
                self._p[ax] += n
                self._uncertain.add(ax)            # неизвестно, доехал ли
                raise
            # «ещё едет» (04): команда НЕ выполнена — прошлое движение не кончилось; подождать и повторить
            if st and st[0] == P.MOTOR_MOVING and time.monotonic() < t_end:
                L.debug('MOV-013', f"{ax.upper()}: still moving, the command is sent again.")
                self._sleep(0.03)
                continue
            break
        self._p[ax] += n
        L.debug('MOV-006', f"{ax.upper()} {n:+d} steps, now {self._p[ax]}. Status {st.hex() or '-'}.")
        # контроллер отвечает до окончания движения: ждём путь / скорость + успокоение
        self._sleep(abs(n) * self.motor_step / self.motor_speed + self.move_pause)
        self._remember()
        code = st[0] if st else 0
        if code in (P.MOTOR_FAIL, P.MOTOR_MOVING) or code not in P.MOTOR_STATUS:
            self._uncertain.add(ax)                # команда не выполнена или выполнена не так — не повторяем
            raise CommError('E306', f"Axis {ax.upper()}: controller says {P.MOTOR_STATUS.get(code, f'code {code}')}. "
                                    + ("It was still moving: lower m.motor_speed. " if code == P.MOTOR_MOVING else "")
                                    + "Check the stage, then m.confirm_position().")
        # Код концевика говорит, КАКОЙ концевик (1 — сторона +, 2 — сторона −), а не куда ехали.
        # Если ехали от него (столик ещё стоял на концевике в начале движения) — это не упор: не блокируем.
        side = {P.MOTOR_LIMIT_A: 1, P.MOTOR_LIMIT_B: -1}.get(code, 0)
        hit = side != 0 and side == (1 if n > 0 else -1)
        if side and not hit:
            L.event('MOV-012', f"{ax.upper()}: controller reports the {'+' if side > 0 else '-'} limit switch while "
                               f"moving {'+' if n > 0 else '-'} (leaving it). Not blocked.", 30)
        if hit:
            self._blocked[ax] = side
            self._uncertain.add(ax)
            if stop_at_limit:
                raise LimitError('E204', f"Axis {ax.upper()}, direction {'+' if side > 0 else '-'}: the controller "
                                         f"reports the limit switch (code {code}). Move back the other way.")
        elif self._blocked.get(ax) == -(1 if n > 0 else -1):
            self._blocked.pop(ax)
        return hit

    def _check(self, target):
        """Проверить цель (шаги мотора) на пределы, ничего не двигая."""
        s = self.motor_step
        for ax, t in target.items():
            lo, hi = self._lim[ax]
            if ax == 'z' and t > hi:
                raise SafetyError('E203', f"Z {t * s:.0f} um, the ceiling is {hi * s:.0f} um.")
            if not lo <= t <= hi:
                raise SafetyError('E202', f"{ax.upper()} {t * s:.0f} um, allowed {lo * s:.0f} .. {hi * s:.0f} um.")
            d = t - self._p[ax]
            if d and self._blocked.get(ax) == (1 if d > 0 else -1):
                raise LimitError('E205', f"Axis {ax.upper()}, direction {'+' if d > 0 else '-'}: the last move this way "
                                         "stopped at a limit switch. Move the other way, or m.confirm_position() "
                                         "if the stage is not at the switch.")

    def _go(self, target):
        """Поехать в target (шаги): проверки, затем Z вниз -> X -> Y -> Z вверх."""
        self._check(target)
        z = target.get('z', self._p['z'])
        for ax in (['z'] if z < self._p['z'] else []) + ['x', 'y'] + (['z'] if z > self._p['z'] else []):
            d = target.get(ax, self._p[ax]) - self._p[ax]
            while d:
                n = max(-CHUNK_STEPS, min(CHUNK_STEPS, d))
                self._step(ax, n)
                d -= n

    # ---------------- движение ----------------
    @property
    def real_position(self):
        """Настоящее положение столика (X, Y, Z), мкм — от точки подключения (или концевиков после home)."""
        s = self._cfg.get('motor_step', UM_PER_STEP)
        return Position(round(self._p[a] * s, 1) for a in AXES)

    @property
    def position(self):
        """Положение столика (X, Y, Z), мкм, в рабочих координатах (с учётом offset). m.position.x — одна ось."""
        mp = self.real_position
        return Position(round(v - self._off[a], 1) for a, v in zip(AXES, mp))

    def move(self, dx=0, dy=0, dz=0):
        """Сдвинуть столик на dx, dy, dz мкм (как G91 + G0). +dz — вверх, к объективу.
        Один шаг не больше max_move по X/Y, max_move_z вверх (к объективу), max_move_down вниз; дальше — goto()."""
        for a, v in dict(x=dx, y=dy, z=dz).items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise SafetyError('E206', f"d{a.upper()} = {v!r}.")
        lim = dict(x=self.max_move, y=self.max_move, z=self.max_move_z if dz > 0 else self.max_move_down)
        for a, v in dict(x=dx, y=dy, z=dz).items():
            if abs(v) > lim[a]:
                raise SafetyError('E201', f"d{a.upper()} {v:g} um, the limit is {lim[a]:g} um"
                                          f"{' (m.max_move_z, up towards the objective)' if a == 'z' and v > 0 else ''}. "
                                          "Use smaller steps or goto().")
        if not (dx or dy or dz):
            raise SafetyError('E213', "Give dx, dy or dz.")
        steps = {a: self._u2p(v) for a, v in dict(x=dx, y=dy, z=dz).items()}
        for a, v in dict(x=dx, y=dy, z=dz).items():
            if v and not steps[a]:
                raise SafetyError('E214', f"d{a.upper()} {v:g} um. One motor step is {self.motor_step:.2f} um.")
        self._go({a: self._p[a] + n for a, n in steps.items()})
        real = {a: n * self.motor_step for a, n in steps.items()}
        L.event('MOV-001', f"Moved by dX {real['x']:+.1f}, dY {real['y']:+.1f}, dZ {real['z']:+.1f} um. Now {self._fmt()}.")
        r = Result("Moved", Position=self.position)
        asked = dict(x=dx, y=dy, z=dz)
        if any(abs(real[a] - asked[a]) > 0.05 for a in AXES):
            r['Note'] = ("Rounded to whole motor steps: moved by " +
                         ", ".join(f"d{a.upper()} {real[a]:+.1f}" for a in AXES if asked[a]) + " um.")
        return r

    def goto(self, x=None, y=None, z=None):
        """Поехать в точку x, y, z, мкм (как G90 + G0). Не указанные оси остаются на месте.
        Если включена плоскость фокуса (G29) и z не указан, Z берётся с плоскости.
        Переезд длиннее max_jump или подъём Z больше max_move_z отклоняется как подозрительный (E210)."""
        if x is None and y is None and z is None:
            raise SafetyError('E213', "Give x, y or z.")
        t = {}
        for a, v in dict(x=x, y=y, z=z).items():
            if v is not None:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise SafetyError('E206', f"{a.upper()} = {v!r}.")
                t[a] = self._u2p(v + self._off[a])               # рабочие -> машинные
        self._need_certain()
        if z is None and self._plane is not None and ('x' in t or 'y' in t):
            X = t.get('x', self._p['x']) * self.motor_step
            Y = t.get('y', self._p['y']) * self.motor_step
            lo, hi = self._lim['z']
            t['z'] = min(hi, max(lo, self._u2p(self._plane @ (1, X, Y))))
        self._check(t)                                  # сначала пределы и потолок, потом «подозрительность»
        self._suspicious(t)
        self._go(t)
        L.event('MOV-002', f"Arrived at {self._fmt()}.")
        return Result("Arrived", Position=self.position)

    def offset(self, x=0, y=0, z=0):
        """Сдвинуть ноль для удобства (как G92): текущая точка получает рабочие координаты (x, y, z),
        по умолчанию (0, 0, 0). Настоящие координаты (m.real_position) и пределы не меняются.
        Отменить: m.offset_clear()."""
        for a, v in dict(x=x, y=y, z=z).items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise SafetyError('E206', f"{a.upper()} = {v!r}.")
        old = dict(self._off)
        mp = self.real_position
        self._off = {a: round(m - v, 3) for a, m, v in zip(AXES, mp, (x, y, z))}
        self._shift_area(old)
        L.event('MOV-003', f"Offset X {self._off['x']:.1f}, Y {self._off['y']:.1f}, Z {self._off['z']:.1f} um. "
                           f"Real position {mp}.")
        return Result("Offset set", Position=self.position, **{"Real position": mp})

    def offset_clear(self):
        """Убрать смещение нуля: рабочие координаты снова равны настоящим."""
        old = dict(self._off)
        self._off = dict.fromkeys(AXES, 0.0)
        self._shift_area(old)
        L.event('MOV-003', "Offset cleared.")
        return Result("Offset cleared", Position=self.position)

    def _shift_area(self, old):
        """Зона хранится в рабочих координатах: при смене нуля она пересчитывается и остаётся на месте."""
        if self.area is not None:
            dx, dy = old['x'] - self._off['x'], old['y'] - self._off['y']
            a = self.area
            a.x0, a.x1, a.y0, a.y1 = a.x0 + dx, a.x1 + dx, a.y0 + dy, a.y1 + dy

    def _need_certain(self):
        if self._uncertain:
            raise SafetyError('E211', f"Axis {', '.join(a.upper() for a in sorted(self._uncertain))} may be off "
                                      "after a limit switch, a lost reply or since the last session. Small moves with m.move() still work. "
                                      "Check the stage, then m.confirm_position().")

    @property
    def crosses(self):
        """Решётка крестов, отмеченная в ручном режиме (клавиша P): шаг, поворот, крайние кресты."""
        cr = getattr(self, '_crosses', None)
        if cr is None or not cr.marks:
            return Result("Crosses", Marks="none: in m.manual() put the red lines on a cross and press P")
        d = cr.to_dict()
        r = Result("Crosses", Marks=len(cr.marks))
        if 'pitch_um' in d:
            i0, i1, j0, j1 = d['extent']
            r.update(**{"Pitch, um": f"{d['pitch_um'][0]:.2f} x {d['pitch_um'][1]:.2f}",
                        "Turn, deg": f"{d['turn_deg']:+.3f}",
                        "Skew, deg": "-" if d['skew_deg'] is None else f"{d['skew_deg']:+.3f}",
                        "Cells": f"{i1 - i0} x {j1 - j0} (crosses {i1 - i0 + 1} x {j1 - j0 + 1})",
                        "Off the red grid, um": "-" if cr.residual() is None else f"{cr.residual():.2f}"})
        return r

    def crosses_clear(self):
        """Удалить отметки крестов."""
        if getattr(self, '_crosses', None) is not None:
            self._crosses.clear()

    def set_limits(self, x=None, y=None, z=None):
        """Пределы хода столика, мкм, в рабочих координатах (как m.position): пара (от, до) для каждой оси.
        m.set_limits(z=(-50000, 50000)); m.set_limits(x=(-20000, 20000), y=(-20000, 20000)).
        Текущее положение должно быть внутри; не дальше 100 мм в каждую сторону."""
        if x is None and y is None and z is None:
            raise SafetyError('E213', "Give at least one axis: m.set_limits(z=(-50000, 50000)).")
        new = {}
        for a, v in (('x', x), ('y', y), ('z', z)):
            if v is None:
                continue
            try:
                lo, hi = (float(v[0]), float(v[1]))
            except (TypeError, ValueError, IndexError):
                raise SafetyError('E208', f"{a} = {v!r}. Use a pair: {a}=(lo, hi) in um.") from None
            cur = getattr(self.position, a)
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
                raise SafetyError('E208', f"{a}=({lo:g}, {hi:g}): lo must be below hi.")
            if not lo <= cur <= hi:
                raise SafetyError('E208', f"{a}=({lo:g}, {hi:g}): the stage is now at {cur:.1f} um, outside.")
            if max(abs(lo - cur), abs(hi - cur)) > 100000:
                raise SafetyError('E208', f"{a}=({lo:g}, {hi:g}): more than 100 mm from the current position.")
            new[a] = (self._u2p(lo + self._off[a]), self._u2p(hi + self._off[a]))
        self._lim.update(new)
        self._sync_limits()
        L.event('SET-001', "Limits " + ", ".join(f"{a.upper()} {lo * self.motor_step - self._off[a]:.0f} .. "
                                                 f"{hi * self.motor_step - self._off[a]:.0f} um" for a, (lo, hi) in new.items()) + ".")
        return self.limits

    @property
    def limits(self):
        """Пределы хода, мкм, в рабочих координатах."""
        s = self.motor_step
        return Result("Limits", **{a.upper(): f"{lo * s - self._off[a]:.0f} .. {hi * s - self._off[a]:.0f} um"
                                   for a, (lo, hi) in self._lim.items()})

    def confirm_position(self):
        """Подтвердить, что положение столика верное (после концевика или сбоя связи)."""
        axes = ", ".join(a.upper() for a in sorted(self._uncertain)) or "-"
        self._uncertain.clear()
        self._blocked.clear()                      # концевик тоже снят: прибор сам не даст заехать за него
        L.event('MOV-008', f"Position confirmed by the user ({axes}) at {self._fmt()}.")
        return Result("Position confirmed", Position=self.position)

    # ---------------- дом и память прибора ----------------
    def _remember(self, moving=False):
        """Записать положение в память прибора (после каждого куска движения)."""
        if self._mem is None:
            return
        if self._world is not None:
            self._mem.data['sim_true'] = self._world.p          # симулятор: где столик на самом деле
        try:
            self._mem.position(self._p, moving)
        except OSError as e:
            warnings.warn(f"Could not save the position: {e}")

    def _restore(self):
        """При подключении: если дом задан — взять положение из прошлого сеанса (неподтверждённым)."""
        mem = self._mem
        if self._world is not None and 'sim_true' in mem.data:       # симулятор помнит, где стоит столик
            self._world._p = {a: float(v) for a, v in mem.data['sim_true'].items()}
        if not mem.home or not mem.last:
            return None
        self._p = {a: int(mem.last['p'][a]) for a in AXES}
        self._uncertain = set(AXES)
        broken = mem.last.get('moving')
        L.event('CON-004', f"Position {self.position} restored from {mem.last['time']}"
                           + (" (that session ended during a move)" if broken else "") + ". Not confirmed yet.",
                30 if broken else 20)
        return ("The last session ended DURING A MOVE: the position may be off. Go home by eye, then "
                "m.set_home(replace=True), or check with m.go_home() after m.confirm_position()." if broken else
                "If nobody moved the stage since then (by hand or in the Optosky software): m.confirm_position(). "
                "Check: m.go_home() compares the picture with the home photo.")

    def set_home(self, replace=False):
        """Дом этого микроскопа: текущая точка становится настоящим (0, 0, 0) — один раз и навсегда.
        Запоминается по модели и серийному номеру вместе с фото этого места; при следующих подключениях
        координаты продолжаются от того же дома. Задать заново: m.set_home(replace=True)."""
        mem = self._mem
        if mem.home and not replace:
            raise SafetyError('E215', f"Home of this microscope was set on {mem.home['time']} and is kept, "
                                      "so coordinates stay the same between sessions. "
                                      "Go home: m.go_home(). Set it again: m.set_home(replace=True).")
        s = self.motor_step
        d = dict(self._p)
        for a in AXES:                                     # всё сдвигается так, чтобы здесь стал 0
            lo, hi = self._lim[a]
            self._lim[a] = (lo - d[a], hi - d[a])
            self._p[a] = 0
        if self.area is not None:                          # зона остаётся на том же месте образца
            a_ = self.area
            dx, dy = self._off['x'] - d['x'] * s, self._off['y'] - d['y'] * s
            a_.x0, a_.x1, a_.y0, a_.y1 = a_.x0 + dx, a_.x1 + dx, a_.y0 + dy, a_.y1 + dy
        self._off = dict.fromkeys(AXES, 0.0)
        self._plane = None
        self._uncertain.clear()                            # здесь положение известно точно: это 0
        photo = None
        if self.cam:
            photo = imwrite(mem.photo_path, self.image())
        mem.set_home(objective=self.objective, photo=photo, light=self.light, exposure=self.exposure)
        self._remember()
        self._sync_limits()
        L.event('MOV-009', f"Home set{' again' if replace else ''} for {mem.data['model']} {mem.data['serial']}. "
                           f"File {mem.path}.")
        return Result("Home set", Position=self.position, Microscope=f"{mem.data['model']}, serial {mem.data['serial']}",
                      Photo=photo or "-", File=mem.path)

    def go_home(self, z=False, correct=False):
        """Поехать домой: X и Y в 0 (настоящие координаты). z=True — и Z в 0 (подъём — мелкими шагами,
        не выше потолка). Затем картинка сравнивается с фото дома: так видно, не сбились ли координаты.
        correct=True — если картинка сдвинута, исправить координаты по ней и доехать точно."""
        mem = self._mem
        if not mem.home:
            raise SafetyError('E216', "Go to the place you want as home and run m.set_home().")
        self._need_certain()
        self._travel(x=-self._off['x'], y=-self._off['y'])
        if z:
            while self._p['z'] != 0:
                d = -self._p['z']
                d = min(d, self._u2p(self.max_move_z)) if d > 0 else max(d, -self._u2p(self.max_move_down))
                self._go({'z': self._p['z'] + d})
        r = Result("At home", Position=self.position)
        shift = self._compare_home()
        if shift is None:
            r['Picture'] = "not compared (no camera, no home photo, or another objective)"
        elif shift == 'lost':
            r['Picture'] = "does not match the home photo (another sample, light or a large shift)"
        else:
            e = shift                                       # где столик на самом деле, шаги от дома
            um = {a: e[a] * self.motor_step for a in e}
            if max(abs(v) for v in e.values()) <= 1:
                r['Picture'] = "matches the home photo"
            elif correct:
                for a in e:
                    self._p[a] = e[a]
                self._go({'x': 0, 'y': 0})
                r['Picture'] = f"was shifted by X {um['x']:+.1f}, Y {um['y']:+.1f} um; coordinates corrected"
                r['Position'] = self.position
            else:
                r['Picture'] = f"shifted from the home photo by X {um['x']:+.1f}, Y {um['y']:+.1f} um"
                r['Note'] = "If the sample did not move, fix the coordinates: m.go_home(correct=True)."
        L.event('MOV-010', f"At home {self._fmt()}. Picture: {r['Picture']}.")
        return r

    def _compare_home(self):
        """Сдвиг текущего кадра относительно фото дома -> где столик на самом деле (шаги от дома)."""
        h = self._mem.home
        if not self.cam or not h.get('photo') or not os.path.exists(h['photo']) or h.get('objective') != self.objective:
            return None
        g = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) if im.ndim == 3 else im.astype(np.float32)
        a, b = g(imread(h['photo'])), g(self.image())
        if a.shape != b.shape:
            return None
        win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
        (sx, sy), resp = cv2.phaseCorrelate(a, b, win)
        if resp < 0.1:
            return 'lost'
        e = np.linalg.solve(self._pixels_per_step(), np.array([sx, sy]))
        return {'x': int(round(e[0])), 'y': int(round(e[1]))}

    def _suspicious(self, t):
        """Подозрительно большой переезд за одну команду — ошибка до движения."""
        s = self.motor_step
        dxy = math.hypot(t.get('x', self._p['x']) - self._p['x'], t.get('y', self._p['y']) - self._p['y']) * s
        dz = (t.get('z', self._p['z']) - self._p['z']) * s
        if dxy > self.max_jump + 1e-6:
            raise SafetyError('E210', f"{dxy:.0f} um in X/Y at once, the limit is {self.max_jump:g} um (m.max_jump). "
                                      "Check the numbers, go in steps, or raise m.max_jump.")
        if dz > self.max_move_z + 1e-6:
            raise SafetyError('E210', f"Stage up by {dz:.0f} um at once, the limit is {self.max_move_z:g} um (m.max_move_z).")
        if -dz > self.max_jump + 1e-6:
            raise SafetyError('E210', f"Stage down by {-dz:.0f} um at once, the limit is {self.max_jump:g} um (m.max_jump).")

    def _travel(self, x=None, y=None, z=None):
        """Плановый переезд (скан, G29) в рабочих координатах: длинный путь делится на отрезки ≤ max_jump."""
        tx = self._p['x'] * self.motor_step if x is None else x + self._off['x']
        ty = self._p['y'] * self.motor_step if y is None else y + self._off['y']
        cx, cy = self._p['x'] * self.motor_step, self._p['y'] * self.motor_step
        n = max(1, math.ceil(math.hypot(tx - cx, ty - cy) / (self.max_jump * 0.95)))
        for i in range(1, n + 1):
            k = i / n
            self.goto(x=cx + (tx - cx) * k - self._off['x'], y=cy + (ty - cy) * k - self._off['y'],
                      z=z if i == n else None)

    def find_limits(self, x=True, y=True, z=False):
        """Найти концевики (как G28): столик опускается на 2 мм, едет в минус до концевика.
        Если дом не задан — там ставится 0. Если задан — координаты остаются от дома, концевик становится нижним пределом.
        Заблокировано, пока не подключиться с allow_homing=True: сигнал концевиков на приборе ещё не проверен."""
        if not self.allow_homing:
            raise SafetyError('E209', "Finding the limit switches is locked. Connect with Microscope(..., allow_homing=True). "
                                      "The limit switch signal on this device is not verified yet.")
        keep = bool(self._mem and self._mem.home)
        if keep:
            self._need_certain()
        axes = [a for a, f in (('z', z), ('x', x), ('y', y)) if f]
        if x or y:
            self._go({'z': max(self._lim['z'][0], self._p['z'] - self._u2p(2000))})
        for ax in axes:
            maxp, moved, hit = self._u2p(80000), 0, False
            for chunk in (180, 9):                     # быстро до концевика, потом точно
                while not hit and moved < maxp:
                    hit = self._step(ax, -chunk, stop_at_limit=False)
                    moved += chunk
                if chunk == 180:
                    if not hit:
                        break
                    self._step(ax, 180); moved -= 180; hit = False
            if not hit:
                raise LimitError('E209', f"Axis {ax.upper()}: nothing within 80 mm.")
            d = self._p[ax]
            if keep:                                   # координаты от дома, концевик — нижний предел
                self._lim[ax] = (d, max(self._lim[ax][1], d + 1))
            else:
                # ход отсчитывается от концевика: X/Y — от 0 до 2 × xy_limit, Z — от 0 до прежнего потолка
                self._lim[ax] = (0, 2 * self._u2p(self.xy_limit)) if ax != 'z' else (0, self._lim['z'][1] - d)
                self._p[ax] = 0
            self._blocked[ax] = -1
            self._uncertain.discard(ax)
        if not keep:
            self.area, self._plane = None, None
            self._off = dict.fromkeys(AXES, 0.0)
        self._remember()
        L.event('MOV-004', f"Limit switches found: {', '.join(a.upper() for a in axes)}. Now {self.position}.")
        return Result("Limits found", Axes=", ".join(a.upper() for a in axes), Position=self.position,
                      Note="Coordinates are from home; the switches are the lower limits." if keep
                      else "Zero is at the negative limit switches.")

    def _screen_axes(self):
        """Какая ось столика и с каким знаком сдвигает поле зрения вправо (h) и вниз (v) на экране."""
        K = self._pixels_per_step()                 # сдвиг картинки (u, v) на шаг (x, y)
        ax_h = 'x' if K[0, 0] else 'y'
        ax_v = 'y' if K[1, 1] else 'x'
        sign = lambda k: 1 if k < 0 else -1          # картинка влево = поле зрения вправо
        return {'h': (ax_h, sign(K[0, 0 if ax_h == 'x' else 1])), 'v': (ax_v, sign(K[1, 1 if ax_v == 'y' else 0]))}

    def _frame_step(self, fraction=1.0):
        """Шаг «в кадр» (или долю кадра), мкм, округлённый вниз до шага мотора — без щелей между кадрами."""
        s = self.motor_step
        w, h = self.frame_size
        return {'x': max(1, math.floor(w * fraction / s + 1e-9)) * s,
                'y': max(1, math.floor(h * fraction / s + 1e-9)) * s}

    def _screen_dir(self):
        """Знак сдвига столика по X и Y, при котором поле зрения уходит вправо / вниз на экране (по своей оси)."""
        a = self._screen_axes()
        return {a['h'][0]: a['h'][1], a['v'][0]: a['v'][1]}

    def _pixels_per_step(self):
        """Матрица K: сдвиг картинки (u — вправо, v — вниз, пиксели) на 1 шаг мотора по (x, y).
        Камера ровная: K диагональная; повёрнута на 90° (swap_xy) — X двигает картинку по вертикали, Y — по горизонтали."""
        k = self.motor_step / self.pixel_size
        kx, ky = (k if self.mirror_x else -k), (k if self.mirror_y else -k)
        if self.turn_180:                          # перевёрнутый кадр: оба сдвига меняют знак
            kx, ky = -kx, -ky
        return np.array([[0.0, ky], [kx, 0.0]]) if self.swap_xy else np.diag([kx, ky])

    def check_orientation(self, fix=True, distance=None):
        """Проверить камерой, куда идёт картинка при движении X и Y, и исправить swap_xy, mirror_x, mirror_y.
        Столик сдвигается на ~20% кадра по X и по Y и возвращается. От этого зависит сшивка: при неверном знаке
        кадры встают в обратном порядке. Делается сама в начале каждого скана (scan_check = True)."""
        self._need_cam()
        g = lambda: cv2.cvtColor(self.image(), cv2.COLOR_BGR2GRAY).astype(np.float32)
        fw, fh = self.frame_size
        D = distance or max(self.motor_step * 3, 0.2 * min(fw, fh))
        win = None
        meas = {}
        for ax in 'xy':
            a = g()
            self.move(**{'d' + ax: D})
            b = g()
            self.move(**{'d' + ax: -D})
            if win is None:
                win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
            (sx, sy), resp = cv2.phaseCorrelate(a, b, win)
            meas[ax] = (np.array([sx, sy]), resp)
        n = round(D / self.motor_step)
        if min(meas['x'][1], meas['y'][1]) < 0.05:
            L.event('ORI-001', f"Orientation not checked: the picture has too little detail (response "
                               f"{meas['x'][1]:.2f}, {meas['y'][1]:.2f}).", 30)
            return Result("Orientation not checked", Reason="too little detail in the picture: aim at the crosses")
        sx_, sy_ = meas['x'][0], meas['y'][0]
        expect = n * self.motor_step / self.pixel_size * self.image().shape[1] / max(self._frame_px[0], 1)
        bad = [ax for ax, s_ in (('x', sx_), ('y', sy_))
               if not (0.6 * expect < np.hypot(*s_) < 1.6 * expect) or min(abs(s_)) > 0.25 * max(abs(s_))]
        if bad or min(meas['x'][1], meas['y'][1]) < 0.1:      # сдвиг не тот, что должен быть: не верим
            L.event('ORI-001', f"Orientation not checked: shifts X {sx_.round(1)}, Y {sy_.round(1)} px, "
                               f"expected {expect:.0f} px along one axis.", 30)
            return Result("Orientation not checked",
                          Reason="the picture shift is not clear (focus, aim at the crosses or a textured place)")
        swap = abs(sx_[1]) > abs(sx_[0])                    # X двигает картинку по вертикали
        kx = sx_[1] if swap else sx_[0]
        ky = sy_[0] if swap else sy_[1]
        base = -1 if not self.turn_180 else 1               # знак при mirror = False (см. _pixels_per_step)
        new = dict(swap_xy=bool(swap), mirror_x=bool(np.sign(kx) != base), mirror_y=bool(np.sign(ky) != base))
        old = {k: bool(getattr(self, k)) for k in new}
        px = n * self.motor_step / np.mean([abs(kx), abs(ky)])
        r = Result("Orientation checked", **{"Moved, um": round(n * self.motor_step, 1),
                   "Picture moved X, px": tuple(np.round(sx_, 1)), "Picture moved Y, px": tuple(np.round(sy_, 1)),
                   "Pixel, um": f"{px:.4f} (set {self.pixel_size:.4f})"})
        if new != old:
            r['Changed'] = ", ".join(f"{k} {old[k]} -> {new[k]}" for k in new if new[k] != old[k])
            if fix:
                for k, v in new.items():
                    setattr(self, k, v)
        else:
            r['Result'] = "correct"
        L.event('ORI-001', f"Orientation: {dict(r)}.")
        return r

    def measure_speed(self, distance=1000):
        """Измерить скорость мотора X камерой: столик едет на distance мкм, камера ждёт, когда картинка
        перестанет меняться. m.motor_speed ставится 0.8 от измеренной (с запасом), столик возвращается.
        Нужен образец с деталями и свет."""
        self._need_cam()
        self._need_certain()
        n = self._u2p(distance)
        if not 0 < n * self.motor_step <= self.max_move:
            raise SafetyError('E206', f"distance {distance} um, allowed {self.motor_step:.1f} .. {self.max_move:g} um.")
        self._check({'x': self._p['x'] + n})
        small = lambda: cv2.cvtColor(cv2.resize(self.image(0), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA),
                                     cv2.COLOR_BGR2GRAY).astype(np.float32)
        a, b = small(), small()
        noise = float(np.abs(a - b).mean()) + 0.5
        pos = (n > 0) != self.reverse_x
        if not self._motors_on:
            self._cmd(P.TURN_MOTOR, b"\x01")
            self._motors_on = True
        self._remember(moving=True)
        t0 = time.monotonic()
        st = self._cmd(P.MOTOR['x'], bytes([0x00 if pos else 0xFF]) + n.to_bytes(2, 'big'))
        self._p['x'] += n
        if st and st[0] != P.MOTOR_OK:
            self._uncertain.add('x')
            raise CommError('E306', f"Axis X: controller says {P.MOTOR_STATUS.get(st[0], st[0])}. Check the stage.")
        prev, last_move, moving, t_limit = a, None, 0, t0 + 30
        while time.monotonic() < t_limit:
            cur = small()
            t = time.monotonic()
            if np.abs(cur - prev).mean() > 3 * noise:
                last_move, moving = t, moving + 1
            elif last_move and t - last_move > 1.0:
                break
            prev = cur
            self._check_stop()
        v = n * self.motor_step / max(last_move - t0, 0.02) if last_move else 0.0
        # проверки, что движение действительно было видно: иначе скорость завышена, а это опасно
        seen = float(np.abs(prev - a).mean()) > 3 * noise   # конечный кадр отличается от начального
        if last_move is None or moving < 3 or not seen or v > 20000:
            self._sleep(n * self.motor_step / 200)              # движение не видели — ждём как при 200 мкм/с
            self._go({'x': self._p['x'] - n})
            raise SafetyError('E212', f"The motion was not seen clearly (frames {moving}, "
                                      f"{'picture changed' if seen else 'picture did not change'}). "
                                      "Speed is not changed. Focus first (m.focus()) on a place with details, light on.")
        self._sleep(self.move_pause)
        self.motor_speed = float(max(50, round(0.8 * v)))
        self._go({'x': self._p['x'] - n})
        L.event('MOV-007', f"Motor speed {v:.0f} um/s over {n * self.motor_step:.0f} um; m.motor_speed = {self.motor_speed:.0f} um/s.")
        return Result("Speed measured", Measured=f"{v:.0f} um/s", **{"m.motor_speed set to": f"{self.motor_speed:.0f} um/s (80 %)"})

    # ---------------- камера ----------------
    def _need_cam(self):
        if not self.cam:
            raise CameraError('E503', "Connect with Microscope(..., camera=True).")

    @property
    def frame_size(self):
        """Поле зрения на образце вдоль осей столика (по X, по Y), мкм. При swap_xy по X идёт высота кадра."""
        w, h = self._frame_px[0] * self.pixel_size, self._frame_px[1] * self.pixel_size
        return (h, w) if self.swap_xy else (w, h)

    @property
    def sharpness(self):
        """Резкость текущего кадра (больше — резче)."""
        return round(F.sharpness(self.image(0), self.focus_method), 1)

    def image(self, flush=None):
        """Текущий кадр с камеры (numpy BGR) без сохранения."""
        self._need_cam()
        import threading
        lock = self.__dict__.setdefault('_cam_lock', threading.RLock())   # живой поток manual и остальные — по очереди
        with lock:
            img = self.cam.grab(FRAME_FLUSH if flush is None else flush)
        if self.turn_180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        self._frame_px = (img.shape[1], img.shape[0])
        return img

    def photo(self, name=None):
        """Снимок в photos/<дата>/ (как M240). name — своё имя файла, иначе время и координаты.
        Возвращает кадр (numpy)."""
        img = self.image()
        p = self.position
        name = name or datetime.now().strftime("%H%M%S_") + f"X{p.x:.0f}_Y{p.y:.0f}_Z{p.z:.0f}.png"
        path = os.path.join(self.dirs['photos'], datetime.now().strftime("%Y-%m-%d"), name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imwrite(path, img)
        self.last_photo = path
        L.event('CAM-001', f"Picture {path} at {p}.")
        return img

    def white_balance(self):
        """Однократный баланс белого по текущему кадру (наведите на пустой участок)."""
        self._need_cam()
        r = self.cam.set(white_balance=True)
        L.event('CAM-002', "White balance done.")
        return Result("White balance done")

    def _average(self, n):
        acc = None
        for _ in range(n):
            f = self.image().astype(np.float32)
            acc = f if acc is None else acc + f
        return acc / n

    def _save_ref(self, kind, img):
        os.makedirs(self.dirs['references'], exist_ok=True)
        path = os.path.join(self.dirs['references'],
                            datetime.now().strftime(f"{kind}_{self.objective}x_%Y%m%d_%H%M%S.png"))
        imwrite(path, np.clip(img * 256, 0, 65535).astype(np.uint16))   # 16 бит: точность среднего
        self.references[kind] = path
        L.event('CAM-003', f"{kind.capitalize()} frame {path}, mean {img.mean():.1f}.")
        return Result(f"{kind.capitalize()} frame saved", File=path, **{"Mean level": round(float(img.mean()), 1)})

    def dark_frame(self, frames=16):
        """Тёмный кадр: свет выключается, усредняются frames кадров, свет возвращается.
        Шум и «горячие» пиксели матрицы — для коррекции снимков датасета."""
        lvl = self.light
        self.light = 0
        try:
            img = self._average(frames)
        finally:
            self.light = lvl
        return self._save_ref('dark', img)

    def empty_frame(self, frames=16):
        """Кадр однородного поля: наведите на пустой участок (чистая подложка), усредняются frames кадров.
        Неравномерность освещения — для коррекции снимков датасета."""
        return self._save_ref('empty', self._average(frames))

    # ---------------- фокус ----------------
    @property
    def depth_of_field(self):
        """Глубина резкости объектива, мкм: λ/NA² + e/(M·NA)."""
        from .config import OBJECTIVES, WAVELENGTH_UM, SENSOR_PIXEL_UM
        na = OBJECTIVES.get(self.objective, {}).get('na', 0.25)
        return F.depth_of_field(na, self.objective, WAVELENGTH_UM, SENSOR_PIXEL_UM)

    def expected_z(self):
        """Ожидаемый Z фокуса здесь (настоящие мкм) по точкам фокуса из m.manual(), или None."""
        regs = getattr(self, '_regions', None) or []
        pts = [p for r in regs for p in r['pts']] or getattr(self, '_focus_pts', None) or []
        if not pts:
            return None
        from .scan import focus_z
        rp = self.real_position
        return focus_z(pts, rp.x, rp.y)

    def focus(self, wide=False):
        """Автофокус (как G30). Обычный: узкое окно вокруг ожидаемого Z (по точкам фокуса, иначе текущий Z),
        ±4 глубины резкости, шаг — половина глубины резкости (не меньше шага мотора), проход снизу вверх,
        пик уточняется параболой. wide=True — прежний широкий поиск (focus_range, focus_step): если фокус
        совсем потерян."""
        self._need_cam()
        t0 = time.monotonic()
        if wide:
            r = F.run(self, self.focus_method, self.focus_range, self.focus_step)
        else:
            r = F.run_physics(self, self.focus_method, center_um=self.expected_z())
        weak = r['contrast'] < (WEAK_FOCUS if wide else 1.05)
        L.event('AF-001', f"Focused at Z {r['z_um']:.1f} um, sharpness {r['score']:.1f}, contrast {r['contrast']:.2f}, "
                          f"{r['points']} frames, {time.monotonic() - t0:.1f} s." + (" Weak peak." if weak else ""),
                30 if weak else 20)
        res = Result("Focused", Z=f"{r['z_um'] - self._off['z']:.1f} um", Sharpness=round(r['score'], 4 if not wide else 1),
                     Contrast=round(r['contrast'], 2), Frames=r['points'], Time=f"{time.monotonic() - t0:.1f} s")
        if not wide:
            res.update(**{"Depth of field, um": round(r['dof_um'], 1), "Window, um": f"±{r['half_um']:.0f}, step {r['step_um']:.1f}"})
            if r['dof_um'] < self.motor_step:
                res['Limit'] = (f"Z motor step {self.motor_step:.2f} um > depth of field {r['dof_um']:.1f} um: "
                                f"the stage can stop up to {self.motor_step / 2:.1f} um off the sharpest place")
        if weak:
            res['Note'] = "Weak focus peak: little detail in view. Check the picture, add light, or move to a feature."
        return res

    # ---------------- сценарий ----------------
    def manual(self, fullscreen=True, _keys=None):
        """Ручной режим: живая картинка на весь экран.
        Стрелки или WASD — на кадр в сторону, [ ] — шаг мельче/крупнее, Q / E — столик вниз/вверх,
        F — автофокус, 1 и 2 — углы зоны, Пробел — снимок, L — свет, H — подсказка, Esc — выход.
        Возвращает зону (Area) и запоминает её в m.area."""
        from .manual import run
        return run(self, fullscreen=fullscreen, _keys=_keys)

    def plan(self, area=None, focus='auto', overlap=None, path='snake'):
        """Программа скана в G-коде (её можно посмотреть, изменить и запустить через m.gcode)."""
        from .scan import plan
        from .config import scan_overlap
        return plan(self, area=area, focus=focus, overlap=scan_overlap(self.objective) if overlap is None else overlap, path=path)

    def scan(self, area=None, focus='auto', overlap=None, path='snake', name='', notes=''):
        """Снять зону. Для каждого кадра: встать на координаты, фокус, пауза (photo_pause), снимок, сдвиг.
        focus: 'auto' — по умолчанию: есть точки фокуса из m.manual() (углы 1, 2 и клавиша F) — Z каждого кадра
               по 4 ближайшим (веса 1/d²), иначе 'off'; 'points' — то же, но точки обязательны;
               'off' — Z не меняется (фокус ставится один раз в m.manual(); смена Z сдвигает
               картинку по X и Y и портит сшивку); 'plane' — плоскость фокуса по точкам через focus_every кадров;
               'each' — автофокус в каждом кадре. Пересъёмка размытых кадров с фокусом — m.refocus = True.
        overlap — перекрытие соседних кадров 0..0.5; path — 'snake' или 'rows'.
        name, notes — название образца и заметки (пишутся в описание скана).
        Несколько образцов (области 1, 2, 3… из m.manual()): каждая область снимается в свою папку
        sample_01, sample_02… внутри одной папки сеанса, Z кадров — по точкам фокуса своей области.
        Возвращает папку скана (для нескольких образцов — папку сеанса)."""
        from .scan import plan
        regs = getattr(self, '_regions', None) or []
        pts = None
        if area is None and len(regs) > 1:
            return self._scan_samples(regs, focus, overlap, path, name, notes)
        if area is None and len(regs) == 1:                # одна область — обычный скан в одну папку
            from .scan import Area
            rg, off = regs[0], self._off
            area = Area(rg['A'][0] - off['x'], rg['A'][1] - off['y'], rg['B'][0] - off['x'], rg['B'][1] - off['y'])
            pts = rg['pts']
        if self.cam and self.scan_check:                   # знаки осей картинки — до плана (от них зависит сшивка)
            r = self.check_orientation()
            if 'Changed' in r or 'Reason' in r:
                print(r)
        from .config import scan_overlap
        prog = plan(self, area=area, focus=focus, overlap=scan_overlap(self.objective) if overlap is None else overlap, path=path,
                    pts=pts)
        r = self.gcode(prog, save=name or 'scan', notes=notes, reuse=bool(name))   # то же имя — та же папка
        r.title = "Scan finished" if r.title == "Program finished" else "Scan stopped"
        print(r)
        return r['Folder']

    def _scan_samples(self, regs, focus, overlap, path, name, notes):
        """Скан нескольких областей: папка сеанса, в ней по обычной папке скана на образец."""
        import json as _json
        from datetime import datetime as _dt
        from .scan import plan, Area
        from .config import scan_overlap
        from .samples import color
        if self.cam and self.scan_check:
            r = self.check_orientation()
            if 'Changed' in r or 'Reason' in r:
                print(r)
        ov = scan_overlap(self.objective) if overlap is None else overlap
        off = self._off
        from .util import named_folder
        ses = named_folder(self.dirs['scans'], self.objective, name or 'samples', reuse=bool(name))   # то же имя — та же папка
        os.makedirs(ses, exist_ok=True)
        info = dict(project="MicroControl", name=name, notes=notes, objective=self.objective,
                    date=_dt.now().isoformat(timespec='seconds'), samples=[])
        n0 = 0
        from .samples import init_areas
        init_areas(self)                                      # у каждой области — постоянный номер и uid
        for k, reg in enumerate(regs):
            sn, uid = reg['num'], reg['uid']
            area = Area(reg['A'][0] - off['x'], reg['A'][1] - off['y'], reg['B'][0] - off['x'], reg['B'][1] - off['y'])
            pts = [dict(p, tag=str(n0 + i + 1)) for i, p in enumerate(reg['pts'])]
            nums = [n0 + 1, n0 + len(pts)]
            n0 += len(pts)
            info['samples'].append(dict(index=sn, uid=uid, folder=f"sample_{sn:02d}", color_bgr=list(color(sn - 1)),
                                        area_um=[area.x0, area.y0, area.x1, area.y1], focus_points=nums))
            _json.dump(info, open(os.path.join(ses, 'session.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
            prog = plan(self, area=area, focus=focus, overlap=ov, path=path, pts=pts)
            print(f"Sample {sn} ({k + 1} of {len(regs)}): {area}")
            r = self.gcode(prog, save=f"{name or 'sample'} {sn}", notes=notes,
                           folder=os.path.join(ses, f"sample_{sn:02d}"),
                           extra=dict(sample_index=sn, sample_uid=uid, session=os.path.basename(ses), focus_points=nums))
            print(r)
            if r.title != "Program finished":
                break
        print(f"Session: {ses}")
        return ses

    def gcode(self, program, save=None, notes='', folder=None, extra=None, reuse=False):
        """Выполнить G-код (строка или путь к файлу .gcode). Единицы — мкм.
        save='имя' — все снимки (M240) пишутся в новую папку скана; без save — в photos/.
        Перед запуском программа целиком проверяется: при ошибке столик не двигается."""
        from .gcode import run
        return run(self, program, save=save, notes=notes, folder=folder, extra=extra, reuse=reuse)
