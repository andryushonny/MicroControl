"""Симулятор микроскопа (порт 'sim'): учиться и проверять скрипты без железа."""
import numpy as np
import cv2
from . import protocol as P
from .config import UM_PER_STEP, SENSOR_PIXEL_UM


class SimWorld:
    """Виртуальный прибор: столик с концевиками, лампа, лазер 532 нм и образец.

    Образец — подложка с разметкой: кресты с шагом pitch мкм (каждый 5-й крупнее, как реперные метки),
    между ними — чешуйки разной толщины и мелкая текстура (по ним работает автофокус).
    Образец рисуется в микрометрах, поэтому при смене объектива масштаб честный.
    Лазер — пятно в центре кадра: размер зависит от объектива и расфокуса, яркость — от мощности.
    От мощности burn_mw в фокусе на образце остаётся тёмный след (как настоящий ожог)."""
    def __init__(self, seed=0, pitch=100.0, cross=20.0, line=2.5, burn_mw=20.0):
        rng = np.random.default_rng(seed)
        self.pitch, self.cross, self.line, self.burn_mw = pitch, cross, line, burn_mw
        n, self.tex_um = 3000, 0.5                    # фон: 3000 px по 0.5 мкм = 1.5 мм, дальше повторяется
        noise = np.zeros((n, n), np.float32)
        for k in (3, 10, 40):
            noise += cv2.resize(rng.random((n // k, n // k)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
        noise = cv2.normalize(noise, None, -1, 1, cv2.NORM_MINMAX)
        sub = np.array([150, 120, 110], np.float32)   # подложка (BGR), розовато-серая как SiO2/Si
        tex = sub + 10 * noise[..., None]
        fine = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), 1.2)
        tex += 6 * fine[..., None] / fine.std()        # шероховатость подложки (мелкие детали, ~1 мкм)
        for _ in range(6000):                          # пылинки 0.5–2 мкм
            c = tuple(int(v) for v in rng.integers(0, n, 2))
            cv2.circle(tex, c, int(rng.integers(1, 4)), tuple(float(v) * rng.uniform(0.5, 0.8) for v in sub), -1)
        for _ in range(120):                           # чешуйки: многоугольники разной толщины (контраста)
            c = rng.integers(0, n, 2)
            r = rng.uniform(8, 60)
            ang = np.sort(rng.uniform(0, 2 * np.pi, rng.integers(5, 9)))
            pts = (c + np.c_[np.cos(ang), np.sin(ang)] * r * rng.uniform(0.5, 1.0, len(ang))[:, None]).astype(np.int32)
            k = rng.uniform(0.55, 0.92)                # тоньше — светлее
            mask = np.zeros((n, n), np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            tex[mask > 0] *= np.array([k * 1.05, k * 0.85, k * 0.9], np.float32)
        self.tex = np.clip(tex, 0, 255).astype(np.float32)
        self._noise = np.zeros((480, 640, 1), np.float32)   # шум матрицы: новый в каждом кадре
        self.burns = []                                # следы лазера: (x мкм, y мкм, r мкм)
        self._p = {'x': 0.0, 'y': 0.0, 'z': 0.0}       # истинное положение, шаги
        self.steps_per_s = 900                         # скорость мотора: ~5 мм/с
        self.motion = None                             # (ось, откуда, куда, t0, t1)
        self.overlaps = 0                              # команды, пришедшие во время движения
        self.hw = {'x': (-4500, 4500), 'y': (-4500, 4500), 'z': (-3000, 1200)}  # концевики
        self.z_focus = 18                              # фокус в точке (0, 0): 18 шагов = 100 мкм над началом (в окне автофокуса)
        self.tilt = (0.02, -0.015)                    # наклон образца: шагов Z на шаг X, Y
        self.set_objective(10)
        self.light = 0
        self.laser = 0                                 # мВт
        self.motors_on = False                         # моторы X, Y без питания до TURN_MOTOR 1 (как в штатном ПО)
        self.tec = None                                # уставка охлаждения ПЗС, °C
        self.ignored = 0                               # команды X/Y, пришедшие при выключенных моторах

    @property
    def p(self):
        """Истинное положение сейчас (с учётом идущего движения), шаги."""
        import time
        out = dict(self._p)
        if self.motion:
            ax, a, b, t0, t1 = self.motion
            k = min(1.0, (time.monotonic() - t0) / max(t1 - t0, 1e-9))
            out[ax] = a + (b - a) * k
        return out

    def set_objective(self, obj):
        """Настоящий объектив: размер пикселя камеры на образце и апертура (NA)."""
        self.objective = obj
        self.px_um = SENSOR_PIXEL_UM / obj               # мкм на пиксель камеры
        self.na = {4: 0.1, 10: 0.25, 20: 0.4, 100: 0.9}.get(obj, 0.25)

    def handle(self, cmd, data):
        """Ответ виртуального контроллера на команду."""
        import time
        if cmd == P.TURN_MOTOR:
            self.motors_on = bool(data and data[0]); return b"\x00"
        if cmd == P.SET_TEC:
            self.tec = (-1 if data[0] == 0xFF else 1) * data[1]; return b"\x00"
        for ax, c in P.MOTOR.items():
            if cmd == c and ax in 'xy' and not self.motors_on:
                self.ignored += 1                     # ответ есть, движения нет
                return b"\x00"
            if cmd == c:
                n = int.from_bytes(data[1:3], "big")
                sign = -1 if data[0] == 0xFF else 1
                if ax == 'z':
                    sign = -sign                      # Z: 0xFF = вверх
                now = time.monotonic()
                if self.motion and now < self.motion[4]:
                    self.overlaps += 1                # новая команда во время движения:
                    return bytes([P.MOTOR_MOVING])    # контроллер отвечает «ещё едет» и её не выполняет
                self._p = {k: float(v) for k, v in self.p.items()}
                lo, hi = self.hw[ax]
                bl = getattr(self, 'backlash', {}).get(ax, 0.0)          # люфт: при смене направления
                last = getattr(self, '_dir', {}).get(ax)                 # первые bl шагов уходят впустую
                lost = min(n, bl) if (last is not None and last != sign and bl) else 0.0
                self._dir = {**getattr(self, '_dir', {}), ax: sign}
                t = self._p[ax] + sign * (n - lost)
                end = min(max(t, lo), hi)
                self.motion = (ax, self._p[ax], end, now, now + abs(end - self._p[ax]) / self.steps_per_s)
                self._p[ax] = end
                if end == t:
                    return bytes([P.MOTOR_OK])
                # как MotorXResult / MotorYResult / MotorZResult: 1 — вперёд / влево / ВВЕРХ, 2 — назад / вправо / вниз
                return bytes([P.MOTOR_LIMIT_A if sign > 0 else P.MOTOR_LIMIT_B])
        if cmd == P.LIGHT:
            self.light = int.from_bytes(data, "big"); return b"\x00"
        if cmd == P.LASER:
            self.laser = int.from_bytes(data, "big"); return b"\x00"
        if cmd == P.GET_SN:
            return b"SIM-0001"
        if cmd == P.GET_PN:
            return b"ATR8300-SIM"
        if cmd == P.GET_MAX_LASER:
            return b"\xFF\x00"                         # как настоящий прибор (запись 20.11): «не задано»
        if cmd == P.GET_DEVICE_TYPE:
            return bytes([2])
        if cmd == P.GET_CALIBRATION:
            if len(data) != 1:
                return None                            # как настоящий контроллер: без индекса молчит
            return bytes(16)
        return b"\x01"

    def frame(self, W=None, H=None):
        """Кадр камеры. swap=True — камера повёрнута на 90°: X столика идёт по вертикали кадра (как на ATR8300)."""
        W, H = W or getattr(self, 'size', (640, 480))[0], H or getattr(self, 'size', (640, 480))[1]
        img = np.ascontiguousarray(self._frame(H, W).transpose(1, 0, 2)) if getattr(self, 'swap', False) \
            else self._frame(W, H)
        k1 = getattr(self, 'dist_k1', 0.0)                    # дисторсия объектива (бочка < 0 < подушка)
        if k1:
            h, w = img.shape[:2]
            c, R = np.array([w / 2, h / 2], np.float32), float(np.hypot(w / 2, h / 2))
            y, x = np.mgrid[0:h, 0:w].astype(np.float32)
            dx, dy = x - c[0], y - c[1]
            sc = 1 - k1 * (dx * dx + dy * dy) / R ** 2          # снятый пиксель -> точка ровного кадра
            img = cv2.remap(img, c[0] + dx * sc, c[1] + dy * sc, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        vg = getattr(self, 'vignette', 0.0)                    # виньетирование: края темнее
        if vg:
            h, w = img.shape[:2]
            y, x = np.mgrid[0:h, 0:w].astype(np.float32)
            rr = ((x - w / 2) ** 2 + (y - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
            img = np.clip(img * (1 - vg * rr)[..., None], 0, 255).astype(np.uint8)
        vc = getattr(self, 'vignette_bgr', None)               # цветное виньетирование: края другого оттенка
        if vc:
            h, w = img.shape[:2]
            y, x = np.mgrid[0:h, 0:w].astype(np.float32)
            rr = ((x - w / 2) ** 2 + (y - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
            img = np.clip(img * (1 - rr[..., None] * np.array(vc, np.float32)), 0, 255).astype(np.uint8)
        fl = getattr(self, 'flicker', 0.0)                     # яркость лампы от кадра к кадру
        if fl:
            img = np.clip(img * (1 + np.random.uniform(-fl, fl)), 0, 255).astype(np.uint8)
        return img

    def _frame(self, W=640, H=480):
        """Кадр камеры в текущем положении: образец, расфокус, лампа, лазерное пятно."""
        p = self.p
        um = UM_PER_STEP
        x0, y0 = p['x'] * um, p['y'] * um            # центр кадра на образце, мкм
        X = (x0 + (np.arange(W, dtype=np.float32) - W / 2) * self.px_um).astype(np.float32)
        Y = (y0 + (np.arange(H, dtype=np.float32) - H / 2) * self.px_um).astype(np.float32)
        n = self.tex.shape[0]
        mx = np.broadcast_to(np.mod(X / self.tex_um, n)[None], (H, W)).astype(np.float32)
        my = np.broadcast_to(np.mod(Y / self.tex_um, n)[:, None], (H, W)).astype(np.float32)
        img = cv2.remap(self.tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        for fx, fy, fl, fa, fw in getattr(self, 'flakes', ()):     # чешуйки: светлые полоски (x, y, длина, угол, ширина), мкм
            if abs(fx - x0) > W * self.px_um / 2 + fl or abs(fy - y0) > H * self.px_um / 2 + fl:
                continue
            ca, sa = np.cos(np.radians(fa)), np.sin(np.radians(fa))
            u_ = (X[None] - fx) * ca + (Y[:, None] - fy) * sa
            v_ = -(X[None] - fx) * sa + (Y[:, None] - fy) * ca
            fm = (np.clip((fl / 2 - np.abs(u_)) / self.px_um + 0.5, 0, 1) *
                  np.clip((fw / 2 - np.abs(v_)) / self.px_um + 0.5, 0, 1)).astype(np.float32)[..., None]
            img = img * (1 - fm) + np.array([235, 240, 170], np.float32) * fm
        # кресты: золото на подложке; каждый 5-й (через 5 × pitch) вдвое крупнее
        P_, soft = self.pitch, max(self.px_um, 0.3)
        t = np.radians(getattr(self, 'turn_deg', 0.0))       # подложка повёрнута на столике вокруг Z
        Xr = np.cos(t) * X[None] + np.sin(t) * Y[:, None]
        Yr = -np.sin(t) * X[None] + np.cos(t) * Y[:, None]
        dx = (Xr + P_ / 2) % P_ - P_ / 2
        dy = (Yr + P_ / 2) % P_ - P_ / 2
        ix, iy = np.round((Xr - dx) / P_).astype(int), np.round((Yr - dy) / P_).astype(int)
        big = (ix % 5 == 0) & (iy % 5 == 0)
        L = np.where(big, 2 * self.cross, self.cross) / 2
        wl = np.where(big, 1.6 * self.line, self.line) / 2
        ax, ay = np.abs(dx), np.abs(dy)
        edge = lambda d, h: np.clip((h - d) / soft + 0.5, 0, 1)
        m = np.maximum(edge(ax, wl) * edge(ay, L), edge(ay, wl) * edge(ax, L)).astype(np.float32)[..., None]
        img = img * (1 - m) + np.array([70, 185, 225], np.float32) * m
        for bx, by, br in self.burns:                  # следы лазера
            d2 = (X[None] - bx) ** 2 + (Y[:, None] - by) ** 2
            if d2.min() < (3 * br) ** 2:
                img *= (1 - 0.8 * np.exp(-d2 / (2 * br ** 2)))[..., None]
        zf = self.z_focus + self.tilt[0] * p['x'] + self.tilt[1] * p['y']
        dz_um = abs(p['z'] - zf) * um
        sigma = 0.5 * self.na * dz_um / self.px_um     # расфокус в пикселях: конус света с апертурой NA
        if sigma > 0.3:
            img = _blur(img, sigma)
        img *= 1.0 if self.light else 0.15
        if self.laser > 0:
            img += self._laser_spot(W, H, dz_um)
        if self._noise.shape[:2] != img.shape[:2]:
            self._noise = np.zeros(img.shape[:2] + (1,), np.float32)
        cv2.randn(self._noise, 0, 2)
        img += self._noise
        return np.clip(img, 0, 255).astype(np.uint8)

    def _laser_spot(self, W, H, dz_um):
        """Пятно лазера 532 нм в центре кадра. Размер: дифракция + конус расфокуса. Яркость ~ мощность / площадь."""
        s_um = 0.61 * 0.532 / self.na / 2 + self.na * dz_um / 2   # «сигма» пятна, мкм
        s_px = max(s_um / self.px_um, 0.7)
        peak = 400 * self.laser / max(1.0, (s_um / 0.5) ** 2)      # мВт -> яркость, засветка допустима
        u = np.arange(W, dtype=np.float32) - W / 2
        v = np.arange(H, dtype=np.float32) - H / 2
        g = np.exp(-(u[None] ** 2 + v[:, None] ** 2) / (2 * s_px ** 2))
        halo = np.exp(-(u[None] ** 2 + v[:, None] ** 2) / (2 * (4 * s_px) ** 2)) * 0.08
        spot = (min(peak, 4000) * g)[..., None] * np.array([0.8, 1.0, 0.8], np.float32) + \
               (min(peak, 4000) * halo)[..., None] * np.array([0.1, 1.0, 0.1], np.float32)   # белый центр, зелёный ореол
        if self.laser >= self.burn_mw and s_um < 3:                 # в фокусе и мощно — ожог
            p = self.p
            bx, by = p['x'] * UM_PER_STEP, p['y'] * UM_PER_STEP
            if all((bx - a) ** 2 + (by - b) ** 2 > (2 * r) ** 2 for a, b, r in self.burns):
                self.burns.append((bx, by, max(1.0, 2 * s_um)))
        return spot


def _blur(img, sigma):
    """Расфокус; сильный — на уменьшенной копии (быстро на слабом ПК)."""
    if sigma < 4:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    k = sigma / 2
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w / k)), max(8, int(h / k))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


class SimPort:
    """Имитация serial.Serial поверх SimWorld."""
    def __init__(self, world):
        self.w, self.buf, self.timeout, self.is_open = world, b"", 1, True

    def reset_input_buffer(self):
        self.buf = b""

    @property
    def in_waiting(self):
        return len(self.buf)

    def write(self, frame):
        cmd, data = frame[4], frame[5:-1]
        if getattr(self.w, 'stuck', 0):              # контроллер не в такте: кадр уходит «в данные» прошлого
            self.w.stuck -= 1
            return
        reply = self.w.handle(cmd, data)
        if reply is not None:
            self.buf += P.pack(cmd, reply)

    def read(self, n):
        r, self.buf = self.buf[:n], self.buf[n:]
        return r

    def close(self):
        self.is_open = False


class SimCamera:
    """Виртуальная камера."""
    def __init__(self, world):
        self.w, self.exposure_ms = world, 20.0

    def grab(self, flush=0):
        return self.w.frame()

    def set(self, exposure_ms=None, white_balance=False, **kw):
        if exposure_ms is not None:
            self.exposure_ms = float(exposure_ms)
        return self.info()

    def info(self):
        return dict(model="SIM", exposure_ms=self.exposure_ms, size=(640, 480))

    def close(self):
        pass
