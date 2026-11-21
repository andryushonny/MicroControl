"""Метрики резкости и автофокус."""
import numpy as np
import cv2
from .errors import FocusError, LimitError
from . import log as L


def _prep(img, roi=0.5, down=2):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    dh, dw = int(h * (1 - roi) / 2), int(w * (1 - roi) / 2)
    g = g[dh:h - dh, dw:w - dw]
    if down > 1:
        g = cv2.resize(g, None, fx=1 / down, fy=1 / down, interpolation=cv2.INTER_AREA)
    return g.astype(np.float32)


def _laplacian(g):  return float(cv2.Laplacian(g, cv2.CV_32F).var())
def _tenengrad(g):  return float(np.mean(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2))
def _brenner(g):    return float(np.mean((g[:, 2:] - g[:, :-2]) ** 2))
def _variance(g):   return float(g.var() / max(g.mean(), 1.0))


METHODS = {'laplacian': _laplacian, 'tenengrad': _tenengrad,
           'brenner': _brenner, 'variance': _variance}


def sharpness(img, method='laplacian', roi=0.5, down=2):
    """Резкость кадра (больше = резче). method: имя из METHODS или своя функция f(gray)->float.
    down — во сколько раз уменьшить кадр: 8 — только крупные детали (широкий пик для грубого поиска)."""
    f = method if callable(method) else METHODS.get(method)
    if f is None:
        raise FocusError('E602', f"{method!r}. Use one of {list(METHODS)}.")
    return f(_prep(img, roi, down))


def run(m, method, range_um, step_um, roi=0.5):
    """Автофокус в два прохода.
    1) Грубо: сверху вниз по окну ±range с шагом step, резкость по уменьшенному кадру (крупные детали
       видны и при расфокусе, поэтому пик широкий и не проскакивается, даже если глубина резкости меньше шага).
    2) Точно: ±step вокруг лучшей точки с шагом в один шаг мотора, выбранной мерой резкости.
    Подход к лучшему Z — сверху (выбирается люфт)."""
    upp = m.motor_step
    zlo, zhi = m._lim['z']
    z0 = m._p['z']
    R, s = max(1, round(range_um / upp)), max(1, round(step_um / upp))
    top, bot = min(z0 + R, zhi), max(z0 - R, zlo)
    if top <= bot:
        raise FocusError('E601', "The Z window is empty. Check m.z_max and m.z_min.")
    curve = []

    def scan(zs, st, down):
        best, vals = (-np.inf, zs[0]), []
        for i, z in enumerate(zs):
            try:
                m._go({'z': z})
            except LimitError:                              # концевик прибора: окно кончилось раньше
                L.event('AF-003', f"Limit switch at Z {m._p['z'] * upp:.0f} um during autofocus; the window is cut there.", 30)
                if i:
                    break
                z = m._p['z']
            v = sharpness(m.image(), method, roi, down)
            curve.append((z, v))
            vals.append(v)
            if v > best[0]:
                best = (v, z)
            if i >= 3 and z < best[1] - 2 * st and v < 0.6 * best[0]:
                break                                       # пик пройден
        return best, vals

    (cv, cz), coarse = scan(list(range(top, bot - 1, -s)), s, 8)       # грубо, вниз (от объектива)
    contrast = float(cv / max(np.median(coarse), 1e-9))                 # пик / типичная: ~1 — пика нет
    best = (cv, cz)
    if s > 1:                                                            # точно: шаг мотора, своя мера
        best, _ = scan(list(range(min(top, cz + s), max(bot, cz - s) - 1, -1)), 1, 2)
    m._go({'z': min(top, best[1] + s)})                                  # подход к лучшей точке сверху
    m._go({'z': best[1]})
    return dict(z_um=best[1] * upp, score=best[0], points=len(curve), contrast=contrast,
                curve=[(z * upp, v) for z, v in curve])


# ------------------------------------------------------------------ автофокус по физике
def depth_of_field(na, mag, wavelength=0.55, sensor_px=2.2, n=1.0):
    """Глубина резкости, мкм: волновая часть λn/NA² плюс геометрическая n·e/(M·NA) (e — пиксель матрицы).
    Пример: 10x, NA 0,25: 8,8 + 0,9 = 9,7 мкм; 100x, NA 0,9: 0,68 + 0,02 = 0,7 мкм."""
    return wavelength * n / na ** 2 + n * sensor_px / (mag * na)


def _score(m, method, roi, frames=2):
    """Резкость, усреднённая по нескольким кадрам; кадр слегка сглажен (шум матрицы не считается деталью)."""
    v = 0.0
    for _ in range(frames):
        img = m.image()
        g = _prep(img, roi, 2)
        g = cv2.GaussianBlur(g, (0, 0), 1.0)
        f = method if callable(method) else METHODS[method]
        v += f(g) / max(float(g.mean()) ** 2, 1.0)           # не зависит от яркости и экспозиции
    return v / frames


def run_physics(m, method, center_um=None, half_um=None, step_um=None, roi=0.5, extend=2):
    """Автофокус вокруг ожидаемого Z (по точкам фокуса или текущему).
    Окно ±half (по умолчанию 4 глубины резкости, не меньше 3 шагов мотора), шаг — половина глубины
    резкости, не меньше шага мотора. Проход всегда снизу вверх (люфт выбирается одинаково), пик уточняется
    параболой по логарифму резкости, к лучшему Z столик тоже подходит снизу. Пик на краю окна — окно
    сдвигается (до extend раз)."""
    upp = m.motor_step
    zlo, zhi = m._lim['z']
    dof = m.depth_of_field
    s = max(1, int(round((step_um or dof / 2) / upp)))
    H = max(3 * s, int(round((half_um or 4 * dof) / upp)))
    c = int(round((center_um / upp) if center_um is not None else m._p['z']))
    curve = []
    for k in range(extend + 1):
        lo, hi = max(zlo, c - H), min(zhi, c + H)
        zs = list(range(lo, hi + 1, s))
        try:
            m._go({'z': lo - 2 * s})                           # подход снизу
        except LimitError:
            pass
        vals = []
        for z in zs:
            m._go({'z': z})
            v = _score(m, method, roi)
            vals.append(v)
            curve.append((z, v))
        i = int(np.argmax(vals))
        if max(vals) / max(min(vals), 1e-12) < 1.15:          # кривая плоская: фокус далеко — широкий поиск
            L.event('AF-004', "No focus peak in the narrow window: wide search.", 20)
            m._go({'z': c})
            r = run(m, method, m.focus_range, m.focus_step, roi)
            r.update(dof_um=dof, step_um=s * upp, half_um=H * upp, wide=True)
            return r
        if 0 < i < len(zs) - 1 or k == extend:
            break
        c = zs[i] + (H if i == len(zs) - 1 else -H)            # пик на краю: окно туда
        L.event('AF-004', f"Focus peak at the edge of the window, the window moves to Z {c * upp:.0f} um.", 20)
    zbest = float(zs[i])
    if 0 < i < len(zs) - 1:                                    # парабола по log: пик резкости ~ гауссов
        a, b, cc = np.log(np.maximum([vals[i - 1], vals[i], vals[i + 1]], 1e-12))
        den = a - 2 * b + cc
        if den < 0:
            zbest = zs[i] + s * 0.5 * (a - cc) / den
    zt = int(round(zbest))
    m._go({'z': max(zlo, zt - 2 * s)})                         # к лучшему Z — тоже снизу
    m._go({'z': zt})
    contrast = float(max(vals) / max(min(vals), 1e-12))
    return dict(z_um=zt * upp, z_fit_um=zbest * upp, score=float(max(vals)), points=len(curve),
                contrast=contrast, dof_um=dof, step_um=s * upp, half_um=H * upp,
                curve=[(z * upp, v) for z, v in curve])
