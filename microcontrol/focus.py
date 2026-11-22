"""Метрики резкости и автофокус."""
import numpy as np
import cv2
from .errors import FocusError, LimitError
from . import log as L


def _prep(img, roi=0.5, down=2):
    g = img if img.ndim == 2 else img[..., 1]                   # зелёный канал: резче всего, без цветной каймы
    h, w = g.shape
    dh, dw = int(h * (1 - roi) / 2), int(w * (1 - roi) / 2)
    g = g[dh:h - dh, dw:w - dw]
    if down > 1:
        g = cv2.resize(g, None, fx=1 / down, fy=1 / down, interpolation=cv2.INTER_AREA)
    return g.astype(np.float32)


def _vollath(g):
    """Vollath F4: автокорреляция со сдвигом 1 минус со сдвигом 2 (по обеим осям). Шум пикселей почти не
    даёт вклада (он не коррелирует ни на 1, ни на 2), детали дают — лучший выбор для светлого поля
    (Groen 1985; Sun, Duthaler, Nelson 2004)."""
    g = g - float(g.mean())                                     # без среднего: иначе разность огромных чисел
    x = float(np.mean(g[:, :-1] * g[:, 1:]) - np.mean(g[:, :-2] * g[:, 2:]))
    y = float(np.mean(g[:-1] * g[1:]) - np.mean(g[:-2] * g[2:]))
    return max(x + y, 1e-12)


def _normvar(g):     return float(g.var() / max(float(g.mean()), 1.0) ** 2)
def _laplacian(g):  return float(cv2.Laplacian(g, cv2.CV_32F).var())
def _tenengrad(g):  return float(np.mean(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2))
def _brenner(g):    return float(np.mean((g[:, 2:] - g[:, :-2]) ** 2))
def _variance(g):   return float(g.var() / max(g.mean(), 1.0))


METHODS = {'vollath': _vollath, 'normvar': _normvar, 'laplacian': _laplacian, 'tenengrad': _tenengrad,
           'brenner': _brenner, 'variance': _variance}


def sharpness(img, method='laplacian', roi=0.5, down=2):
    """Резкость кадра (больше = резче). method: имя из METHODS или своя функция f(gray)->float.
    down — во сколько раз уменьшить кадр: 8 — только крупные детали (широкий пик для грубого поиска)."""
    f = method if callable(method) else METHODS.get(method)
    if f is None:
        raise FocusError('E602', f"{method!r}. Use one of {list(METHODS)}.")
    return f(_prep(img, roi, down))


def run(m, method, range_um, step_um, roi=0.5, pr=None):
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
        if pr:
            pr.total = pr.done + len(zs) + (2 * s + 1 if st > 1 else 0) + 1
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
            if pr:
                pr.update(1, f"{'wide search' if st > 1 else 'fine'}: Z {z * upp:.1f} um, sharpness {v:.3g}")
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


def _score(m, method, roi, frames=2, down=2):
    """Резкость, усреднённая по нескольким кадрам; кадр слегка сглажен (шум матрицы не считается деталью)."""
    v = 0.0
    for _ in range(frames):
        img = m.image()
        g = _prep(img, roi, down)
        g = cv2.GaussianBlur(g, (0, 0), 1.0)
        f = method if callable(method) else METHODS[method]
        v += f(g) / max(float(g.mean()) ** 2, 1.0)           # не зависит от яркости и экспозиции
    return v / frames


def run_physics(m, method, center_um=None, half_um=None, step_um=None, roi=0.5, extend=2, pr=None):
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
        if pr:
            pr.total = pr.done + len(zs) + 1                  # +1: подход к лучшему Z
        for z in zs:
            m._go({'z': z})
            v = _score(m, method, roi)
            vals.append(v)
            curve.append((z, v))
            if pr:
                pr.update(1, f"{'window ' + str(k + 1) + ': ' if k else ''}Z {z * upp:.1f} um, sharpness {v:.3g}")
        i = int(np.argmax(vals))
        if max(vals) / max(min(vals), 1e-12) < 1.15:          # кривая плоская: фокус далеко — широкий поиск
            L.event('AF-004', "No focus peak in the narrow window: wide search.", 20)
            m._go({'z': c})
            r = run(m, method, m.focus_range, m.focus_step, roi, pr=pr)
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


# ------------------------------------------------------------------ автофокус: скобка, точный проход, подгонка
def fit_peak(zs, vs):
    """Пик кривой резкости: около фокуса она близка к гауссиане, ln f — к параболе. МНК по точкам выше
    половины максимума (не меньше 3, с весом f). -> (z пика, σ, хорошая ли подгонка)."""
    zs, vs = np.asarray(zs, float), np.asarray(vs, float)
    i = int(np.argmax(vs))
    base = float(np.min(vs))
    use = np.flatnonzero(vs - base >= 0.5 * (vs[i] - base))
    if len(use) < 3:                                          # узкий пик: три точки вокруг максимума
        use = np.arange(max(0, i - 1), min(len(vs), i + 2))
    if len(use) < 3:
        return float(zs[i]), None, False
    z, f = zs[use], np.maximum(vs[use] - 0.9 * base, 1e-12)
    zc = z.mean()
    A = np.c_[np.ones(len(z)), z - zc, (z - zc) ** 2] * f[:, None]
    a, b, c = np.linalg.lstsq(A, np.log(f) * f, rcond=None)[0]
    if c >= 0:
        return float(zs[i]), None, False
    mu = zc - b / (2 * c)
    ok = z.min() - 0.5 * np.ptp(z) <= mu <= z.max() + 0.5 * np.ptp(z)
    return (float(mu) if ok else float(zs[i])), float(np.sqrt(-1 / (2 * c))), bool(ok)


def run_af(m, method, center_um=None, step_um=None, range_um=None, roi=0.5, pr=None):
    """Автофокус в три шага, всё — подходом снизу (люфт выбирается одинаково):
    1) скобка: окно ±range вокруг ожидаемого Z, шаг step (грубо, уменьшенный кадр — пик шире); пик на краю —
       окно сдвигается; кривая плоская — окно и шаг ×3, затем широкий поиск (focus_range прибора);
    2) точно: ±1.5 шага вокруг лучшей точки, через один шаг мотора (или треть глубины резкости, если она больше);
    3) подгонка: ln(резкость) — парабола (гауссиана) по точкам выше половины пика; столик — в ближайший к пику
       шаг мотора; проверка: там резкость не хуже 85 % лучшей измеренной, иначе — лучшая измеренная точка.
    step, range — для объектива (m.af_step, m.af_range), по умолчанию от глубины резкости."""
    upp = m.motor_step
    zlo, zhi = m._lim['z']
    dof = m.depth_of_field
    sc = max(1, int(round((step_um or max(dof, upp)) / upp)))
    H = max(3 * sc, int(round((range_um or 6 * dof) / upp)))
    sf = max(1, int(round(dof / 3 / upp)))
    c = int(round((center_um / upp) if center_um is not None else m._p['z']))
    curve = []

    coarse = 'normvar'                                        # грубо — контраст: пик широкий, виден издалека;
                                                              # точно — выбранная мера (по деталям)
    def sweep(zs, down, label, meth=None):
        try:
            m._go({'z': max(zlo, zs[0] - 2 * max(1, zs[1] - zs[0] if len(zs) > 1 else 1))})   # подход снизу
        except LimitError:
            pass
        vals = []
        if pr:
            pr.total = pr.done + len(zs) + 1
        for z in zs:
            try:
                m._go({'z': z})
            except LimitError:
                break
            v = _score(m, meth or method, roi, 2, down)
            vals.append(v)
            curve.append((z, v))
            if pr:
                pr.update(1, f"{label}: Z {z * upp:.1f} um, sharpness {v:.3g}")
            if len(vals) >= 4 and v < 0.5 * max(vals) and vals[-2] < 0.6 * max(vals) and \
                    int(np.argmax(vals)) < len(vals) - 2:
                break                                        # пик пройден
        return zs[:len(vals)], vals

    tries, wide_done, sc0, H0 = 0, False, sc, H
    while True:
        lo, hi = max(zlo, c - H), min(zhi, c + H)
        zs, vals = sweep(list(range(lo, hi + 1, sc)), 4, "bracket", coarse)
        if not vals:
            raise FocusError('E601', "The Z window is empty. Check m.z_max and m.z_min.")
        i = int(np.argmax(vals))
        med = float(np.median(vals))
        mad = float(np.median(np.abs(np.asarray(vals) - med))) + 1e-30
        flat = max(vals) < 2.0 * med or (max(vals) - med) < 8 * mad      # пика нет: только шум (мера без шумового пола)
        if flat and tries == 0:                               # далеко от фокуса: шире и крупнее
            H, sc, tries = 3 * H, 3 * sc, 1
            L.event('AF-004', f"No focus peak within the window: wider (±{H * upp:.0f} um, step {sc * upp:.0f} um).", 20)
            continue
        if flat and not wide_done:
            L.event('AF-004', "No focus peak: wide search.", 20)
            m._go({'z': c})
            r = run(m, coarse, m.focus_range, m.focus_step, roi, pr=pr)
            c, H, sc, tries = int(round(r['z_um'] / upp)), max(3 * sc0, H0), sc0, 2
            wide_done = True
            continue                                          # широкий поиск нашёл место: скобка ещё раз там
        if flat:                                              # и там пика нет: в лучшую точку, слабый фокус
            break
        if (i == 0 or i == len(zs) - 1) and tries < 3 and len(zs) > 2 and not (zs[i] in (zlo, zhi)):
            c = zs[i] + (H if i == len(zs) - 1 else -H)       # пик на краю: окно туда
            tries += 1
            L.event('AF-004', f"Focus peak at the edge of the window, the window moves to Z {c * upp:.0f} um.", 20)
            continue
        break
    zb = zs[i]
    contrast = float(max(vals) / max(min(vals), 1e-12))
    fz, fv = sweep(list(range(max(zlo, zb - int(1.5 * sc)), min(zhi, zb + int(1.5 * sc)) + 1, sf)), 2, "fine")
    if len(fz) < 3:
        fz, fv = zs, vals
    mu, sig, ok = fit_peak(fz, fv)
    zt = int(round(mu))
    m._go({'z': max(zlo, zt - 2 * sf)})                       # к пику — снизу
    m._go({'z': zt})
    v_end = _score(m, method, roi, 2, 2)
    best_i = int(np.argmax(fv))
    if v_end < 0.85 * fv[best_i] and fz[best_i] != zt:        # подгонка подвела: в лучшую измеренную точку
        zt = fz[best_i]
        m._go({'z': max(zlo, zt - 2 * sf)})
        m._go({'z': zt})
        v_end = _score(m, method, roi, 2, 2)
    return dict(z_um=zt * upp, z_fit_um=mu * upp, score=float(v_end), points=len(curve), contrast=contrast,
                dof_um=dof, step_um=sc * upp, half_um=H * upp, fine_um=sf * upp, fit=ok,
                width_um=(2.355 * sig * upp) if sig else None, curve=[(z * upp, v) for z, v in curve])
