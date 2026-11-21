"""Сколько займёт скан: оценка по физике (переезд, пауза, экспозиция, запись), уточняемая по прошлым сканам
этого микроскопа. Для каждого набора настроек (объектив, фокус, паузы, проверки, формат кадров) запоминаются
последние сканы — (кадров, секунд); время подгоняется прямой T = T0 + t·N (МНК; T0 — подготовка, t — на кадр).
Чем больше сканов, тем больше вес выученного; память — в файле прибора (~/.microcontrol/devices/...).
"""
import math

KEEP = 20                       # сколько последних сканов помнить для одного набора настроек


def key(m, focus, objective=None):
    return "|".join(str(v) for v in (
        f"{objective or m.objective}x", focus, f"pp{m.photo_pause:g}", f"mp{m.move_pause:g}", f"ex{round(m.exposure)}",
        f"sc{int(bool(m.still_check))}", f"rf{int(bool(m.refocus))}", m.frame_format,
        f"jpg{int(bool(getattr(m, 'save_jpg', False)))}"))


def prior(m, focus, objective=None):
    """Оценка по физике: (подготовка T0, с; секунд на кадр). objective — для выделения другого объектива
    (кадр на образце меньше во столько раз, во сколько больше увеличение)."""
    from .config import scan_overlap
    o = objective or m.objective
    st = m._frame_step((1 - scan_overlap(o)) * m.objective / o)
    travel = st['x'] / max(1.0, m.motor_speed) + m.move_pause
    per = travel + m.photo_pause + 2 * m.exposure / 1000 + 0.3 + (0.15 if m.still_check else 0.0)
    if focus == 'points':
        per += m.move_pause
    elif focus == 'each':
        n = 2 * m.focus_range / max(1.0, m.focus_step) + 1
        per += n * (m.focus_step / max(1.0, m.motor_speed) + m.move_pause + 2 * m.exposure / 1000)
    elif focus == 'plane':
        n = 2 * m.focus_range / max(1.0, m.focus_step) + 1
        per += n * (m.move_pause + 2 * m.exposure / 1000) / max(1, m.focus_every)
    t0 = 3.0 + (m.light_pause if m.light_pause else 0.0)
    return t0, per


def _store(m):
    mem = getattr(m, '_mem', None)
    if mem is not None:
        return mem.data.setdefault('timing', {}), mem
    return m.__dict__.setdefault('_timing', {}), None


def _fit(runs):
    """T = T0 + t·N по точкам (N, T); при одном N — только t (T0 = 0)."""
    Ns = [n for n, _ in runs]
    if len(set(Ns)) < 2:
        return 0.0, sum(t for _, t in runs) / max(1, sum(Ns))
    n_ = len(runs)
    mn, mt = sum(Ns) / n_, sum(t for _, t in runs) / n_
    sxx = sum((n - mn) ** 2 for n in Ns)
    sxy = sum((n - mn) * (t - mt) for n, t in runs)
    t = max(0.0, sxy / sxx)
    t0 = max(0.0, mt - t * mn)
    return t0, t


def estimate(m, frames, focus='points', objective=None):
    """Ожидаемое время скана, с, и откуда оценка: (секунды, текст). objective — объектив выделения
    (по умолчанию текущий): своя физика и свои прошлые сканы."""
    p0, pt = prior(m, focus, objective)
    runs = _store(m)[0].get(key(m, focus, objective), [])
    if not runs:
        return p0 + pt * frames, "estimate (no scans with these settings yet)"
    l0, lt = _fit(runs)
    w = len(runs) / (len(runs) + 2.0)               # 1 скан — 33 % выученного, 4 — 67 %, 10 — 83 %
    sec = (1 - w) * (p0 + pt * frames) + w * (l0 + lt * frames)
    return sec, f"learned from {len(runs)} scan{'s' if len(runs) > 1 else ''} on this microscope"


def learn(m, focus, frames, seconds):
    """Запомнить, сколько на самом деле шёл скан (только законченный, кадров > 0)."""
    if frames <= 0 or seconds <= 0:
        return
    data, mem = _store(m)
    runs = data.setdefault(key(m, focus), [])
    runs.append([int(frames), round(float(seconds), 1)])
    del runs[:-KEEP]
    if mem is not None:
        try:
            mem.save()
        except OSError:
            pass


def fmt(sec):
    """Время для людей: «~40 s», «~12 min», «~1 h 20 min»."""
    if sec < 90:
        return f"~{max(1, int(round(sec)))} s"
    mins = int(math.ceil(sec / 60))
    return f"~{mins} min" if mins < 90 else f"~{mins // 60} h {mins % 60} min"
