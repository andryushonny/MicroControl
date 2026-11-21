"""Зона скана (Area) и программа скана в G-коде.

Зона задаётся двумя углами — центрами первого и последнего кадра (клавиши 1 и 2 в m.manual()).
Шаг между кадрами = кадр × (1 − overlap), округлённый вниз до шага мотора: кадры встают
встык без щелей (перекрытие — несколько пикселей).
"""
import math
import numpy as np
from .errors import SafetyError
from .gcode import Program


class Area:
    """Зона скана в мкм: углы (x0, y0) и (x1, y1) — центры крайних кадров.
    Обычно задаётся в m.manual() клавишами 1 и 2; вручную: mc.Area(0, 0, 2000, 1500)."""

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.x1 = sorted((float(x0), float(x1)))
        self.y0, self.y1 = sorted((float(y0), float(y1)))

    def grid(self, step, path='snake'):
        """Центры кадров: список (строка, столбец, x, y) в порядке обхода. step — {'x':, 'y':} мкм."""
        nx = max(1, math.ceil((self.x1 - self.x0) / step['x'] - 0.05) + 1)   # допуск на округление
        ny = max(1, math.ceil((self.y1 - self.y0) / step['y'] - 0.05) + 1)
        # кадры ровно от угла до угла: шаг подогнан (не больше заданного), последний кадр — точно в углу B
        sx = (self.x1 - self.x0) / (nx - 1) if nx > 1 else step['x']
        sy = (self.y1 - self.y0) / (ny - 1) if ny > 1 else step['y']
        out = []
        for r in range(ny):
            cols = range(nx) if path == 'rows' or r % 2 == 0 else range(nx - 1, -1, -1)
            out += [(r, c, self.x0 + c * sx, self.y0 + r * sy) for c in cols]
        return out

    def size(self, step):
        """Число кадров (по X, по Y)."""
        g = self.grid(step)
        return max(t[1] for t in g) + 1, max(t[0] for t in g) + 1

    def __repr__(self):
        return f"X {self.x0:.0f} .. {self.x1:.0f}, Y {self.y0:.0f} .. {self.y1:.0f} um"


def focus_z(pts, X, Y, k=4):
    """Z в точке (X, Y) по точкам фокуса (настоящие мкм): k ближайших, веса 1/d² (в самой точке — её Z)."""
    d = np.array([math.hypot(p['x'] - X, p['y'] - Y) for p in pts])
    z = np.array([p['z'] for p in pts])
    if d.min() < 1.0:
        return float(z[d.argmin()])
    near = np.argsort(d)[:k]
    w = 1 / d[near] ** 2
    return float((w * z[near]).sum() / w.sum())


def plan(m, area=None, focus='auto', overlap=None, path='snake', pts=None):
    """Программа скана в G-коде."""
    area = area or m.area
    if area is None:
        raise SafetyError('E701', "Mark two corners in m.manual() (keys 1 and 2) or pass area=mc.Area(x0, y0, x1, y1).")
    pts = (getattr(m, '_focus_pts', None) or []) if pts is None else pts
    if focus == 'auto':                    # точки фокуса из m.manual() и галочка «по углам» — Z по ним
        focus = 'points' if (pts and getattr(m, '_use_fp', True)) else 'off'
    if focus not in ('plane', 'each', 'off', 'points'):
        raise SafetyError('E703', f"focus = {focus!r}. Allowed: 'auto', 'points', 'plane', 'each', 'off'.")
    if focus == 'points' and not pts:
        raise SafetyError('E703', "No focus points: in m.manual() keys 1, 2 (corners) and F keep Z.")
    if overlap is None:
        from .config import scan_overlap
        overlap = scan_overlap(m.objective)
    if isinstance(overlap, bool) or not isinstance(overlap, (int, float)) or not 0 <= overlap <= 0.5:
        raise SafetyError('E703', f"overlap = {overlap!r}. Allowed: 0 .. 0.5.")
    if path not in ('snake', 'rows'):
        raise SafetyError('E703', f"path = {path!r}. Allowed: 'snake', 'rows'.")
    m.area = area
    step = m._frame_step(1 - overlap)
    w0, h0 = m.frame_size
    if step['x'] > m.max_jump or step['y'] > m.max_jump:
        raise SafetyError('E703', f"One frame step is longer than m.max_jump ({m.max_jump:g} um).")
    tiles = area.grid(step, path)
    nx, ny = area.size(step)
    w, h = m.frame_size
    L = [f"; MicroControl scan program",
         f"; Objective {m.objective}x, pixel {m.pixel_size:g} um, frame {w:.0f} x {h:.0f} um",
         f"; Area {area}: {nx} x {ny} = {len(tiles)} frames, step {step['x']:.1f} x {step['y']:.1f} um",
         f"; Focus: {focus}" + (f" (points every {m.focus_every} frames)" if focus == 'plane' else "") +
         (f" (Z from {len(pts)} focus points, nearest 4, weights 1/d^2)" if focus == 'points' else "") +
         f", blurry frames re-taken: {'yes' if m.refocus else 'no'}",
         "G90                     ; absolute coordinates, um",
         f"M355 S1 P{m.light or 5}             ; light on"]
    # переезды пишутся в настоящих координатах (G53): смена нуля после plan() не сдвигает съёмку;
    # длинные переезды делятся на отрезки не длиннее max_jump
    ox, oy = m._off['x'], m._off['y']
    cur = (m.real_position.x, m.real_position.y)

    def go(x, y, note=''):
        nonlocal cur
        X, Y = x + ox, y + oy
        k = max(1, math.ceil(math.hypot(X - cur[0], Y - cur[1]) / (m.max_jump * 0.95)))
        for i in range(1, k + 1):
            px, py = cur[0] + (X - cur[0]) * i / k, cur[1] + (Y - cur[1]) * i / k
            L.append(f"G53 G0 X{px:.1f} Y{py:.1f}" + (f"    ; {note}" if note and i == k else ""))
        cur = (X, Y)

    if focus == 'plane':
        go(area.x0, area.y0, "to the first frame")
        L.append("G29                     ; focus plane over the area")
        cur = (area.x0 + ox, area.y0 + oy)             # G29 заканчивает в первом кадре зоны
    else:
        L.append("M420 S0                 ; no focus plane")
    for r, c, x, y in tiles:
        go(x, y)
        if focus == 'points':
            L.append(f"G53 G0 Z{focus_z(pts, x + ox, y + oy):.1f}")
        if focus == 'each':
            L.append("G30                     ; autofocus")
        if m.photo_pause:
            L.append(f"G4 S{m.photo_pause:g}")
        L.append(f"M240 R{r} C{c}")
    go(area.x0, area.y0, "back to the first frame")
    return Program("\n".join(L) + "\n")
