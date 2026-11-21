"""G-код MicroControl — диалект в духе Marlin (3D-принтеры). Единицы — микрометры (мкм).

Движение
  G0 / G1 X Y Z      поехать (G90 — в точку, G91 — на столько); F (скорость) нет: она задана контроллером
  G53 G0 X Y Z       поехать в настоящих (машинных) координатах, без учёта G92
  G90 / G91          абсолютные / относительные координаты (по умолчанию G90)
  G92 X Y Z          сдвинуть ноль: текущая точка получает эти координаты (столик не двигается)
  G92.1              убрать сдвиг нуля
  G28 [X] [Y] [Z]    найти концевики, поставить там 0 (без букв — X и Y; экспериментально)
  G4 P<мс> | S<с>    пауза
Фокус
  G30                автофокус в текущей точке
  G29                плоскость фокуса: автофокус в сетке точек зоны, дальше Z берётся с плоскости
  M420 S1 / S0       плоскость фокуса вкл / выкл
Камера, свет, лазер
  M240 [R<строка> C<столбец>]   снимок (при записи скана — в скан, иначе в photos/)
  M355 S1 [P<0..5>] / M355 S0   свет вкл (яркость) / выкл
  M3 S<мВт> / M5                лазер вкл (мощность) / выкл; нужен allow_laser=True
  M114                          напечатать положение
Комментарии: ; до конца строки и (в скобках).
"""
import os, re, time
from datetime import datetime
import numpy as np
import cv2
from . import log as L, focus as F
from .errors import GcodeError, SafetyError, CameraError, LimitError, FocusError
from .util import Result, Progress

SPEC = {   # команда: (разрешённые буквы, смысл)
    'G0': ('XYZ', "Move"), 'G1': ('XYZ', "Move"), 'G4': ('PS', "Dwell"),
    'G28': ('XYZ', "Home"), 'G29': ('', "Focus plane"), 'G30': ('', "Autofocus"),
    'G90': ('', "Absolute coordinates"), 'G91': ('', "Relative coordinates"), 'G92': ('XYZ', "Offset"),
    'G92.1': ('', "Clear offset"),
    'M3': ('S', "Laser on"), 'M5': ('', "Laser off"), 'M114': ('', "Report position"),
    'M240': ('RC', "Picture"), 'M355': ('SP', "Light"), 'M420': ('S', "Focus plane on/off"),
}
TOKEN = re.compile(r'([A-Z])\s*([-+]?(?:\d+\.?\d*|\.\d+))')


class Program(str):
    """Текст G-кода. Печатается построчно; .save('файл.gcode') — сохранить."""
    def __repr__(self):
        lines = self.splitlines()
        head = "\n".join(lines[:30])
        return head + (f"\n... ({len(lines)} lines in total)" if len(lines) > 30 else "")

    def _repr_pretty_(self, p, cycle):
        p.text(repr(self))

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self)
        return path


def parse(text):
    """Разобрать программу: список (номер строки, команда, {буква: число}). Ошибка — GcodeError."""
    out = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = re.sub(r'\(.*?\)', ' ', raw.split(';', 1)[0]).strip().upper()
        if not line:
            continue
        toks = TOKEN.findall(line)
        if re.sub(r'\s+', '', line) != ''.join(a + b for a, b in toks):
            raise GcodeError('E901', f"Line {n}: '{raw.strip()}'.")
        if toks and toks[0][0] == 'N':                       # номер строки N.. допускается
            toks = toks[1:]
        machine = False
        if toks and toks[0][0] == 'G' and float(toks[0][1]) == 53:   # G53 G0 ... — машинные координаты
            machine = True
            toks = toks[1:]
            if not toks or toks[0][0] != 'G' or float(toks[0][1]) not in (0, 1):
                raise GcodeError('E901', f"Line {n}: G53 must be followed by G0 or G1: '{raw.strip()}'.")
        if not toks or toks[0][0] not in 'GM':
            raise GcodeError('E901', f"Line {n}: A line must start with a G or M command: '{raw.strip()}'.")
        num = float(toks[0][1])
        cmd = toks[0][0] + (str(int(num)) if num == int(num) else f"{num:g}")
        if cmd not in SPEC:
            raise GcodeError('E902', f"Line {n}: {cmd}. Supported: {', '.join(SPEC)}.")
        params = {}
        for a, v in toks[1:]:
            if a == 'F':
                raise GcodeError('E902', f"Line {n}: F (speed) is not supported, the controller has a fixed speed.")
            if a not in SPEC[cmd][0] or a in params:
                raise GcodeError('E901', f"Line {n}: Letter {a} is not allowed in {cmd} ({SPEC[cmd][1]}).")
            params[a] = float(v)
        if machine:
            params['_machine'] = 1.0
        out.append((n, cmd, params))
    return out


def _preflight(m, cmds, area):
    """Проверить всю программу до движения: пределы, лазер, свет, паузы, камера, зона."""
    pos = {a: v for a, v in zip('xyz', m.real_position)}       # машинные координаты, мкм
    off = dict(m._off)
    known = {'x': True, 'y': True, 'z': True}
    absolute = True
    m._need_certain()
    for n, c, p in cmds:
        try:
            if c in ('G0', 'G1'):
                new = dict(pos)
                for a in 'xyz':
                    if a.upper() in p:
                        v = p[a.upper()]
                        new[a] = v if p.get('_machine') else (v + off[a] if absolute else pos[a] + v)
                t = {a: m._u2p(new[a]) for a in 'xyz' if a.upper() in p and known[a]}
                if t:
                    m._check(t)
                    s = m.motor_step
                    cur = {a: m._u2p(pos[a]) for a in 'xyz'}
                    dxy = ((t.get('x', cur['x']) - cur['x']) ** 2 + (t.get('y', cur['y']) - cur['y']) ** 2) ** 0.5 * s
                    dz = (t.get('z', cur['z']) - cur['z']) * s
                    if known['x'] and known['y'] and dxy > m.max_jump + 1e-6:
                        raise SafetyError('E210', f"{dxy:.0f} um in X/Y at once, the limit is {m.max_jump:g} um (m.max_jump).")
                    if known['z'] and (dz > m.max_move_z + 1e-6 or -dz > m.max_jump + 1e-6):
                        raise SafetyError('E210', f"Z by {dz:+.0f} um at once, the limits are +{m.max_move_z:g} / -{m.max_jump:g} um.")
                pos = new
            elif c == 'G90':
                absolute = True
            elif c == 'G91':
                absolute = False
            elif c == 'G92':
                for a in 'xyz':
                    if a.upper() in p:
                        off[a] = pos[a] - p[a.upper()]
            elif c == 'G92.1':
                off = dict.fromkeys('xyz', 0.0)
            elif c == 'G28':
                if not m.allow_homing:
                    raise SafetyError('E209', "Homing is locked. Connect with Microscope(..., allow_homing=True).")
                for a in ([x.lower() for x in p] or ['x', 'y']):
                    known[a] = False
                known['z'] = False
            elif c in ('G29', 'G30'):
                known['z'] = False
                if c == 'G29' and area is not None:            # G29 заканчивает в первом кадре зоны
                    pos['x'], pos['y'] = area.x0 + off['x'], area.y0 + off['y']
                if not m.cam:
                    raise CameraError('E503', "Autofocus needs the camera.")
                if c == 'G29' and area is None:
                    raise SafetyError('E701', "G29 needs an area: mark it in m.manual() (keys 1 and 2).")
            elif c == 'G4':
                s = p.get('S', p.get('P', 0) / 1000)
                if not 0 <= s <= 3600:
                    raise SafetyError('E206', f"Dwell {s:g} s, allowed 0 .. 3600 s.")
            elif c == 'M3':
                type(m).laser.check(m, p.get('S', 0))
                if p.get('S', 0) and not m.allow_laser:
                    raise SafetyError('E401', "Connect with Microscope(..., allow_laser=True).")
            elif c == 'M355':
                if p.get('S', 1) not in (0, 1) or not 0 <= p.get('P', 5) <= 5 or p.get('P', 5) != int(p.get('P', 5)):
                    raise SafetyError('E208', "M355 S0 or M355 S1 P0..5.")
            elif c == 'M420':
                if p.get('S', 1) not in (0, 1):
                    raise SafetyError('E208', "M420 S0 or M420 S1.")
            elif c == 'M240' and not m.cam:
                raise CameraError('E503', "M240 needs the camera.")
        except (SafetyError, CameraError, LimitError) as e:
            if e.code == 'E211':
                raise
            raise type(e)(e.code, f"G-code line {n} ({c}). {e.detail}") from None


def focus_plane(m, area):
    """G29: автофокус в сетке точек зоны (через focus_every кадров), плоскость z = a + b·x + c·y."""
    step = m._frame_step()
    xs, ys = _points(area.x0, area.x1, step['x'] * m.focus_every), _points(area.y0, area.y1, step['y'] * m.focus_every)
    pts, weak = [], []
    m._plane = None
    pr = Progress(len(xs) * len(ys), "Focus plane")
    for j, y in enumerate(ys):
        for x in (xs if j % 2 == 0 else xs[::-1]):
            m._travel(x=x, y=y)
            f = m.focus()
            (weak if 'Note' in f else pts).append(tuple(m.real_position))   # плоскость — в машинных координатах
            pr.update(1, f"Z {m.position.z:.0f} um")
    pr.close()
    m._travel(x=area.x0, y=area.y0)                     # вернуться к первому кадру зоны
    if not pts:                                          # нигде нет деталей — плоскость строить не по чему
        raise FocusError('E603', f"All {len(weak)} focus points had no clear focus peak. "
                                 "Add light, check the sample, or use focus='each' / 'off'.")
    if len(pts) < 3 and weak:
        L.event('AF-002', f"Only {len(pts)} good focus points of {len(pts) + len(weak)}: plane is flat (no tilt).", 30)
    P = np.array(pts, float)
    if len({(a, b) for a, b, _ in pts}) >= 3 and np.linalg.matrix_rank(np.c_[np.ones(len(P)), P[:, :2]]) == 3:
        coef = np.linalg.lstsq(np.c_[np.ones(len(P)), P[:, :2]], P[:, 2], rcond=None)[0]
    else:
        coef = np.array([P[:, 2].mean(), 0.0, 0.0])
    res = P[:, 2] - np.c_[np.ones(len(P)), P[:, :2]] @ coef
    m._plane = coef
    tilt = np.degrees(np.arctan(np.hypot(coef[1], coef[2])))
    L.event('AF-002', f"Focus plane from {len(P)} points" + (f" ({len(weak)} skipped: no detail)" if weak else "") + f": tilt {tilt:.3f} deg, largest deviation {np.abs(res).max():.1f} um.")
    r = Result("Focus plane", Points=len(P), Tilt=f"{tilt:.3f} deg",
               **{"Largest deviation": f"{np.abs(res).max():.1f} um"})
    if weak:
        r['Skipped'] = f"{len(weak)} points without detail"
    if np.abs(res).max() > m.focus_step:
        r['Note'] = "The sample is not flat. Consider focus='each'."
    return r


def _points(a, b, step):
    n = max(1, int(np.ceil((b - a) / step - 0.05)) + 1)
    return list(np.linspace(a, b, n)) if n > 1 else [(a + b) / 2]


class _Recorder:
    """Запись скана: каждый M240 — кадр в ScanData + проверка резкости и перефокус мутных кадров."""
    def __init__(self, m, name, program, notes, area, folder=None, extra=None):
        from .storage import ScanData
        from . import __version__, AUTHORS
        os.makedirs(m.dirs['scans'], exist_ok=True)
        free = __import__('shutil').disk_usage(m.dirs['scans']).free / 2 ** 20
        n = program.count('M240')
        need = n * m._frame_px[0] * m._frame_px[1] * 3 * m.frame_scale ** 2 * (0.7 if m.frame_format == 'h5' else 0.08) / 2 ** 20
        if free < 1.5 * need + 500:
            raise SafetyError('E702', f"Need about {need:.0f} MB, free {free:.0f} MB.")
        safe = re.sub(r'[^\w\-]+', '_', name).strip('_') or 'scan'
        self.folder = folder or os.path.join(m.dirs['scans'], datetime.now().strftime("%Y-%m-%d_%H%M%S") + f"_{m.objective}x_{safe}")
        if hasattr(m.cam, 'freeze'):
            m.cam.freeze()                                       # одна экспозиция на весь скан
        w, h = m.frame_size
        self.meta = dict(project="MicroControl", version=__version__, authors=AUTHORS, sample=name, notes=notes,
                         date=datetime.now().isoformat(timespec='seconds'), objective=m.objective,
                         pixel_size_um=m.pixel_size / m.frame_scale, fov_um={'x': w, 'y': h}, motor_step_um=m.motor_step,
                         um_per_pulse={'x': m.motor_step}, A=(m._pixels_per_step() * m.frame_scale).tolist(),
                         storage=dict(format=m.frame_format, scale=m.frame_scale, jpg_quality=m.jpg_quality),
                         area=None if area is None else [area.x0, area.y0, area.x1, area.y1],
                         camera=dict(m.cam.info(), exposure_ms=m.exposure), light_level=m.light,
                         focus=dict(metric=m.focus_method, range_um=m.focus_range, step_um=m.focus_step,
                                    every_frames=m.focus_every, refocus=m.refocus),
                         photo_pause_s=m.photo_pause, move_pause_s=m.move_pause, speed_um_s=m.motor_speed, references=dict(m.references),
                         offset_um=dict(m._off), device=dict(getattr(m, 'device', {}) or {}),
                         frames_planned=n, status='running', log=L.path(),
                         crosses=m._crosses.to_dict() if getattr(m, '_crosses', None) is not None and m._crosses.marks else None,
                         **(extra or {}))
        self.store = ScanData.create(self.folder, self.meta, batch_size=m.batch_size, fmt=m.frame_format,
                                     quality=m.jpg_quality)
        self.jpg = os.path.join(self.folder, 'jpg') if (m.save_jpg and m.frame_format == 'h5') else None
        if self.jpg:
            os.makedirs(self.jpg, exist_ok=True)
        cr = getattr(m, '_crosses', None)
        if cr is not None and getattr(cr, 'template', None) is not None:     # образец креста — для join по крестам
            ok, buf = cv2.imencode('.png', cr.template)
            buf.tofile(os.path.join(self.folder, 'cross_template.png'))
            self.meta['crosses']['template'] = dict(file='cross_template.png', pixel_size_um=cr.template_px)
            self.store.save_meta()
        with open(os.path.join(self.folder, 'program.gcode'), 'w', encoding='utf-8') as f:
            f.write(program)
        self.m, self.sharp, self.t0, self.n, self.refocused, self.blurry = m, [], time.monotonic(), 0, 0, 0

    def picture(self, row, col):
        m = self.m
        img = m.image()
        if m.still_check:                                       # столик ещё дрожит — два кадра подряд разные
            for _ in range(3):
                img2 = m.image(0)
                a = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img, None, fx=0.25, fy=0.25,
                               interpolation=cv2.INTER_AREA).astype(np.float32)
                b = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2, None, fx=0.25, fy=0.25,
                               interpolation=cv2.INTER_AREA).astype(np.float32)
                (sx, sy), resp = cv2.phaseCorrelate(a, b)
                if np.hypot(sx, sy) < 0.25 or resp < 0.05:     # < 1 px полного кадра — стоит
                    img = img2
                    break
                L.event('SCN-003', f"Frame {self.n + 1}: the picture still moves ({np.hypot(sx, sy) * 4:.1f} px), waiting.", 20)
                time.sleep(max(0.1, m.photo_pause))
                img = m.image()
        s = F.sharpness(img, m.focus_method)
        ref = np.median(self.sharp[-20:]) if len(self.sharp) >= 5 else None
        refocused = blurry = 0
        if ref is not None and s < 0.5 * ref:
            if m.refocus:                                       # мутный кадр: фокус и переснять
                L.event('SCN-002', f"Frame {self.n + 1} is blurry ({s:.0f} vs {ref:.0f}). Refocusing.")
                m.focus()
                time.sleep(m.photo_pause)
                img = m.image()
                s = F.sharpness(img, m.focus_method)
                refocused = 1
            blurry = int(s < 0.5 * ref)
        self.sharp.append(s)
        p = m.position
        k = m.frame_scale
        stored = img if k >= 1 else cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        self.store.add(stored, row='' if row is None else row, col='' if col is None else col,
                       x_um=p.x, y_um=p.y, z_um=p.z, x_p=m._p['x'], y_p=m._p['y'], z_p=m._p['z'],
                       sharpness=round(s, 1), refocused=refocused, blurry=blurry,
                       t_s=round(time.monotonic() - self.t0, 1))
        self.n += 1
        if self.jpg:                                            # сразу на диск: можно смотреть во время скана
            try:
                k = m.jpg_scale
                small = img if k >= 1 else cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
                rc = f"_r{row + 1:03d}_c{col + 1:03d}" if row not in (None, '') else ''
                ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 90])
                buf.tofile(os.path.join(self.jpg, f"{self.n:05d}{rc}.jpg"))
            except Exception as e:
                L.event('STO-005', f"JPG of frame {self.n} not saved: {e}", 30)
        self.refocused += refocused
        self.blurry += blurry
        L.event('SCN-001', f"Frame {self.n} at {p}, sharpness {s:.0f}" + (", refocused" if refocused else "") +
                (", still blurry" if blurry else "") + ".")

    def close(self, status):
        self.meta.update(status=status, frames=self.n, refocused=self.refocused, blurry=self.blurry,
                         duration_s=round(time.monotonic() - self.t0, 1))
        self.store.meta = self.meta
        self.store.save_meta()
        self.store.close()


def run(m, program, save=None, notes='', folder=None, extra=None):
    if isinstance(program, str) and '\n' not in program and program.lower().endswith(('.gcode', '.nc', '.txt')) \
            and os.path.exists(program):
        with open(program, encoding='utf-8') as f:
            program = f.read()
    program = Program(program)
    cmds = parse(program)
    area = m.area
    _preflight(m, cmds, area)
    rec = _Recorder(m, save, program, notes, area, folder, extra) if save else None
    L.event('GC-001', f"Program of {len(cmds)} commands" + (f", recording to {rec.folder}" if rec else "") + ".")
    absolute, status, done, t0 = True, 'done', 0, time.monotonic()
    pr = Progress(len(cmds), "Scan" if rec else "G-code") if (rec or len(cmds) > 20) else None
    report = []
    try:
        for n, c, p in cmds:
            L.debug('GC-002', f"{n}: {'G53 ' if p.get('_machine') else ''}{c} " + " ".join(f"{k}{v:g}" for k, v in p.items() if k != '_machine'))
            if c in ('G0', 'G1'):
                cur = dict(zip('xyz', m.position))
                if p.get('_machine'):
                    t = {a: p[a.upper()] - m._off[a] for a in 'xyz' if a.upper() in p}
                else:
                    t = {a: (p[a.upper()] if absolute else cur[a] + p[a.upper()]) for a in 'xyz' if a.upper() in p}
                m.goto(**t)
            elif c == 'G90':
                absolute = True
            elif c == 'G91':
                absolute = False
            elif c == 'G92':
                cur = dict(zip('xyz', m.position))
                m.offset(**{a: p.get(a.upper(), cur[a]) for a in 'xyz'})
            elif c == 'G92.1':
                m.offset_clear()
            elif c == 'G28':
                axes = {a.lower(): True for a in p} or {'x': True, 'y': True}
                m.find_limits(x=axes.get('x', False), y=axes.get('y', False), z=axes.get('z', False))
            elif c == 'G4':
                m._sleep(p.get('S', p.get('P', 0) / 1000))
            elif c == 'G30':
                m.focus()
            elif c == 'G29':
                print(focus_plane(m, area))
            elif c == 'M420':
                if p.get('S', 1) == 0:
                    m._plane_off, m._plane = m._plane, None
                else:
                    m._plane = getattr(m, '_plane_off', None) if m._plane is None else m._plane
            elif c == 'M3':
                m.laser = p.get('S', 0)
            elif c == 'M5':
                m.laser = 0
            elif c == 'M355':
                m.light = int(p.get('P', 5)) if p.get('S', 1) else 0
            elif c == 'M114':
                report.append(str(m.position))
                print(f"Position  {m.position}")
            elif c == 'M240':
                if rec:
                    rec.picture(int(p['R']) if 'R' in p else None, int(p['C']) if 'C' in p else None)
                else:
                    m.photo()
            done += 1
            if pr:
                pr.update(1, f"Frames {rec.n}" if rec else '')
    except KeyboardInterrupt:
        status = f"stopped at line {cmds[done][0] if done < len(cmds) else '-'}"
    except Exception as e:
        status = f"error at line {cmds[done][0]}: {e}"
        raise
    finally:
        if pr:
            pr.close()
        if rec:
            rec.close(status)
            m.last_scan = rec.folder
        L.event('GC-003', f"Program {status}. {done} of {len(cmds)} commands, {time.monotonic() - t0:.1f} s.")
    r = Result("Program finished" if status == 'done' else "Program stopped",
               Commands=f"{done} of {len(cmds)}", Time=f"{(time.monotonic() - t0) / 60:.1f} min")
    if status != 'done':
        r['Note'] = "The stage stays where it is."
    if rec:
        r.update(Frames=rec.n, Refocused=rec.refocused, Blurry=rec.blurry, Folder=rec.folder)
    return r
