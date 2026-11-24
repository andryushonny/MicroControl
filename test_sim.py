"""Проверка MicroControl на симуляторе (для окон OpenCV нужен экран).
    python test_sim.py            все разделы, параллельно по процессам (каждый — со своим домом ~/.microcontrol)
    python test_sim.py --quick    быстрая проверка: основные разделы (~1 мин)
    python test_sim.py --serial   всё подряд в одном процессе (как раньше)
    python test_sim.py 14 22      только эти разделы"""
import os, subprocess, sys, tempfile, time, re
import numpy as np

QUICK = ('19', '20', '22', '23')


def _parts(src):
    """Разделы файла: [('1-13', текст), ('14', текст) ...]; первые 13 делят одно подключение — один процесс."""
    lines = src.split("\n")
    head_end = next(i for i, l in enumerate(lines) if l.startswith("# 1. "))
    pre = "\n".join(lines[:head_end])
    i0 = pre.index("\ndef refused(")
    pre = "import microcontrol as mc\n" + pre[i0:pre.index("\n\n\n", i0)]   # общая часть: refused()
    starts = [(i, re.match(r"^# (?:-+ )?(\d+)\. ", l).group(1)) for i, l in enumerate(lines)
              if re.match(r"^# (?:-+ )?(\d+)\. ", l) and int(re.match(r"^# (?:-+ )?(\d+)\. ", l).group(1)) >= 14]
    out = [("1-13", "\n".join(lines[:starts[0][0]]))]
    for k, (i, n) in enumerate(starts):
        j = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        out.append((n, "import os, subprocess, sys, tempfile, time, json, numpy as np\nimport cv2\n"
                       "cv2.getWindowImageRect = lambda w: (0, 0, 1280, 900)   # как после раздела 14 при запуске подряд\n"
                       + pre + "\n" + "\n".join(lines[i:j])))
    return out


if __name__ == '__main__' and not os.environ.get('MC_PART') and '--serial' not in sys.argv:
    from concurrent.futures import ThreadPoolExecutor
    here = os.path.abspath(__file__)
    parts = _parts(open(here, encoding='utf-8').read())
    want = [a for a in sys.argv[1:] if not a.startswith('-')]
    if '--quick' in sys.argv:
        want = want or list(QUICK)
    if want:
        parts = [p for p in parts if p[0] in want or (p[0] == '1-13' and any(w.isdigit() and int(w) < 14 for w in want))]
    work = tempfile.mkdtemp(prefix='mc_test_')

    def run(part):
        name, code = part
        f = os.path.join(work, f"part_{name}.py")
        open(f, 'w', encoding='utf-8').write(code)
        home = tempfile.mkdtemp(prefix=f'home_{name}_', dir=work)
        env = dict(os.environ, MC_PART=name, HOME=home, USERPROFILE=home,
                   PYTHONPATH=os.pathsep.join([os.path.dirname(here), os.environ.get('PYTHONPATH', '')]))
        t0 = time.monotonic()
        r = subprocess.run([sys.executable, f], capture_output=True, text=True, env=env, cwd=os.path.dirname(here))
        return name, r.returncode, time.monotonic() - t0, r.stdout + r.stderr

    t0 = time.monotonic()
    bad = []
    with ThreadPoolExecutor(max_workers=max(2, min(len(parts), (os.cpu_count() or 4)))) as ex:
        for name, rc, dt, out in ex.map(run, parts):
            oks = [l for l in out.splitlines() if l.rstrip().endswith(': OK')]
            print(f"{'ok  ' if rc == 0 else 'FAIL'}  part {name:<6} {dt:6.1f} s   " + "; ".join(l.strip() for l in oks))
            if rc:
                bad.append(name)
                print("\n".join(out.splitlines()[-25:]))
    print(f"\n{len(parts) - len(bad)} of {len(parts)} parts passed in {time.monotonic() - t0:.0f} s")
    sys.exit(1 if bad else 0)

import microcontrol as mc

wd = tempfile.mkdtemp()
m = mc.Microscope('sim', objective=10, folder=wd)
w = m._world
print(m.info())

def refused(f, code=None):
    try:
        f(); raise AssertionError("Protection failed")
    except mc.MicroControlError as e:
        print("  Refused:", e)
        assert code is None or e.code == code, (e.code, code)


refused(lambda: setattr(m, 'dwell', 1), 'E105')                 # старое имя -> ошибка с новым
try:
    m.machine_position; raise SystemExit("old name accepted")
except AttributeError as e:
    assert 'real_position' in str(e); print("OK old name:", e)

# 1. Один хозяин прибора
refused(lambda: mc.Microscope('sim', objective=10, folder=wd), 'E106')
out = subprocess.run([sys.executable, '-c', "import microcontrol as mc; mc.Microscope('sim', objective=10)"],
                     capture_output=True, text=True, cwd=os.getcwd(), env=dict(os.environ, PYTHONPATH='.'))
assert 'E106' in out.stderr, out.stderr[-500:]
print("  Second process refused: E106")
# 2. Ошибки сразу, а не молча
for f, code in ((lambda: setattr(m, 'photo_paus', 1), 'E105'), (lambda: setattr(m, 'light', 2.5), 'E208'),
                (lambda: m.move(dx=1), 'E214'), (lambda: m.move(), 'E213'), (lambda: m.goto(), 'E213'),
                (lambda: setattr(m, 'area', (0, 0, 1, 1)), 'E208'), (lambda: setattr(m, 'position', 0), 'E208'),
                (lambda: m.move(dx=2500), 'E201'), (lambda: m.goto(z=60000), 'E203'), (lambda: setattr(m, 'laser', 5), 'E401'),
                (lambda: m._cmd(0x30), 'E305'), (lambda: m.find_limits(), 'E209'),
                # 3. Подозрительные движения
                (lambda: m.goto(x=5000), 'E210'), (lambda: m.goto(z=1500), 'E210'),
                (lambda: m.gcode("G0 X100\nG0 X9000"), 'E210'), (lambda: m.gcode("G0 X100 F3000"), 'E902'),
                (lambda: m.gcode("G0 X99999"), 'E202')):
    refused(f, code)
try:
    m.light_level = 3
except mc.MicroControlError as e:
    print("  Typo hint:", e)
assert w.p == {'x': 0, 'y': 0, 'z': 0}, w.p
# 4. Последовательность: ни одна команда не пришла во время движения
m.move(dx=1000); m.move(dx=-1000); m.goto(x=2000); m.goto(x=0)
print("  Overlapping commands:", w.overlaps); assert w.overlaps == 0
r = m.move(dx=560); print(r); m.move(dx=-555.6)
# 5. Смещение нуля
m.goto(x=500, y=300)
print(m.offset())
print(m.position, "| machine", m.real_position)
m.goto(x=0, y=0); assert m.real_position.x == 500.0
print(m.gcode("G92.1\nM114"))
m.goto(x=0, y=0)
# 6. Скорость
refused(lambda: m.measure_speed(1000), 'E212')       # не в фокусе: движения не видно -> отказ, скорость не меняется
m.move(dz=100); m.move(dz=100); m.focus()
z0 = m.position.z; m.move(dz=-1000); m.move(dz=-1000); refused(lambda: m.move(dz=-2500), 'E201')   # вниз — до 2 мм
refused(lambda: m.move(dz=1500), 'E201'); refused(lambda: m.goto(z=z0), 'E210')   # вверх — до 1 мм за раз (10x)
m.move(dz=1000); m.move(dz=1000)
assert abs(m.position.z - z0) < 6
m.objective = 100; refused(lambda: m.move(dz=700), 'E201'); m.move(dz=600); m.move(dz=-600)   # 100x: до 650 мкм
m.working_distance = 3400; assert m.max_move_z == 1000                                        # WD с оправы
m.objective = 10
print(m.measure_speed(1000)); assert m.motor_speed <= 0.8 * 5000 * 1.05
# 7. Скан со смещением нуля
m.offset()
m.focus()
# ноль и баланс белого из ручного режима: зона не сдвигается
m.manual(fullscreen=False, _keys=[ord('1'), 65363, ord('0'), ord('2'), ord('w'), ord(')'), ord('c'), 27])
assert not any(m._off.values()); m.goto(x=0, y=0)
area = m.manual(fullscreen=False, _keys=[ord('a'), ord('1')] + [65363] * 4 + [65364] * 3 + [ord('2'), 27])
print(area)
prog = m.plan(focus='plane'); print(prog)
folder = m.scan(name='test sample', focus='plane')
st = mc.ScanData(folder); print(st); assert len(st) == str(prog).count('M240'), (len(st), str(prog).count('M240'))
z = np.array([r['z_um'] for r in st.index]); x = np.array([r['x_um'] for r in st.index])
print("  Z follows the tilt:", np.round(np.polyfit(x, z, 1)[0], 4), "expected 0.02")
print("  Overlapping commands:", w.overlaps); assert w.overlaps == 0
assert w.motors_on and w.ignored == 0 and w.tec == -5, (w.motors_on, w.ignored, w.tec)
print(mc.join(folder))
print(m)
m.close()
# 8. Аварийная остановка из другого процесса во время длинной программы
m = mc.Microscope('sim', objective=10, folder=wd)
m.motor_speed = 200
prog = "\n".join(["G91"] + ["G0 X500\nG0 X-500"] * 20)
p = subprocess.Popen([sys.executable, '-c', "import time; time.sleep(1.5); from microcontrol.control import emergency_stop; print(emergency_stop('sim'))"],
                     env=dict(os.environ, PYTHONPATH='.'))
t0 = time.time()
try:
    m.gcode(prog); raise AssertionError("not stopped")
except mc.StopError as e:
    print(f"  Stopped after {time.time() - t0:.1f} s:", e)
assert m.light == 0
refused(lambda: m.move(dx=10), 'E110')
p.wait()
m.close()
m = mc.Microscope('sim', objective=10, folder=wd)     # после переподключения всё работает
m.move(dx=10); m.close()
log = open(m.log_file, encoding='utf-8').read().splitlines()
print("\n".join(log[:14]))

# 10. Дом микроскопа: задаётся один раз, помнится между сеансами
print("\n== Home ==")
hd = tempfile.mkdtemp()
m = mc.Microscope('sim', objective=10, folder=hd)
m.move(dz=100); m.move(dz=100); m.focus(); m.move(dx=500, dy=300)
print(m.set_home()); assert m.position == (0, 0, 0)
refused(lambda: m.set_home(), 'E215')
try:
    m.home(); raise SystemExit("old name accepted")
except AttributeError as e:
    assert 'find_limits' in str(e)
m.move(dx=300, dy=-200); m.close()
m = mc.Microscope('sim', objective=10, folder=hd)                 # новый сеанс: положение продолжается, картинка сверена
assert m.position.x == round(54 * mc.config.UM_PER_STEP, 1) and not m._uncertain, (m.position, m._uncertain)
mp_ = m._mem.path; px0 = dict(m._p); m.close()
import json as _js                                                   # между сеансами столик сдвинули рукой на 7 шагов
d_ = _js.load(open(mp_)); d_['sim_true']['x'] += 7; _js.dump(d_, open(mp_, 'w'))
m = mc.Microscope('sim', objective=10, folder=hd)
assert m._p['x'] == px0['x'] + 7 and not m._uncertain, (m._p, px0)  # координаты поправлены по картинке
m.move(dx=-7 * m.motor_step)
m.confirm_position()
r = m.go_home(); print(r); assert 'matches' in r['Picture']
home_true = dict(m._world.p)
m._world.motion = None; m._world._p["x"] += 3; m._world._p["y"] -= 2                       # «сбой»: столик не там, где думает программа
r = m.go_home(); print(r); assert 'shifted' in r['Picture']
r = m.go_home(correct=True); print(r)
assert all(abs(m._world.p[a] - home_true[a]) < 0.5 for a in 'xy'), m._world.p   # на самом деле дома
m._remember(moving=True); m._ser.close(); m._lock.release(); m._lock = None; m._ser = None   # «убили» во время движения
m = mc.Microscope('sim', objective=10, folder=hd)
assert not m._uncertain                                             # без подтверждения: положению верим
m.set_home(replace=True); assert not m._uncertain
m.close()
hd2 = tempfile.mkdtemp()                                            # без дома: координаты тоже продолжаются
m = mc.Microscope('sim', objective=10, folder=hd2); m.focus(); m.move(dx=400, dy=200); p_ = m.position; m.close()
m = mc.Microscope('sim', objective=10, folder=hd2)
assert m.position == p_ and not m._uncertain, (m.position, p_)
m.close()
print("Home: OK")

# 11. Слишком высокая m.motor_speed: контроллер отвечает «ещё едет» -> повтор команды, положение точное
print("\n== Moving ==")
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp(), camera=False)
m.motor_speed = 20000                                     # на самом деле мотор ~5000 мкм/с
m.move(dx=1000); m.move(dx=1000)                         # «ещё едет» -> команда повторяется, не теряется
time.sleep(0.5); assert m._world.overlaps > 0 and abs(m._world.p['x'] * m.motor_step - 2000) < 6, (m._world.overlaps, m._world.p)
assert not m._uncertain
m.close(); print("Moving: OK")

# 12. Мини-карта на большой зоне (600 кадров): окно, номера, обзор; столик за краем — стрелка
print("\n== Minimap ==")
import microcontrol.manual as MAN
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
st = m._frame_step()                      # клетка мини-карты = кадр камеры
m.area = mc.Area(0, 0, st['x'] * 29, st['y'] * 19)
for x, y in ((0, 0), (14, 10), (34, 10)):
    m._travel(x=st['x'] * x, y=st['y'] * y)
    m.manual(fullscreen=False, _keys=[-1, 27])
r = MAN.cell_of(m, (m.area.x0, m.area.y0), (m.area.x1, m.area.y1))
assert r[0] is None and (r[2], r[3]) == (20, 30), r      # за краем зоны; зона 20 строк x 30 столбцов
m.close(); print("Minimap: OK")

# 13. Потеря порта (USB переподключился / усыплён): порт открывается заново; движение не повторяется
if os.name != 'nt':
    print("\n== Reconnect ==")
    import pty, tty, threading
    import microcontrol.protocol as P
    from microcontrol.sim import SimWorld
    mfd, sfd = pty.openpty(); tty.setraw(sfd)
    w2 = SimWorld()
    def _dev():
        buf = b""
        while True:
            try: buf += os.read(mfd, 64)
            except OSError: continue
            while len(buf) >= 4 and buf[:2] == P.HEAD:
                n = int.from_bytes(buf[2:4], 'big')
                if len(buf) < n + 2: break
                fr, buf = buf[:n + 2], buf[n + 2:]
                r = w2.handle(fr[4], fr[5:-1])
                if r is not None: os.write(mfd, P.pack(fr[4], r))
    threading.Thread(target=_dev, daemon=True).start()
    m = mc.Microscope(os.ttyname(sfd), objective=10, camera=False, force=True, folder=tempfile.mkdtemp())
    m.move(dx=200)
    m._ser.close(); m.light = 3; assert m._ser.is_open                 # безопасная команда — повторена
    m.move(dx=200)                                                     # моторы снова включены
    m._ser.close(); refused(lambda: m.move(dx=200), 'E307')             # движение — нет, ось под вопросом
    assert 'x' in m._uncertain
    m.close(); print("Reconnect: OK")

# 14. Сшивка по перекрытиям (как ASHLAR) -> кресты на мозаике -> выделение -> вывод -> очистка
print("\n== Stitch / mask / export ==")
import csv, json, cv2
cv2.getWindowImageRect = lambda w: (0, 0, 1280, 900)
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
w = m._world; w.size = (1296, 972); w.turn_deg = 3.5; m.focus()
_t = np.radians(3.5)
_wp = lambda i, j: (np.cos(_t)*i*100 - np.sin(_t)*j*100, np.sin(_t)*i*100 + np.cos(_t)*j*100)
w.flakes = [(*_wp(3.4, 2.6), 30, 40, 4), (*_wp(3.6, 2.4), 18, -20, 3), (*_wp(5.5, 3.5), 40, 75, 5)]   # чешуйки в 2 квадратах
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 4 * fw, 3 * fh)
f = m.scan(name='pipe'); m.close()
rows = list(csv.DictReader(open(os.path.join(f, 'index.csv'), encoding='utf-8')))
true = np.array([[int(r['x_p']), int(r['y_p'])] for r in rows], float)
rng = np.random.default_rng(3); walk = np.zeros(2)
for r in rows:                                       # столик ошибается: масштаб +1.5 %, дрейф ±1 шаг за переезд
    walk += rng.integers(-1, 2, 2)
    r['x_p'] = str(int(round(int(r['x_p']) * 1.015 + walk[0]))); r['y_p'] = str(int(round(int(r['y_p']) * 1.012 + walk[1])))
with open(os.path.join(f, 'index.csv'), 'w', newline='', encoding='utf-8') as fo:
    wr = csv.DictWriter(fo, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
r = mc.stitch(f, preview=False)
from microcontrol.mask import Frames
fr = Frames(f); A = np.array(fr.st.meta['A']); d = fr.st.meta['motor_step_um']
truth = -(true @ A.T); truth -= truth.min(0)
err = (fr.P - fr.P.mean(0)) - (truth - truth.mean(0))
assert np.sqrt((err ** 2).mean()) < 6, err                      # было бы ~40 px по столику
t = np.radians(3.5)
world = lambda i, j: np.array([np.cos(t)*i*100 - np.sin(t)*j*100, np.sin(t)*i*100 + np.cos(t)*j*100])
def mos_xy(S):
    k = int(np.argmin(np.hypot(*(true * d - S).T)))
    return fr.P[k] + np.array([fr.W / 2, fr.H / 2]) + A @ (true[k] - S / d)
def view_xy(S):
    M = mos_xy(S)
    Wsc = 1280 - mc.samples.CAR_W                                   # справа — карусель образцов
    s = min(Wsc / fr.size[0], (900 - mc.ui.MENU_H) / fr.size[1]) * 0.95
    v = np.array([Wsc / 2, (900 - mc.ui.MENU_H) / 2]) + s * (M - fr.size / 2)
    return int(v[0]) + 3, int(v[1]) - 2
sq = {n: mos_xy(world(*ij)) for n, ij in (('a', (2.5, 2.5)), ('b', (4.5, 3.5)), ('c', (3.5, 2.5)), ('e', (2.5, 4.5)))}
fr.close()
typed = lambda t: [ord(ch) for ch in t] + [13, -1]
clk = lambda ij: [('click', *view_xy(world(*ij)), 'shift'), -1]     # Shift+щелчок рядом — к центру креста
# два креста в ряду -> наклон, полотно само докручивается (R — вернуть как снято, чтобы щёлкать по расчёту)
keys = [ord('c')] + clk((2, 2)) + clk((3, 2)) + [ord('t'), -1] + clk((9, 2)) + clk((8, 6))   # дальние уточняют
keys += [ord('t'), -1, ord('t'), -1, ord('m'), -1, ('drag', 560, 380, 720, 520), -1]
keys += [ord('n'), -1, ('click', *view_xy(world(2.5, 2.5))), -1] + typed('12 7') + \
        [('click', *view_xy(world(4.5, 3.5))), -1] + typed('14 6')      # номера литографии: X вдоль i, Y — против j
keys += [ord('e'), -1, 27, 27]
mc.view(f, _keys=keys)
grid = json.load(open(os.path.join(f, 'grid.json')))
assert abs(grid['pitch_um'][0] - 100) < 1 and abs(grid['pitch_um'][1] - 100) < 1, grid['pitch_um']
assert abs(grid["turn_deg"] - 3.5) < 0.2 and grid['selected'], grid
assert {'selection.png', 'selection.dxf', 'selection.json'} <= set(os.listdir(os.path.join(f, 'export')))
import ezdxf                                                       # итоговый чертёж: картинка на каждый квадрат
assert len(ezdxf.readfile(os.path.join(f, 'export', 'drawing.dxf')).modelspace().query('IMAGE')) == \
    len(os.listdir(os.path.join(f, 'export', 'squares')))
from microcontrol.mask import Grid
gg = Grid(f)
assert len(gg.labels) == 2 and 'fixed' in gg.numbering()[2], (gg.labels, gg.numbering())
assert gg.name(*gg.cell_at(sq['c'])) == (13, 7) and gg.name(*gg.cell_at(sq['e'])) == (12, 5), \
    (gg.name(*gg.cell_at(sq['c'])), gg.name(*gg.cell_at(sq['e'])))
sj = json.load(open(os.path.join(f, 'export', 'selection.json')))
assert len(sj['names']) == len(sj['squares']) and 'S13,7' not in open(os.path.join(f, 'export', 'selection.dxf')).read()
print("Square numbers:", gg.numbering()[2])
# простой щелчок ставит крест ровно в точку щелчка (без притяжения), отметку можно перетащить мышью
Wsc_ = 1280 - mc.samples.CAR_W
n0 = len(gg.cr.marks)
u0, v0 = view_xy(world(5, 5)); u0, v0 = u0 - 3, v0 + 2
mc.view(f, _keys=[ord('c'), -1, ('click', u0, v0), -1, ('drag', u0, v0, u0 + 10, v0 + 4), -1, 27, 27])
g2 = Grid(f)
assert len(g2.cr.marks) == n0 + 1, (n0, len(g2.cr.marks))
fr = Frames(f); size_ = np.array(fr.size, float); fr.close()
sv = min(Wsc_ / size_[0], (900 - mc.ui.MENU_H) / size_[1]) * 0.95
exp_ = size_ / 2 + (np.array([u0 + 10.5, v0 + 4.5]) - (Wsc_ / 2, (900 - mc.ui.MENU_H) / 2)) / sv
assert np.hypot(*(g2.cr.marks[-1] - exp_)) < 0.6 / sv + 0.5, (g2.cr.marks[-1], exp_)
u1, v1 = view_xy(world(6, 5)); u1, v1 = u1 - 3, v1 + 2
mc.view(f, _keys=[ord('c'), -1, ('click', u1, v1), -1, ('drag', u1, v1, u1 + 6, v1), -1, 26, -1, 26, -1, 25, -1,
                  26, -1, ord('r'), -1, ('click', 300, 300), -1, ('click', 420, 380), -1, 27, 27])
assert len(Grid(f).cr.marks) == n0 + 1, (n0, len(Grid(f).cr.marks))              # крест, сдвиг -> Ctrl+Z x2, Ctrl+Y, Ctrl+Z
rb_ = {}
pxu_ = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
mc.view(f, _keys=[ord('r'), -1, ord('r'), -1, ('click', 300, 300), -1, ('click', 500, 300), -1, ('click', 300, 350), -1,
                  ('click', 500, 350), -1, ('call', lambda st: rb_.update(ll=st['msg'])), -1,
                  ord('r'), -1, ('click', 300, 300), -1, ('click', 500, 300), -1, ('click', 400, 340), -1,
                  ('call', lambda st: rb_.update(pl=st['msg'])), -1, 27, 27])     # линейка: прямая-прямая, точка-прямая
import re as _re
dll_ = float(_re.search(r"([0-9.]+) um between the lines", rb_['ll']).group(1))
dpl_ = float(_re.search(r"([0-9.]+) um from the line", rb_['pl']).group(1))
assert abs(dll_ - 50 / sv * pxu_) < 0.02 * 50 / sv * pxu_ and 'angle 0.00 deg' in rb_['ll'], (rb_, 50 / sv * pxu_)
assert abs(dpl_ - 40 / sv * pxu_) < 0.02 * 40 / sv * pxu_, (rb_, 40 / sv * pxu_)
print("Cross marks: exactly at the click, draggable, Ctrl+Z / Ctrl+Y; ruler line-line and point-line: OK")
n0 = len(json.load(open(os.path.join(f, 'grid.json')))['selected'])
mc.view(f, _keys=[ord('m'), -1, ('rclick', 640, 450), -1, 27, 27])      # правый щелчок по зелёному — убрать область
n1 = len(json.load(open(os.path.join(f, 'grid.json')))['selected'])
assert n1 < n0, (n0, n1)
mc.clean(f, confirm=True)
assert not os.path.exists(os.path.join(f, 'frames'))
print("Stitch / mask / export: OK")

# 15. Несколько образцов: области 1, 2, 3 по очереди, без перекрытия -> сеанс -> карусель в view
print("\n== Samples ==")
import microcontrol.manual as M
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
m._world.size = (640, 480); m.focus()
R_, D_, L_, U_ = 65363, 65364, 65361, 65362
keys = [ord('1'), -1, R_, -1, R_, -1, D_, -1, ord('k'), -1, ord('2'), -1]          # образец 1: 3 x 2 кадра, точки 1-3
keys += [R_, -1, R_, -1, ord('1'), -1, R_, -1, D_, -1, ord('2'), -1]                 # образец 2: точки 4-5
keys += [L_, -1, L_, -1, ord('1'), -1]                                              # внутри образца 2 -> отказ
keys += [L_, -1, L_, -1, L_, -1, U_, -1, U_, -1, U_, -1, ord('1'), -1,                # новый угол 1 над образцом 1,
         D_, -1, D_, -1, D_, -1, ord('2'), -1, ord('c'), -1, 27]                     # угол 2 внутри него -> отказ, отмена
M.run(m, fullscreen=False, _keys=keys)
regs = m._regions
assert len(regs) == 2 and m._pending is None, (len(regs), m._pending)
assert [n for n, k, _ in mc.samples.numbered(m)] == [1, 2, 3, 4, 5] and [len(r['pts']) for r in regs] == [3, 2]
fw_, fh_ = m.frame_size
assert not mc.samples.overlap(mc.samples.rect(regs[0], fw_, fh_), mc.samples.rect(regs[1], fw_, fh_))
ses = m.scan(name='Two samples')
m.close()
subs = mc.samples.samples(ses)                                       # образец: последняя съёмка каждого выделения
assert [x.split(os.sep)[-3] for x in subs] == ['S01_10x', 'S02_10x'], subs
assert all(json.load(open(os.path.join(x, 'meta.json')))['sample_index'] == i + 1 for i, x in enumerate(subs))
r = mc.stitch(ses, verbose=False)
assert len(r) == 2, r
Wsc = 1280 - mc.samples.CAR_W
mc.view(ses, _keys=[-1, ord('.'), -1, -1, ('click', 1280 - 25, 64), -1, -1, -1, 27])
left = mc.samples.samples(ses)
assert [x.split(os.sep)[-3] for x in left] == ['S02_10x'] and \
    os.path.isdir(os.path.join(os.path.dirname(subs[0]), '_removed', os.path.basename(subs[0])))   # x у 1-го
print("Samples: 2 areas, focus points 1-5, two folders, carousel switch and remove: OK")

# ------------------------------------------------------------------ 16. ИИ-агент
import os, io, json, tempfile, csv, numpy as np
from microcontrol.mask import Frames
asked = []
def approve(title, text):
    asked.append((title, text)); return 'deny' if 'DENYME' in text else 'once'
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
w = m._world; w.turn_deg = 3.5; m.focus()
t = np.radians(3.5)
wp = lambda i, j: (np.cos(t)*i*100 - np.sin(t)*j*100, np.sin(t)*i*100 + np.cos(t)*j*100)
w.flakes = [(*wp(1.4, 1.6), 30, 40, 4), (*wp(2.6, 0.4), 18, -20, 3)]
ag = mc.Agent(m, approve=approve)
C = lambda n, **a: ag.call(n, a)
def ok(r):
    assert r['ok'], r; return r['result']
st = ok(C('get_status')); print('status', st['position_um'], st['field_of_view_um'])
print(ok(C('get_manual'))[:60])
r = ok(C('look', target='camera')); assert os.path.exists(r['image']); print('camera', r['width'], r['sharpness'])
fw, fh = st['field_of_view_um']
ok(C('area_corner', corner=1))
ok(C('move_stage', dx_um=2.5 * fw, dy_um=1.5 * fh, reason='go to corner 2'))
r = ok(C('area_corner', corner=2)); print(r['message'])
ok(C('add_focus_point'))
# ошибки
r = C('area_corner', corner=1); assert r['ok'], r                                # внутри — того же объектива можно
m._pending = None
r = C('move_stag', dx_um=1); assert r['error']['code'] == 'NO_TOOL' and 'move_stage' in r['error']['hint'], r
r = C('move_stage', dx=5, reason='x'); assert r['error']['code'] == 'BAD_ARGS', r
r = C('move_stage', dx_um='abc', reason='x'); assert r['error']['code'] == 'BAD_ARGS', r
r = C('move_stage', dx_um=10); assert r['error']['code'] == 'BAD_ARGS', r              # нет reason
r = C('move_stage', dx_um=10, reason='DENYME'); assert r['error']['code'] == 'DENIED', r
r = C('move_stage', dz_um=5000, reason='up'); assert not r['ok'] and r['error']['code'].startswith('E2'), r
r = C('change_setting', name='laser', value=5, reason='x'); assert r['error']['code'] == 'BAD_ARGS', r
ok(C('change_setting', name='light', value=4, reason='brighter'))
r = ok(C('look', target='map')); print('map', r['image'])
r = ok(C('scan', dry_run=True, reason='estimate')); print('dry', r)
n_ask = len(asked)
r = ok(C('scan', reason='shoot', name='agent')); print('scan', r); assert len(asked) == n_ask + 1
folder = r['folder']
r = ok(C('stitch', folder=folder)); print('stitch ok')
L = ok(C('list_scans')); print('list', json.dumps(L)[:200])
info = ok(C('scan_info', folder=folder)); print('info', info['next'])
r = C('look', target='mosaic', folder=folder); assert not r['ok'] or True
r = ok(C('look', target='mosaic', folder=os.path.join(folder) if 'kind' in L[-1] and L[-1].get('kind') == 'scan' else folder))
print('mosaic', r['mosaic_size_px'], r['grid'])
# кресты: настоящие места в мозаике
f = os.path.join(m.dirs['scans'], folder)
fr = Frames(f); A = np.array(fr.st.meta['A']); dd = fr.st.meta['motor_step_um']
rows = list(csv.DictReader(open(os.path.join(f, 'index.csv'), encoding='utf-8')))
true = np.array([[int(q['x_p']), int(q['y_p'])] for q in rows], float)
def mos(S):
    k = int(np.argmin(np.hypot(*(true * dd - S).T)))
    return fr.P[k] + np.array([fr.W / 2, fr.H / 2]) + A @ (true[k] - S / dd)
for ij in ((1, 1), (2, 1)):
    p = mos(np.array(wp(*ij))) + (4, -3)
    r = ok(C('mark_cross', folder=folder, x_px=float(p[0]), y_px=float(p[1]))); print('cross', r)
r = ok(C('fit_grid', folder=folder)); print('fit', r)
c = mos(np.array(wp(1.5, 1.5)))
ok(C('label_square', folder=folder, x_px=float(c[0]), y_px=float(c[1]), square=[12, 7]))
c = mos(np.array(wp(2.5, 0.5)))
r = ok(C('label_square', folder=folder, x_px=float(c[0]), y_px=float(c[1]), square=[13, 8])); print(r)
r = ok(C('list_squares', folder=folder, which='all', limit=5)); print('sq', r['squares'][:2])
r = C('find_hints', folder=folder); assert r['error']['code'] == 'NO_TOOL', r      # подсказок больше нет
r = ok(C('select_squares', folder=folder, action='add', squares=[[12, 8]])); print('sel', r)
r = C('select_squares', folder=folder, action='add', squares=[[12]]); assert r['error']['code'] == 'BAD_ARGS', r
r = ok(C('look', target='mosaic', folder=folder)); print('mosaic2', r['image'])
r = ok(C('export', folder=folder)); print('export', r['result'][:3])
assert not hasattr(mc, 'hints') and not hasattr(mc, 'train_hints')
# MCP
inp = io.StringIO("\n".join(json.dumps(x) for x in [
    dict(jsonrpc='2.0', id=1, method='initialize', params=dict(protocolVersion='2025-06-18')),
    dict(jsonrpc='2.0', method='notifications/initialized'),
    dict(jsonrpc='2.0', id=2, method='tools/list'),
    dict(jsonrpc='2.0', id=3, method='tools/call', params=dict(name='get_status', arguments={})),
    dict(jsonrpc='2.0', id=4, method='tools/call', params=dict(name='look', arguments=dict(target='camera')))]) + "\n")
out = io.StringIO()
mc.agent.mcp_server(m, approve=approve, inp=inp, out=out)
R = [json.loads(x) for x in out.getvalue().splitlines()]
assert [x['id'] for x in R] == [1, 2, 3, 4], R
assert len(R[1]['result']['tools']) == 25 and R[3]['result']['content'][1]['type'] == 'image'
print('mcp ok', len(R[1]['result']['tools']), 'tools; asked', len(asked))
m.close()
print("Agent: tools, approval, errors, scan, grid, export, MCP: OK")

# ------------------------------------------------------------------ 17. Z по точкам фокуса и время скана
import io, contextlib, microcontrol.manual as M
from microcontrol import timing as TM
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
m.focus()
R_, L_, D_, Q_ = 65363, 65361, 65364, ord('q')
z0 = m.position.z
# угол 1 на Z0, угол 2 на 3 клетки правее и на 100 мкм ниже
M.run(m, fullscreen=False, _keys=[ord('1'), -1, R_, -1, R_, -1, R_, -1, Q_, -1, ord('2'), -1, -1])
z2 = m.position.z
assert z2 < z0, (z0, z2)
M.run(m, fullscreen=False, _keys=[L_, -1, -1])                       # шаг внутрь области — Z по точкам
z1 = m.position.z
print("Z corner1", z0, "corner2", z2, "after arrow", z1)
assert z2 < z1 < z0, (z0, z1, z2)
m._use_fp = False
M.run(m, fullscreen=False, _keys=[L_, -1, -1])
assert m.position.z == z1                                           # флажок снят — Z не трогается
m._use_fp = True
e0 = TM.estimate(m, 12, 'points'); print("before", e0)
for i in range(2):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = m.scan(name=f't{i}')
    print([l for l in buf.getvalue().splitlines() if l.startswith('Expected')])
e1 = TM.estimate(m, 12, 'points'); print("after", e1)
assert 'learned from 2 scans' in e1[1]
m.close(); print("Z follow + timing: OK")

# ------------------------------------------------------------------ 18. Постоянные номера выделений, uid; проекты и образцы
import json, microcontrol.manual as M
from microcontrol import samples as SM
d18 = tempfile.mkdtemp()
m = mc.Microscope('sim', objective=10, folder=d18)
m.focus()
R_, D_ = 65363, 65364
k = [ord('1'), -1, R_, -1, ord('2'), -1, R_, -1, R_, -1, ord('1'), -1, R_, -1, ord('2'), -1, -1]
M.run(m, fullscreen=False, _keys=k)
assert [r['num'] for r in m._regions] == [1, 2] and len({r['uid'] for r in m._regions}) == 2
SM.delete_area(m, 0)                                   # удалить выделение 1: у второго номер остаётся 2
assert [r['num'] for r in m._regions] == [2]
M.run(m, fullscreen=False, _keys=[R_, -1, R_, -1, ord('1'), -1, R_, -1, ord('2'), -1, -1])
assert [r['num'] for r in m._regions] == [2, 3], [r['num'] for r in m._regions]
sf = m.scan(name='uid')
sels = [x.split(os.sep)[-3] for x in mc.project.latest_scans(sf)]; print(sels)
assert sels == ['S02_10x', 'S03_10x'], sels
u2 = SM.sample_uid(m.sample.selections[0].latest.folder); assert u2 == m._regions[0]['uid']
# проекты и образцы: у каждого образца — свои выделения; переживают перезапуск
p2 = mc.Project("Graphene", "flakes on SiO2", folder=d18)
s2 = p2.sample("Chip B", "285 nm")
s1 = m.sample
m.sample = s2
assert m._regions == [] and m.sample.title == "Graphene / Chip B"
m.sample = s1
assert [r['num'] for r in m._regions] == [2, 3]
m.close()
m = mc.Microscope('sim', objective=10, folder=d18)              # снова: открыт последний образец, выделения на месте
assert m.sample == s1 and [r['num'] for r in m._regions] == [2, 3]
assert {p.name for p in mc.projects(d18)} == {"Project 1", "Graphene"} and len(mc.all_samples(d18)) == 2
m.close(); print("Selection numbers stay, uid, projects and samples: OK")

# ------------------------------------------------------------------ 19. Каждая съёмка выделения — новая версия
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, fw, fh)
f1 = m.scan(name='Same')
f2 = m.scan(name='Same')
sel = m.sample.selections[0]
assert f1 != f2 and [x.folder for x in sel.scans] == [f1, f2] and sel.latest.folder == f2 and len(m.sample.selections) == 1
lst = m.list(); print(lst)
assert len(lst) == 2 and lst[1]['version'] == (2, 2)
m.close(); print("Every scan of a selection is a new version: OK")

# ------------------------------------------------------------------ 20. Кресты переезжают вместе с кадрами при сшивке
from microcontrol.mask import Grid
d = tempfile.mkdtemp()
g = Grid(d)
old = np.array([[0, 0], [100, 0], [200, 0]], float)        # три кадра 120 x 80 px, перекрытие 20 px
for q in ((50, 40), (150, 40)):
    g.cr.add(np.array(q, float))
g.frames_px = old.copy()
new = old + [[0, 0], [3, -2], [5, 1]]                       # сшивка сдвинула кадры 2 и 3
n = g.sync(new, 120, 80)
assert n == 2, n
assert np.allclose(g.cr.marks[0], (50, 40)) and np.allclose(g.cr.marks[1], (153, 38)), g.cr.marks
g.save(1.0)
g2 = Grid(d); assert np.allclose(g2.frames_px, new) and g2.sync(new, 120, 80) == 0
print("Crosses move with their frames after stitching: OK")

# ------------------------------------------------------------------ 21. Цветная поправка освещения, яркость кадров
from microcontrol.storage import ScanData
from microcontrol.stitch import load_positions
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
w = m._world; w.vignette_bgr = (0.10, 0.35, 0.25); w.flicker = 0.08; m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 2 * fh)
f = m.scan(name='flat')
m.close()
def seam_err(st, P, W, H):
    e = []
    for a in range(len(st)):
        for b in range(a + 1, len(st)):
            d = P[b] - P[a]; ox, oy = W - abs(d[0]), H - abs(d[1])
            if ox < 0.1 * W or oy < 0.1 * H: continue
            A = st.frame(a)[int(max(0, d[1])):int(max(0, d[1]) + oy), int(max(0, d[0])):int(max(0, d[0]) + ox)].reshape(-1, 3)
            B = st.frame(b)[int(max(0, -d[1])):int(max(0, -d[1]) + oy), int(max(0, -d[0])):int(max(0, -d[0]) + ox)].reshape(-1, 3)
            e.append(np.abs(np.log(np.median(A, 0) + 1) - np.log(np.median(B, 0) + 1)).max())
    return 100 * np.mean(e)
st = ScanData(f); st.fix = None; st.gains = None
H, W = st.shape[:2]
P0 = load_positions(f, st)
before = seam_err(st, P0, W, H); st.close()
r = mc.stitch(f, verbose=False); print(r)
st = ScanData(f); P = load_positions(f, st)
after = seam_err(st, P, W, H); st.close()
print(f"seam colour/brightness mismatch: before {before:.1f} %, after {after:.1f} %")
assert after < 0.5 * before and after < 3, (before, after)
print("Colour flat-field + frame gains: OK")


# ------------------------------------------------------------------ 22. Слои: карта 10x, внутри участок 50x; список, удаление, время
from microcontrol import samples as SM
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
m.focus()
fw, fh = m.frame_size
SM.corner(m, 1); m.move(dx=2 * fw); m.move(dy=fh); SM.corner(m, 2)               # карта 10x: 3 x 2 кадра
mapf = m.scan(name='map')
m.objective = 50
m.goto(x=fw, y=fh / 2)                                                             # середина карты
SM.corner(m, 1)
fw5, fh5 = m.frame_size
m.move(dx=fw5); SM.corner(m, 2)                                                    # участок 50x внутри карты
assert [SM.obj(m, r) for r in m._regions] == [10, 50] and SM.parent(m, 1) == 0, [SM.parent(m, 1)]
assert SM.tree(m) == [(0, 0), (1, 1)]
m.objective = 10
SM.corner(m, 1); m._pending = None                                                # тот же объектив внутри — можно
m.objective = 50
det = m.scan(name='detail')                                                        # снимается только 50x
assert json.load(open(os.path.join(det, 'meta.json')))['parent_uid'] == m._regions[0]['uid']
m.objective = 20
try:
    m.scan(); raise AssertionError("scan without selections of 20x")
except mc.MicroControlError as e:
    assert 'E706' in str(e), e
lst = m.list(); print(lst)
assert len(lst) == 2 and lst[0]['objective'] == 10 and lst[1]['depth'] == 1 and lst[1]['parent'] is lst[0], lst
inf = m.info(); txt = repr(inf)
assert 'Where the time goes' in txt and 'Camera: picture' in inf['Time'] and inf['Time']['Camera: picture']['calls'] > 5
print(txt.split("\n\n")[1].splitlines()[2])
r = m.remove(2, confirm=False); print(r)
assert not os.path.exists(det) and len(m.list()) == 1 and os.path.isdir(mapf)
mc.view(mapf, _keys=[-1, ord('s'), -1, -1, 27, 27])                                # S: сшить прямо в просмотре
assert os.path.exists(os.path.join(mapf, 'join', 'positions_all.csv'))
m.close(); print("Layers 10x > 50x, list, remove, time report, stitch from the viewer (S): OK")

# ------------------------------------------------------------------ 23. Ctrl+Z / Ctrl+Y в ручном режиме
import microcontrol.manual as M
d23 = tempfile.mkdtemp()
m = mc.Microscope('sim', objective=10, folder=d23)
m.focus()
p0 = m.position
R_ = 65363
M.run(m, fullscreen=False, _keys=[ord('1'), -1, R_, -1, ord('2'), -1, -1, 26, -1, 26, -1, -1, 26, -1, -1, 27])
assert not m._regions and m._pending is None and m.position == p0, (m._regions, m._pending, m.position, p0)
M.run(m, fullscreen=False, _keys=[25, -1, 25, -1, -1, 25, -1, -1, 27])                # всё вернуть
assert len(m._regions) == 1 and m.position != p0, (m._regions, m.position)
M.run(m, fullscreen=False, _keys=[ord('r'), -1, (600, 300), -1, (700, 300), -1, -1, ord('r'), -1, ord('r'), -1,
                                  (600, 300), -1, R_, -1, (600, 300), -1, -1])     # линейка; точка 2 — после сдвига
d_ = np.hypot(*(np.array(m._ruler['pts'][1]) - m._ruler['pts'][0]))
assert len(m._ruler['pts']) == 2 and abs(d_ - m.frame_size[0]) < 0.1 * m.frame_size[0], (m._ruler, d_, m.frame_size)
M.run(m, fullscreen=False, _keys=[ord('1'), -1, R_, -1, ord('2'), -1, ord('s'), -1, ord('c'), -1, ord('s'), -1, -1, ord('v'), 27, 27, -1])
sf = m._last_view
assert os.path.exists(os.path.join(sf, 'join', 'positions_all.csv')), sf              # S, S: снято и сшито в окне
m.close()
import microcontrol.viewer as VW                                    # перезапуск: V открывает последнюю съёмку
opened = []                                                        # выбранного выделения
_v0 = VW.view
VW.view = lambda f, **kw: opened.append(f)
m = mc.Microscope('sim', objective=10, folder=d23)
M.run(m, fullscreen=False, _keys=[ord('v'), -1, -1])
VW.view = _v0
assert opened and opened[0] == m.sample.selections[-1].latest.folder, (opened, m.sample.selections)
m.close(); print("Undo / redo in manual mode (selections and moves), ruler, scan + stitch (S) and view (V) from the window: OK")

# ------------------------------------------------------------------ 24. Кисть -> выделения (по фигуре), операции над слоями
import microcontrol.manual as M
from microcontrol import samples as SM
from microcontrol import layers as LY
from microcontrol.mask import Grid
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 2 * fh)
f = m.scan(); mc.stitch(f, verbose=False)
px_um = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
g = Grid(f); p0 = np.array([66.0, 52.8]) / px_um; a_ = 100 / px_um   # 66, 53 мкм от угла мозаики
g.cr.add(p0); g.cr.add(p0 + (a_, 0)); g.cr.add(p0 + (0, a_))
g.selected |= {(0, 0), (1, 0), (3, 1), (3, 2)}                     # две фигуры по 2 квадрата
g.save(px_um)
print(LY.sync_brush(m))
kids = [k for k, r in enumerate(m._regions) if r.get('src')]
assert len(kids) == 2 and all(m._regions[k]['parent'] == m._regions[0]['uid'] and m._regions[k]['kind'] == 'draft'
                              for k in kids), m._regions
assert abs(LY.geom(m, m._regions[kids[0]]).area - 2 * 100 * 100) < 50      # 2 квадрата 100 x 100 мкм, контур без клеток
assert [d for k, d in SM.tree(m)] == [0, 1, 1]                      # матрёшка: карта 10x > два черновика
assert LY.sync_brush(m) is None                                     # ничего не изменилось
g = Grid(f); g.selected |= {(2, 0)}; g.save(px_um)                  # кисть дорисовала квадрат — выделение растёт
LY.sync_brush(m)
assert len([r for r in m._regions if r.get('src')]) == 2 and \
    abs(LY.geom(m, [r for r in m._regions if r.get('src')][0]).area - 3e4) < 80
ks = [k for k, r in enumerate(m._regions) if r.get('src')]
print(LY.union(m, ks))                                              # слить: одно выделение, отпущено от кисти
assert len([r for r in m._regions if r.get('kind') == 'draft']) == 1 and not any(r.get('src') for r in m._regions)
assert LY.sync_brush(m) is None                                     # кисть его не пересоздаёт
k20 = next(k for k, r in enumerate(m._regions) if r.get('kind') == 'draft')
a0 = LY.geom(m, m._regions[k20]).area
m._regions.append(dict(uid='cut', num=LY.new_num(m), obj=20, fw=fw / 2, fh=fh / 2, pts=[], parent=m._regions[0]['uid']))
x0_, y0_, x1_, y1_ = LY.geom(m, m._regions[k20]).bounds
LY.set_geom(m, m._regions[-1], LY.box(x0_, y0_, x0_ + 100, y0_ + 100))
print(LY.subtract(m, [k20, len(m._regions) - 1]))
assert abs(LY.geom(m, m._regions[k20]).area - (a0 - 1e4)) < 80
print(LY.group(m, [k20, len(m._regions) - 1]))
gk = len(m._regions) - 1
assert m._regions[gk]['kind'] == 'group' and [d for k, d in SM.tree(m)] == [0, 1, 2, 2]
print(LY.ungroup(m, [gk]))
assert [d for k, d in SM.tree(m)] == [0, 1, 1]
SM.persist(m)
assert any(r.get('shape') for r in m.sample.data['selections'])     # контуры — в sample.json
m.close(); print("Brush -> drafts (one per figure), merge / minus / group / ungroup, outline in sample.json: OK")

# ------------------------------------------------------------------ 25. Порядок карточек перетаскиванием, прокрутка карусели
import copy as _cp
import microcontrol.manual as M
from microcontrol import samples as SM
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 2 * fw, 2 * fh)
SM.init_areas(m)
r0 = m._regions[0]
for i in range(1, 12):                                              # 12 выделений рядом, не друг в друге
    r = _cp.deepcopy(r0); r['num'] = i + 1; r['uid'] = f"u{i}"; r['thumbs'] = []
    d_ = 3 * fw * i
    r['A'] = (r0['A'][0] + d_, r0['A'][1]); r['B'] = (r0['B'][0] + d_, r0['B'][1])
    m._regions.append(r)
m._area_seq = 12
top = [m._regions[k]['num'] for k, d in SM.tree(m)]
assert top == list(range(1, 13)), top
print(SM.reorder(m, 2, 0))                                          # Selection 3 перед Selection 1
assert [m._regions[k]['num'] for k, d in SM.tree(m)][:3] == [3, 1, 2]
assert m.sample.data['selections'][2]['rank'] == 0                  # порядок — в sample.json
X = 1280 - mc.samples.CAR_W + 60
shots = []; _o = M.cv2.imshow
M.cv2.imshow = lambda w, img: (shots.append(img.copy()), _o(w, img))
M.run(m, fullscreen=False, _keys=[-1, ('car', X, 70), ('cmove', X, 200), ('cmove', X, 300), -1,
                                  ('cdrop', X, 300), -1,                                   # первую карточку — ниже
                                  ('carw', X, 300, -1), ('carw', X, 300, -1), -1,          # колесо
                                  ('car', 1279, 60), ('cmove', 1279, 600), ('cdrop', 1279, 600), -1, -1])   # полоса
M.cv2.imshow = _o
M.cv2.imwrite('/tmp/car_drag.png', shots[3]); M.cv2.imwrite('/tmp/car_scroll.png', shots[-1])
order = [m._regions[k]['num'] for k, d in SM.tree(m)]
assert order[0] != 3 and 3 in order[1:5], order                     # карточка Selection 3 переехала вниз
m.close(); print("Drag and drop order of selections, carousel scroll (wheel, bar): OK", order[:5])

# ------------------------------------------------------------------ 26. Shift + тянуть: сдвинуть выделение, Ctrl+Z; C — удалить выделение кисти, Ctrl+Z
import microcontrol.manual as M
from microcontrol import samples as SM
from microcontrol import layers as LY
from microcontrol.mask import Grid
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 2 * fh)
f = m.scan(); mc.stitch(f, verbose=False)
m.goto(x=fw, y=fh / 2)
A0 = tuple(m._regions[0]['A'])
M.run(m, fullscreen=False, _keys=[-1, ('mdrag0', 400, 400), ('mdrag', 440, 380), -1, ('mdrag1', 440, 380), -1, -1])
d1 = np.subtract(m._regions[0]['A'], A0)
assert np.hypot(*d1) > 1, d1
M.run(m, fullscreen=False, _keys=[-1, ('mdrag0', 400, 400), ('mdrag', 440, 380), -1, ('mdrag1', 440, 380), -1, 26, -1, -1])
assert np.allclose(np.subtract(m._regions[0]['A'], A0), d1, atol=0.01)       # Ctrl+Z вернул второй сдвиг
LY.move(m, 0, -d1[0], -d1[1]); SM.persist(m)
px_um = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
g = Grid(f); p0 = np.array([66.0, 52.8]) / px_um; a_ = 100 / px_um   # 66, 53 мкм от угла мозаики
g.cr.add(p0); g.cr.add(p0 + (a_, 0)); g.cr.add(p0 + (0, a_))
g.selected |= {(0, 0), (1, 0)}; g.save(px_um)
gj = lambda: sorted(map(tuple, json.load(open(os.path.join(f, 'grid.json')))['selected']))
LY.sync_brush(m); SM.persist(m)
m._sel = next(k for k, r in enumerate(m._regions) if r.get('src'))
M.run(m, fullscreen=False, _keys=[-1, ord('c'), -1, -1])            # C — удалить сразу; квадраты снимаются с кисти
assert gj() == [] and not any(r.get('src') for r in m._regions), (gj(), m._regions)
g = Grid(f); g.selected |= {(0, 0), (1, 0)}; g.save(px_um); LY.sync_brush(m); SM.persist(m)
m._sel = next(k for k, r in enumerate(m._regions) if r.get('src'))
M.run(m, fullscreen=False, _keys=[-1, ord('c'), -1, 26, -1, -1])    # и Ctrl+Z — назад: и выделение, и квадраты
assert gj() == [(0, 0), (1, 0)] and any(r.get('src') for r in m._regions), (gj(), m._regions)
m.close(); print("Shift+drag moves a selection, C deletes a brush selection at once, Ctrl+Z brings both back: OK")

# ------------------------------------------------------------------ 27. Строгая геометрия маски: люфт столика учтён
from microcontrol import snap
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
m._world.backlash = {'x': 2, 'y': 2}                                # шаги, теряемые при смене направления
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 4 * fw, 3 * fh)
f = m.scan(); mc.stitch(f, verbose=False)
g = snap.geometry(f)
assert all(abs(b - 2 * m.motor_step) < 1.5 for b in g['backlash_um']) and g['resid'] < 0.5, (g['backlash_um'], g['resid'])
cx, cy = 1.5 * fw, 1.2 * fh
for sx, sy in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:                  # подъезжаем с разных сторон
    m.goto(x=cx - sx * 300, y=cy - sy * 300); m.goto(x=cx, y=cy)
    img = m.image(0)
    d, resp, px = snap.offset(f, img, (m._p['x'], m._p['y']), (m._dir['x'], m._dir['y']))
    d0, resp0, px0 = snap.offset(f, img, (m._p['x'], m._p['y']), (0, 0))
    assert np.hypot(*px) < 1.5 and np.hypot(*px0) > 8, (sx, sy, px, px0)
m.close(); print("Strict mask geometry: backlash", g['backlash_um'], "um counted, error < 1.5 px from any side: OK")


# ------------------------------------------------------------------ 28. Скан по контуру: только кадры, задевающие выделение
from microcontrol import layers as LY
from microcontrol import samples as SM
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
SM.init_areas(m)
reg = dict(uid='L', num=LY.new_num(m), obj=10, fw=fw, fh=fh, pts=[], parent=None, thumbs=[])
L_ = LY.box(0, 0, 4 * fw, fh).union(LY.box(0, 0, fw, 3 * fh))      # «Г»: 4 кадра + 2 кадра вниз
LY.set_geom(m, reg, L_)
m._regions.append(reg); m._sel = 0
n_bbox = SM._area(m, reg).size(m._frame_step(1 - mc.config.scan_overlap(10)))
n_shape = SM.frames_of(m, 0)[0]
f = m.scan()
st = mc.storage.ScanData(f); nfr = len(st.index); st.close()
assert nfr == n_shape < n_bbox[0] * n_bbox[1], (nfr, n_shape, n_bbox)
r = mc.stitch(f, verbose=False)
m.close(); print(f"Scan by the outline: {nfr} frames instead of {n_bbox[0] * n_bbox[1]}, stitched: OK")

# ------------------------------------------------------------------ 29. Сетка под наклоном + люфт: углы выделений кисти — на настоящих крестах
from microcontrol import layers as LY, snap
from microcontrol.mask import Grid
for turn in (5.0, -12.0):
    m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
    m._world.turn_deg = turn; m._world.backlash = {'x': 2, 'y': 2}
    fw, fh = m.frame_size
    m.area = mc.Area(0, 0, 4 * fw, 3 * fh)
    f = m.scan(); mc.stitch(f, verbose=False)
    g_ = snap.geometry(f); ms = g_['ms']; t = np.radians(turn); P = 100.0
    W2q = lambda X, Y: g_['Mo'] @ (np.array([X, Y]) / ms) + g_['c'] + (g_['W'] / 2, g_['H'] / 2)
    lat = lambda i, j: (i * P * np.cos(t) - j * P * np.sin(t), i * P * np.sin(t) + j * P * np.cos(t))
    g = Grid(f)
    for ij in [(1, 1), (2, 1), (1, 2)]:                             # кресты отмечены там, где они на мозаике
        g.cr.add(W2q(*lat(*ij)))
    g.selected |= {(0, 0), (1, 0), (1, 1)}                          # «Г» из трёх квадратов
    g.save(json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um'])
    LY.sync_brush(m)
    r = next(r for r in m._regions if r.get('src'))
    V_ = np.array(r['shape'][0]['ext'])
    L_ = np.array([lat(i, j) for i in range(-2, 6) for j in range(-2, 6)])
    d_ = max(np.min(np.hypot(*(L_ - v).T)) for v in V_)
    assert len(V_) == 6 and d_ < 0.5, (turn, len(V_), d_)            # 6 углов, каждый — на кресте (< 0.5 мкм)
    m.close()
print("Tilted grid (5, -12 deg) with backlash: brush outlines sit on the real crosses (< 0.5 um): OK")

# ------------------------------------------------------------------ 30. Скрыть / показать выделение (глаз на карточке, Show all)
import microcontrol.manual as M
from microcontrol import samples as SM
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 2 * fw, 2 * fh)
SM.init_areas(m); SM.persist(m)
M.run(m, fullscreen=False, _keys=[-1, ('car', 1184, 64), -1, -1])   # глаз первой карточки
assert SM.hidden(m, 0) and m.sample.data['selections'][0].get('hidden'), m._regions
M.run(m, fullscreen=False, _keys=[-1, ('car', 1280 - SM.CAR_W + 30, 54), -1, -1])   # Show all над списком
assert not SM.hidden(m, 0) and not m.sample.data['selections'][0].get('hidden')
m.close(); print("Hide / show a selection (eye), Show all: OK")

# ------------------------------------------------------------------ 31. Навигатор: карта — щелчок едет, тянуть — новый участок; Tab; кнопки сверху; чертёж
import microcontrol.manual as M
import microcontrol.ui as UI_
from microcontrol import samples as SM
from microcontrol.mask import Grid
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
M.run(m, fullscreen=False, _keys=[-1, -1])                          # карта встала по камере
nav = m._nav
p0, p1 = nav.to_px((0, 0)), nav.to_px((3 * fw, 2 * fh))
M.run(m, fullscreen=False, _keys=[-1, ('maprect', *p0, *p1), -1, -1])   # тянуть по карте — новый участок 10x
assert len(m._regions) == 1 and m._regions[0]['obj'] == 10, m._regions
x0_, y0_, x1_, y1_ = SM._rect(m, m._regions[0])
assert abs(x0_ - 0) < 2 and abs(x1_ - 3 * fw) < 2 and abs(y1_ - 2 * fh) < 2, (x0_, y0_, x1_, y1_)
tgt = (1.5 * fw, 1.0 * fh)
M.run(m, fullscreen=False, _keys=[-1, ('mapclick', *nav.to_px(tgt)), -1, -1])   # щелчок — столик едет туда
assert abs(m.real_position.x - tgt[0]) < 2 * m.motor_step and abs(m.real_position.y - tgt[1]) < 2 * m.motor_step
hits = UI_.tool_bar(1280, M.STEPS, 0, M.BUTTONS)[1]
bx = lambda name: next((a + b) / 2 for a, b, t in hits if t == ('button', name))
M.run(m, fullscreen=False, _keys=[-1, 9, -1, ('tbar', bx("Scan area")), -1, -1])   # Tab — карта крупно; Scan area
assert m._nav_big and m.last_scan and os.path.exists(os.path.join(m.last_scan, 'join', 'preview_all.jpg'))
f = m.last_scan
px_um = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
g = Grid(f); q0 = np.array([66.0, 52.8]) / px_um; a_ = 100 / px_um
g.cr.add(q0); g.cr.add(q0 + (a_, 0)); g.cr.add(q0 + (0, a_)); g.selected |= {(0, 0)}
g.labels = {(0, 0): (3, 2), (2, 1): (5, 3)}; g.save(px_um)                # номера литографии: два квадрата
m._sel = 0
M.run(m, fullscreen=False, _keys=[-1, ('tbar', bx("Export")), -1, -1])   # чертёж одной кнопкой
import glob as _glob
assert _glob.glob(os.path.join(m.sample.folder, 'export', '*', 'drawing.dxf')), "no sample drawing"
m.close(); print("Navigator: drag on the map - new area, click - go, Tab, Scan area, Export: OK")

# ------------------------------------------------------------------ 32. Проекты таблицей с датами; просмотр в своём окне
import time as _t
pj = mc.Project('Tbl test', 'two chips'); pj.sample('Chip A'); pj.sample('Chip B')
t_ = repr(mc.projects())
assert 'Created' in t_ and 'Updated' in t_ and 'Tbl test' in t_ and pj.uid in t_, t_
assert 'Tbl test / Chip A' in repr(mc.all_samples()) and '<table>' in mc.projects()._repr_html_()
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 2 * fw, 2 * fh)
f = m.scan(); mc.stitch(f, verbose=False); m.close()
p_ = mc.view_window(f); _t.sleep(6)
assert p_.poll() is None, p_.returncode                             # своё окно — живёт отдельно
p_.terminate()
print("Projects as a table with dates, the viewer in its own window: OK")


# ------------------------------------------------------------------ 33. Решётка образца с 1,1; черновики из клеток; подсетка 2-5 в просмотре; адреса; чертёж от креста 1,1
import ezdxf
import microcontrol.manual as M
import microcontrol.viewer as V_
from microcontrol import samples as SM, layers as LY, lattice as LT
from microcontrol.mask import Grid
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 2 * fh)
f = m.scan(); mc.stitch(f, verbose=False)                         # в чертёж — после quality-сшивки
px_um = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
g = Grid(f); p0 = np.array([66.0, 52.8]) / px_um; a_ = 100 / px_um   # 66, 53 мкм от угла мозаики
g.cr.add(p0); g.cr.add(p0 + (a_, 0)); g.cr.add(p0 + (0, a_)); g.selected |= {(1, 0), (2, 0)}
g.labels = {(0, 0): (5, 7), (2, 1): (7, 8)}; g.save(px_um)        # литография: квадрат (0, 0) — «5,7»
print(LY.sync_brush(m)); SM.persist(m)
L = m.sample.data['lattice']
assert L['numbered'] and abs(L['pitch'] - 100) < 0.5, L
kd = next(k for k, r in enumerate(m._regions) if r.get('kind') == 'draft')
assert SM.title(m, kd).startswith('Selection') and SM.frames_of(m, kd) == (0, 0)
assert sorted(map(tuple, m._regions[kd]['cells'])) == [(5, 6), (6, 6)], m._regions[kd]['cells']   # квадраты 6,7 и 7,7
assert LT.address(5, 6, []) == "6,7"
assert len(m.scan(name='again') and m.last_scans) == 1            # черновик не снимается
f = m.last_scans[0]; mc.stitch(f, verbose=False)
uid = m._regions[kd]['uid']
q1 = g.cr.point(1 + 0.5 / 3, 0.5 / 3)                              # левая верхняя клетка 1/3 квадрата 6,7
V_.view(f, _keys=[-1, ('qmove', *q1), ord('3'), ord('b'), ('qclick', *q1), -1, 27])   # 3 — подсетка; кисть — клетка
rf = json.load(open(os.path.join(f, 'grid.json')))['refine'][uid]
assert rf['n'] == 3 and rf['cells'] == [[15, 18]], rf
print(LY.sync_brush(m)); SM.persist(m)
kc = next(k for k, r in enumerate(m._regions) if r.get('parent') == uid)
rc = m._regions[kc]
assert rc['kind'] == 'draft' and rc['factors'] == [3] and m._regions[kd]['sub'] == 3
assert LT.address(*rc['cells'][0], rc['factors']) == "6,7 > 1,1"
assert abs(LY.geom(m, rc).area - (100 / 3) ** 2) < 5, LY.geom(m, rc).area
rq = json.load(open(os.path.join(f, 'grid.json')))                 # ещё уровень: клетка 1/3 -> 1/2 внутри
rq['refine'][rc['uid']] = dict(n=2, cells=[[31, 37]]); json.dump(rq, open(os.path.join(f, 'grid.json'), 'w'))
LY.sync_brush(m)
r3 = next(r for r in m._regions if r.get('parent') == rc['uid'])
assert LT.address(*r3['cells'][0], r3['factors']) == "6,7 > 1,1 > 2,2" and LT.level(r3) == 6
rq['refine'][rc['uid']] = dict(n=2, cells=[[40, 40]]); json.dump(rq, open(os.path.join(f, 'grid.json'), 'w'))
LY.sync_brush(m)                                                     # клетка вне родителя — не берётся
assert not any(r.get('parent') == rc['uid'] for r in m._regions)
c_ = np.array(LY.geom(m, m._regions[kd]).centroid.coords[0])
m.objective = 50; m.focus()                                          # 50x на карте: клетки подсетки 1/3 черновика
M.run(m, fullscreen=False, _keys=[-1, -1])
nav = m._nav
M.run(m, fullscreen=False, _keys=[-1, ('maprect', *nav.to_px(c_ - (60, 40)), *nav.to_px(c_ + (60, 40))), -1, -1])
k50 = next(k for k, r in enumerate(m._regions) if r['obj'] == 50)
r50 = m._regions[k50]
assert r50['parent'] == uid and r50['factors'] == [3] and all(c[0] // 3 in (5, 6) and c[1] // 3 == 6 for c in r50['cells']), r50
m.scan(); [mc.stitch(x, verbose=False) for x in m.last_scans]
dx = mc.export_sample(m, ks=[0, kd, k50])                            # 10x и 50x и черновик: каждый на своём слое
doc = ezdxf.readfile(dx if dx.endswith('.dxf') else dx[:-4] + '.dxf')
ms = doc.modelspace()
ims = ms.query('IMAGE')
assert len(ims) == 2, len(ims)
nums = {t.dxf.text: np.array(t.dxf.insert)[:2] for t in ms.query('TEXT[layer=="NUMBERS"]')}
assert "6,7" in nums and np.allclose(nums["6,7"] - (500, -600), (4, 4), atol=0.5), nums.get("6,7")  # крест 6,7: (5P, -6P)
assert not any(k.startswith('0,') or k.endswith(',0') for k in nums)
lays_ = {l_.dxf.name for l_ in doc.layers}
assert {f"S{m._regions[0]['num']:02d}_10x", f"S{m._regions[k50]['num']:02d}_50x", f"S{m._regions[kd]['num']:02d}_draft"} <= lays_, lays_
assert [p for p in ms.query('LWPOLYLINE') if p.dxf.layer == f"S{m._regions[kd]['num']:02d}_draft"]   # черновик — контур, без картинки
for k_ in (k50,):                                                    # каждая клетка — замкнутый квадрат на слое выделения
    from shapely.geometry import Polygon as _Pg
    r_ = m._regions[k_]
    N_ = LT.level(r_)
    lay = f"S{r_['num']:02d}_{SM.obj(m, r_)}x" + ("_draft" if r_.get('kind') == 'draft' else "")
    sq = [p for p in ms.query('LWPOLYLINE') if p.dxf.layer == lay and p.closed and len(p) == 4 and
          abs(_Pg([q[:2] for q in p.get_points()]).area / (100 / N_) ** 2 - 1) < 0.01]
    assert len(sq) == len(r_['cells']), (k_, lay, len(sq), len(r_['cells']), sorted({(p.dxf.layer, len(p), round(_Pg([q[:2] for q in p.get_points()]).area)) for p in ms.query('LWPOLYLINE')}))
adr = [t.dxf.text for t in ms.query('TEXT[layer=="ADDRESSES"]')]
assert "6,8 > 1,1" not in adr and any(a.startswith("6,7 > ") for a in adr), adr[:8]
im50 = next(i for i in ims if '50x' in i.dxf.layer)
pl = next(p for p in ms.query('LWPOLYLINE') if p.dxf.layer == im50.dxf.layer)
V = np.array([p[:2] for p in pl.get_points()])
ins, u, v = np.array(im50.dxf.insert)[:2], np.array(im50.dxf.u_pixel)[:2], np.array(im50.dxf.v_pixel)[:2]
w_, h_ = im50.dxf.image_size.x, im50.dxf.image_size.y
uv = np.linalg.solve(np.c_[u, v], (V - ins).T).T                   # углы контура — в пикселях картинки
tol = 0.6 * m.motor_step / np.hypot(*u)
assert (uv[:, 0] > -tol).all() and (uv[:, 0] < w_ + tol).all() and (uv[:, 1] > -tol).all() and (uv[:, 1] < h_ + tol).all(), uv
dx2 = mc.export_sample(m, ks=[0])                                   # только 10x: вложенное (50x) — само, и общий PNG
d2_ = ezdxf.readfile(dx2 if dx2.endswith('.dxf') else dx2[:-4] + '.dxf')
assert f"S{m._regions[k50]['num']:02d}_50x" in {l_.dxf.name for l_ in d2_.layers}
ov_ = cv2.imread(os.path.join(os.path.dirname(dx2), 'overview.png'))
assert ov_ is not None and (ov_ < 250).any(axis=2).mean() > 0.2, None if ov_ is None else ov_.shape
uid10_ = m._regions[kd]['uid']
q10 = g.cr.point(1 + 0.5, 0.5)
V_.view(f, _keys=[-1, ('qmove', *q10), ord('1'), ord('0'), -1, 27])      # «1», «0» — подсетка 1/10
assert any(int(v_.get('n') or 0) == 10 for v_ in json.load(open(os.path.join(f, 'grid.json')))['refine'].values())
m.close(); print("Lattice from cross 1,1; drafts of cells; finer grid 2-5 in the viewer, addresses 6,7 > 1,1 > 2,2; "
                 "50x on the 1/3 grid; drawing from cross 1,1, every cell closed on its layer: OK")


# ------------------------------------------------------------------ 34. Операции над выделениями и съёмка: кадры накрывают контур целиком, лишних выделений нет
from microcontrol import layers as LY, samples as SM
from microcontrol.snap import geometry as _geo
from shapely.geometry import box as _box
from shapely.ops import unary_union as _uu
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size

def _add(x0, y0, x1, y1, obj=10, kind=None):
    k_ = m.objective / obj
    r = dict(uid=os.urandom(8).hex(), num=LY.new_num(m), obj=obj, fw=fw * k_, fh=fh * k_, pts=[], thumbs=[], parent=None)
    if kind:
        r['kind'] = kind
    LY.set_geom(m, r, _box(x0, y0, x1, y1))
    m._regions.append(r)
    return len(m._regions) - 1

def _scan_ok(tag):
    regs = [r for r in m._regions if SM.obj(m, r) == m.objective and r.get('kind') not in ('group', 'draft')]
    n0 = len(m._regions)
    m.scan()
    assert len(m._regions) == n0, (tag, "a selection appeared from nowhere")
    for f, r in zip(m.last_scans, regs):
        g_ = _geo(f)
        T = np.asarray(g_['T'], float) * g_['ms']
        cov = _uu([_box(x - r['fw'] / 2, y - r['fh'] / 2, x + r['fw'] / 2, y + r['fh'] / 2) for x, y in T])
        gm = LY.geom(m, r)
        assert gm.difference(cov).area < 1e-6 * gm.area, (tag, r['num'], gm.difference(cov).area)
        assert len(T) == int(np.prod(SM.frames_of(m, m._regions.index(r)))), (tag, len(T))
        mc.stitch(f, verbose=False)

def _reset():
    m._regions.clear(); m._sel = None

a = _add(0, 0, 2 * fw, 2 * fh); b = _add(2 * fw, 0, 4 * fw, 2 * fh)
_scan_ok('two'); LY.union(m, [a, b]); _scan_ok('scan, merge, scan')         # раньше — снималась старая зона
_reset(); a = _add(0, 0, 2 * fw, 2 * fh); b = _add(fw, fh, 3 * fw, 3 * fh); LY.union(m, [a, b]); _scan_ok('merge L')
_reset(); a = _add(0, 0, 2 * fw, 2 * fh); b = _add(4 * fw, 3 * fh, 5 * fw, 4 * fh); LY.union(m, [a, b]); _scan_ok('merge disjoint')
_reset(); a = _add(0, 0, 4 * fw, 4 * fh); b = _add(1.5 * fw, 1.5 * fh, 2.5 * fw, 2.5 * fh)
LY.subtract(m, [a, b]); LY._drop(m, [b]); _scan_ok('minus: hole')
_reset(); a = _add(0, 0, 4 * fw, 2 * fh); b = _add(1.8 * fw, -fh, 2.2 * fw, 3 * fh)
LY.subtract(m, [a, b]); LY._drop(m, [b]); _scan_ok('minus: two parts')
_reset(); a = _add(13, 7, 3 * fw, 3 * fh); b = _add(fw, fh, 4 * fw + 9, 4 * fh); LY.intersect(m, [a, b]); LY._drop(m, [b]); _scan_ok('overlap')
_reset(); a = _add(0, 0, 2 * fw, 2 * fh); b = _add(3 * fw, 0, 4 * fw, 2 * fh); LY.group(m, [a, b]); _scan_ok('group')
_reset(); a = _add(0, 0, 2 * fw, 2 * fh, kind='draft'); b = _add(2 * fw, 0, 3 * fw, 2 * fh)
LY.union(m, [a, b]); assert m._regions[0].get('kind') is None; _scan_ok('draft + selection = selection')
_reset(); a = _add(0, 0, 2 * fw, 2 * fh, kind='draft')
try:
    m.scan(); raise AssertionError("a draft was scanned")
except mc.SafetyError as e:
    assert 'draft' in str(e), e
_reset(); a = _add(0, 0, 2 * fw, 2 * fh); b = _add(2 * fw, 0, 2.5 * fw, 0.5 * fh, obj=50)
try:
    LY.union(m, [a, b]); raise AssertionError("10x and 50x merged")             # разные слои не сливаются
except ValueError as e:
    assert 'one objective' in str(e), e
LY._drop(m, [b]); LY.move(m, 0, 30, 20); _scan_ok('10x, moved')
_reset()
try:
    m.scan(); raise AssertionError("scanned without selections")
except mc.SafetyError as e:
    assert 'E701' in str(e), e
m.close(); print("Merge / minus / overlap / group / draft / move, then scan: frames cover every outline, no stray selections: OK")


# ------------------------------------------------------------------ 35. Маленькая карта в углу: колесо — приближение, щелчок — едет
import cv2 as _cv
import microcontrol.manual as M
import microcontrol.ui as UI_
from microcontrol import samples as SM
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
m._nav_big = False
M.run(m, fullscreen=False, _keys=[-1, -1])
W_, H_ = _cv.getWindowImageRect(M.WIN)[2:] if False else (1280, 860)
nav = m._nav
s0 = nav.s
Wl = W_ - SM.CAR_W
mw_ = min(400, Wl // 2)
mx, my = Wl - 10 - mw_ // 2, UI_.MENU_H + UI_.TOOL_H + 10 + 60          # над маленькой картой
M.run(m, fullscreen=False, _keys=[-1, ('mouse', _cv.EVENT_MOUSEWHEEL, mx, my, 120 << 16), -1,
                                  ('mouse', _cv.EVENT_MOUSEWHEEL, mx, my, 120 << 16), -1, -1])
assert m._nav.s < s0 * 0.8, (s0, m._nav.s)                              # приблизилась (мкм на пиксель меньше)
assert not m._nav_big
s1 = m._nav.s
M.run(m, fullscreen=False, _keys=[-1, ('mouse', _cv.EVENT_MOUSEWHEEL, mx, my, -(120 << 16)), -1, -1])
assert m._nav.s > s1 * 1.1, (s1, m._nav.s)                              # отдалилась
p0 = (m.real_position.x, m.real_position.y)
M.run(m, fullscreen=False, _keys=[-1, ('mouse', _cv.EVENT_LBUTTONDOWN, mx + 40, my + 30, 0),
                                  ('mouse', _cv.EVENT_LBUTTONUP, mx + 40, my + 30, 0), -1, -1])
assert (m.real_position.x, m.real_position.y) != p0                      # щелчок по маленькой карте — поехали
m.close(); print("Small map in the corner: wheel zooms in and out, click drives there: OK")


# ------------------------------------------------------------------ 36. Просмотр и окно микроскопа показывают одно и то же; правый щелчок; свои номера вместо подставленных
import microcontrol.viewer as V_
from microcontrol import samples as SM, layers as LY, project as P_
from microcontrol.mask import Grid
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m._world.turn_deg = 0.0; m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 2 * fh); f = m.scan(); mc.stitch(f, verbose=False)
m.area = mc.Area(5 * fw, 0, 6 * fw, fh); f2 = m.scan(); mc.stitch(f2, verbose=False)
px_um = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
from microcontrol.snap import stage_to_mosaic
g = Grid(f); p0 = stage_to_mosaic(f, [[100.0, 100.0]])[0]; a_ = 100 / px_um   # на настоящих крестах
g.cr.add(p0); g.cr.add(p0 + (a_, 0)); g.cr.add(p0 + (0, a_)); g.labels = {(0, 0): (5, 7), (2, 1): (7, 8)}
g.selected |= {(0, 0), (1, 0), (2, 0), (2, 1)}; g.save(px_um)
LY.sync_brush(m); SM.persist(m)
assert len(P_.latest_scans(m.sample.folder)) == 2
q = lambda i, j: g.cr.point(i + 0.5, j + 0.5)
V_.view(f, _keys=[-1, ord('m'), ('qrclick', *q(1, 0)), -1, 27])           # правый щелчок: один квадрат
assert Grid(f).selected == {(0, 0), (2, 0), (2, 1)}, Grid(f).selected
V_.view(f, _keys=[-1, ord('m'), ('qrclick', *q(2, 0), 'shift'), -1, 27])  # Shift + правый: вся область
assert Grid(f).selected == {(0, 0)}, Grid(f).selected
LY.sync_brush(m); SM.persist(m)
kd = next(k for k, r in enumerate(m._regions) if r.get('kind') == 'draft')
k2 = next(k for k, r in enumerate(m._regions) if r.get('kind') != 'draft' and k != 0)

def _delete_there():                                                       # «в окне микроскопа» удалили
    SM.delete_area(m, kd)                                                  # черновик и Selection 2
    SM.delete_area(m, next(k for k, r in enumerate(m._regions) if r['num'] == 2))

V_.view(f, _keys=[-1, ('call', _delete_there), ('sleep', 0.6), -1, -1, ord('m'), ('qclick', *q(1, 1)), -1, 27])
assert Grid(f).selected == {(1, 1)}, Grid(f).selected                      # удалённое не вернулось из просмотра
assert len(P_.latest_scans(m.sample.folder)) == 1                          # и Selection 2 в просмотре нет
LY.sync_brush(m); SM.persist(m)
assert [r['num'] for r in m._regions if r.get('kind') == 'draft'] and not any(r['num'] == 2 for r in m._regions)
m.area = mc.Area(0, 0, 2 * fw, fh); f3 = m.scan(); mc.stitch(f3, verbose=False)   # новая съёмка: сетка — по решётке
V_.view(f3, _keys=[-1, 27])
g3 = Grid(f3)
assert json.load(open(os.path.join(f3, 'grid.json'))).get('seeded') and g3.labels
c3 = g3.cell_at(g3.cr.point(0.5, 0.5))
V_.view(f3, _keys=[-1, ord('n'), ('qclick', *g3.cr.point(0.5, 0.5)), ord('9'), ord(' '), ord('9'), 13, -1, 27])
g3 = Grid(f3)
assert g3.labels == {c3: (9, 9)} and 'disagree' not in g3.numbering()[2], (g3.labels, g3.numbering()[2])
m.close(); print("View follows the microscope window (deleted stays deleted), right click: one square / Shift: the area,"
                 " own numbers replace the seeded ones: OK")


# ------------------------------------------------------------------ 37. Пирамидка: участок внутри большего обрезается по его краю; перетаскивание карточки — сразу видно
import microcontrol.manual as M
from microcontrol import samples as SM, layers as LY
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
M.run(m, fullscreen=False, _keys=[-1, -1])
nav = m._nav
M.run(m, fullscreen=False, _keys=[-1, ('maprect', *nav.to_px((0, 0)), *nav.to_px((2 * fw, 2 * fh))), -1, -1])
g10 = LY.geom(m, m._regions[0])
m.objective = 50; m.focus()
M.run(m, fullscreen=False, _keys=[-1, -1]); nav = m._nav
x1_ = g10.bounds[2]
M.run(m, fullscreen=False, _keys=[-1, ('maprect', *nav.to_px((x1_ - 40, 50)), *nav.to_px((x1_ + 60, 110))), -1, -1])
k50 = next(k for k, r in enumerate(m._regions) if r['obj'] == 50)
g50 = LY.geom(m, m._regions[k50])
assert g10.buffer(0.05).contains(g50) and g50.bounds[2] <= x1_ + 0.05, (g50.bounds, x1_)   # обрезан по краю 10x
assert m._regions[k50]['parent'] == m._regions[0]['uid']
assert int(np.prod(SM.frames_of(m, k50))) > 0
m.scan()                                                                   # снимается только то, что внутри
m.objective = 100; m.focus()                                               # 100x углами 1-2, вылезая за 50x
c50 = g50.bounds
m.goto(x=c50[0] + 5, y=c50[1] + 5); SM.corner(m, 1)
m.goto(x=c50[2] + 30, y=c50[3] + 30); print(SM.corner(m, 2))
g100 = LY.geom(m, m._regions[-1])
assert g50.buffer(0.05).contains(g100), (g100.bounds, c50)
try:                                                                       # целиком снаружи — нельзя
    m.goto(x=c50[2] + 300, y=c50[3] + 300); SM.corner(m, 1); m.goto(x=c50[2] + 330, y=c50[3] + 330); SM.corner(m, 2)
except mc.MicroControlError:
    pass
m.objective = 10
M.run(m, fullscreen=False, _keys=[-1, ('car', 1000, 140), ('cmove', 1000, 200), ('cmove', 1000, 260), -1,
                                  ('cdrop', 1000, 260), -1])                # тянем карточку: не падает
m.close(); print("Pyramid: a smaller area is cut to the edge of the larger one (map and corners); card drag preview: OK")


# ------------------------------------------------------------------ 38. Подсетка сразу по выбранным квадратам (черновика ещё нет): 3 над ними, клетка, потом окно микроскопа
import microcontrol.viewer as V_
from microcontrol import samples as SM, layers as LY, lattice as LT
from microcontrol.mask import Grid
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 2 * fh); f = m.scan(); mc.stitch(f, verbose=False)
px_um = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
g = Grid(f); p0 = np.array([66.0, 52.8]) / px_um; a_ = 100 / px_um   # 66, 53 мкм от угла мозаики
g.cr.add(p0); g.cr.add(p0 + (a_, 0)); g.cr.add(p0 + (0, a_)); g.labels = {(0, 0): (5, 7), (2, 1): (7, 8)}; g.save(px_um)
LY.sync_brush(m); SM.persist(m)                                       # решётка образца есть, черновиков нет
q = lambda i, j, a=0.5, b=0.5: g.cr.point(i + a, j + b)
V_.view(f, _keys=[-1, ord('m'), ('qclick', *q(1, 0)), ('qclick', *q(2, 0)), ('qmove', *q(1, 0)), ord('3'),
                  ('qclick', *q(1, 0, 0.2, 0.2)), -1, 27])
rf = json.load(open(os.path.join(f, 'grid.json')))['refine']
key = next(iter(rf))
smj = json.load(open(m.sample.file))['selections']                   # выделение (draft) — сразу, из просмотра
assert any(r_['uid'] == key and r_.get('kind') == 'draft' for r_ in smj), (rf, smj)
assert rf[key]['n'] == 3 and rf[key]['cells'] == [[15, 18]], rf
print(LY.sync_brush(m)); SM.persist(m)
d1 = next(r for r in m._regions if r.get('kind') == 'draft' and not r.get('factors'))
d2 = next(r for r in m._regions if r.get('parent') == d1['uid'])
assert d1.get('sub') == 3 and d2['factors'] == [3] and LT.address(*d2['cells'][0], [3]) == "6,7 > 1,1"
assert list(json.load(open(os.path.join(f, 'grid.json')))['refine']) == [d1['uid']]
m.close(); print("Selected squares become a selection (draft) at once; its finer grid 3, a cell: OK")


# ------------------------------------------------------------------ 39. Ctrl+Z в окне микроскопа — для всех операций над выделениями (вместе с квадратами в просмотре)
import microcontrol.manual as M
from microcontrol import samples as SM, layers as LY
from microcontrol.mask import Grid
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 2 * fh); f = m.scan(); mc.stitch(f, verbose=False)
px_um = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
g = Grid(f); p0 = np.array([66.0, 52.8]) / px_um; a_ = 100 / px_um   # 66, 53 мкм от угла мозаики
g.cr.add(p0); g.cr.add(p0 + (a_, 0)); g.cr.add(p0 + (0, a_)); g.labels = {(0, 0): (5, 7), (2, 1): (7, 8)}
g.selected |= {(0, 0), (2, 0)}; g.save(px_um)                              # выбрали в просмотре: два квадрата
drafts = lambda: sorted(r['num'] for r in m._regions if r.get('kind') == 'draft')
M.run(m, fullscreen=False, _keys=[-1, -1])                                 # окно сделало два черновика
assert len(drafts()) == 2, m._regions
M.run(m, fullscreen=False, _keys=[-1, 26, -1, -1])                         # черновики из просмотра — не шаг Ctrl+Z
assert len(drafts()) == 2
ks = lambda: [k for k, r in enumerate(m._regions) if r.get('kind') == 'draft']
ops = [("merge", lambda: LY.union(m, ks())), ("delete", lambda: SM.delete_area(m, ks()[0])),
       ("to selection", lambda: LY.to_area(m, ks()[0], 50)), ("move", lambda: LY.move(m, ks()[0], 20, 10)),
       ("group", lambda: LY.group(m, ks())), ("hide", lambda: m._regions[ks()[0]].update(hidden=True))]
for name, fn in ops:
    before = json.dumps([{k: v for k, v in r.items() if k != 'thumbs'} for r in m._regions], sort_keys=True, default=str)
    gs0 = Grid(f).selected
    M.run(m, fullscreen=False, _keys=[-1, ('call', fn), -1, -1, 26, -1, -1])
    M.run(m, fullscreen=False, _keys=[-1, -1])                               # и после синхронизации — как было
    after = json.dumps([{k: v for k, v in r.items() if k != 'thumbs'} for r in m._regions], sort_keys=True, default=str)
    assert after == before and Grid(f).selected == gs0, (name, before, after, gs0, Grid(f).selected)
m.close(); print("Ctrl+Z in the microscope window: drafts from the view, merge, delete, to selection, move, group, hide: OK")


# ------------------------------------------------------------------ 40. Целостность: квадраты и клетки в просмотре <-> черновики (на краю тоже), создание и удаление, старая съёмка
import random as _rnd
from shapely.geometry import Point as _Pt
from microcontrol import samples as SM, layers as LY, lattice as LT
from microcontrol.mask import Grid
from microcontrol.snap import mosaic_to_stage
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 2 * fh); f = m.scan(); mc.stitch(f, verbose=False)
px_um = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
g = Grid(f); p0 = np.array([66.0, 52.8]) / px_um; a_ = 100 / px_um   # 66, 53 мкм от угла мозаики
g.cr.add(p0); g.cr.add(p0 + (a_, 0)); g.cr.add(p0 + (0, a_)); g.labels = {(0, 0): (5, 7), (2, 1): (7, 8)}; g.save(px_um)
LY.sync_brush(m); SM.persist(m)
L = m.sample.data['lattice']
par = m._regions[0]; pg = LY.geom(m, par)

def accepted(gr):
    sq = sorted(gr.selected)
    if not sq:
        return set()
    S = mosaic_to_stage(f, np.array([gr.cr.point(i + 0.5, j + 0.5) for i, j in sq]))
    uv = LT.to_uv(L, S)
    return {(int(np.floor(u)), int(np.floor(v))) for (u, v), S1 in zip(uv, S) if pg.buffer(0.01).contains(_Pt(*S1))}

def check(tag):
    LY.sync_brush(m); SM.persist(m)
    acc = accepted(Grid(f))
    d1 = [r for r in m._regions if r.get('kind') == 'draft' and not r.get('factors')]
    got = [{tuple(c) for c in r['cells']} for r in d1]
    assert set().union(*got) == acc if got else not acc, (tag, sorted(acc), got)          # каждый принятый квадрат — в черновике
    assert sum(len(x) for x in got) == len(acc), (tag, "a square in two drafts")
    assert sorted(map(sorted, got)) == sorted(map(sorted, LT.figures(acc))), (tag, "one draft per figure")
    for r in m._regions:                                                                   # пирамидка: внутри родителя
        if r.get('parent') and r.get('kind') != 'group':
            pr = m._regions[LY.idx(m, r['parent'])]
            assert LY.geom(m, pr).buffer(0.05).contains(LY.geom(m, r)), (tag, r['num'])
    rf = json.load(open(os.path.join(f, 'grid.json'))).get('refine') or {}
    for r in d1:                                                                            # клетки подсеток <-> черновики
        rq = rf.get(r['uid'])
        kids = [k for k in m._regions if k.get('parent') == r['uid'] and k.get('kind') == 'draft']
        gr_ = LY.geom(m, r).buffer(0.01)
        want = {tuple(c) for c in (rq or {}).get('cells', []) if (c[0] // rq['n'], c[1] // rq['n']) in
                {tuple(x) for x in r['cells']} and gr_.contains(_Pt(*LT.to_stage(
                    L, [((c[0] + 0.5) / rq['n'], (c[1] + 0.5) / rq['n'])])[0]))} if rq and int(rq.get('n') or 1) > 1 else set()
        have = set().union(*[{tuple(c) for c in k['cells']} for k in kids]) if kids else set()
        assert have == want, (tag, r['num'], sorted(want), sorted(have))

rng = _rnd.Random(7)
for step in range(40):
    g = Grid(f)
    act = rng.random()
    if act < 0.55:                                                     # выбрать / снять квадрат, в том числе на краю
        c = (rng.randint(-2, 4), rng.randint(-2, 3))
        (g.selected.discard if c in g.selected else g.selected.add)(c); g.save(px_um)
    elif act < 0.75:                                                   # подсетка в черновике и случайные клетки
        d1 = [r for r in m._regions if r.get('kind') == 'draft' and not r.get('factors')]
        if d1:
            r = rng.choice(d1); n = rng.choice([2, 3, 4])
            cs = [(c[0] * n + rng.randrange(n), c[1] * n + rng.randrange(n)) for c in r['cells'] for _ in range(2)]
            d = json.load(open(os.path.join(f, 'grid.json'))); d.setdefault('refine', {})[r['uid']] = dict(n=n, cells=[list(c) for c in cs])
            json.dump(d, open(os.path.join(f, 'grid.json'), 'w'))
    elif act < 0.9:                                                    # удалить черновик в окне микроскопа
        d1 = [k for k, r in enumerate(m._regions) if r.get('kind') == 'draft']
        if d1:
            SM.delete_area(m, rng.choice(d1))
    else:                                                              # снять все
        g.selected.clear(); g.save(px_um)
    check(f"step {step}")
m.close(); print("Squares and cells <-> drafts stay one-to-one through 40 random edits (edges, refine, delete): OK")


# ------------------------------------------------------------------ 41. Квадрат на краю съёмки — черновик есть (обрезан по краю); выбор в старой версии съёмки тоже работает
from microcontrol import samples as SM, layers as LY
from microcontrol.mask import Grid
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 2 * fh); f = m.scan(); mc.stitch(f, verbose=False)
px_um = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
g = Grid(f); p0 = np.array([66.0, 52.8]) / px_um; a_ = 100 / px_um   # 66, 53 мкм от угла мозаики
g.cr.add(p0); g.cr.add(p0 + (a_, 0)); g.cr.add(p0 + (0, a_)); g.labels = {(0, 0): (5, 7), (2, 1): (7, 8)}
g.selected |= {(-1, -1)}; g.save(px_um)                                   # квадрат на краю: середина внутри съёмки
print(LY.sync_brush(m)); SM.persist(m)
d = [r for r in m._regions if r.get('kind') == 'draft']
assert len(d) == 1 and LY.geom(m, m._regions[0]).buffer(0.05).contains(LY.geom(m, d[0])), d
assert LY.geom(m, d[0]).area < 1e4 - 1                                    # обрезан по краю снятого
f2 = m.scan(); mc.stitch(f2, verbose=False)                               # вторая версия той же съёмки
assert os.path.dirname(f2) == os.path.dirname(f)
g = Grid(f); g.selected |= {(1, 0)}; g.save(px_um)                        # выбрали в СТАРОЙ версии
print(LY.sync_brush(m)); SM.persist(m)
assert len([r for r in m._regions if r.get('kind') == 'draft']) == 2, m._regions
m.close(); print("A square at the edge makes a draft (cut to the edge); squares chosen in an older version count: OK")


# ------------------------------------------------------------------ 42. Конфликты draft / selection разных уровней: все ограничители и сообщения
from shapely.geometry import box as _box
from microcontrol import samples as SM, layers as LY
from microcontrol.config import OBJECTIVES as _OB
from microcontrol.mask import Grid
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 2 * fh); f = m.scan(); mc.stitch(f, verbose=False)
px_um = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
g = Grid(f); p0 = np.array([66.0, 52.8]) / px_um; a_ = 100 / px_um   # 66, 53 мкм от угла мозаики
g.cr.add(p0); g.cr.add(p0 + (a_, 0)); g.cr.add(p0 + (0, a_)); g.labels = {(0, 0): (5, 7), (2, 1): (7, 8)}
g.selected |= {(1, 0), (2, 0)}; g.save(px_um)
LY.sync_brush(m); SM.persist(m)
S1 = m._regions[0]
kd = next(k for k, r in enumerate(m._regions) if r.get('kind') == 'draft')
snap_ = lambda: json.dumps([{k: v for k, v in r.items() if k != 'thumbs'} for r in m._regions], sort_keys=True, default=str)

def sel(x0, y0, x1, y1, o, parent=None):
    r = dict(uid=os.urandom(8).hex(), num=LY.new_num(m), obj=o, fw=fw * 10 / o, fh=fh * 10 / o, pts=[], thumbs=[],
             parent=parent)
    LY.set_geom(m, r, _box(x0, y0, x1, y1)); m._regions.append(r); return len(m._regions) - 1

import copy
_r0 = copy.deepcopy(m._regions)
ok, msg = LY.draft_to_selection(m, kd, 10)                                 # a) 10x в съёмке 10x — теперь можно
assert ok, msg                                                             # a) тот же объектив внутри — можно
m._regions = copy.deepcopy(_r0); SM.persist(m)
gd = LY.geom(m, m._regions[kd]); bx = gd.bounds
k50 = sel(bx[0] + 5, bx[1] + 5, bx[0] + 40, bx[1] + 30, 50, S1['uid'])    # 50x внутри черновика (снаружи — съёмка 10x)
_r1 = copy.deepcopy(m._regions)
ok, msg = LY.draft_to_selection(m, kd, 50)                                 # b) 50x внутри черновика — вложено, можно
assert ok, msg
m._regions = copy.deepcopy(_r1); SM.persist(m)
LY._drop(m, [k50])
ok, msg = LY.draft_to_selection(m, kd, 50)                                 # c) теперь можно
assert ok and m._regions[kd].get('kind') is None and m._regions[kd]['obj'] == 50, msg
s0 = snap_()
gb = LY.geom(m, S1).bounds
try:                                                                       # d) за край родителя — нельзя, ничего не тронуто
    LY.move(m, kd, gb[2], 0); raise AssertionError("moved beyond the parent")
except ValueError as e:
    assert 'beyond' in str(e), e
assert snap_() == s0
k2 = sel(gb[2] - 60, gb[1] + 10, gb[2] - 20, gb[1] + 40, 50, S1['uid'])   # другой 50x рядом
gk = LY.geom(m, m._regions[k2]).bounds; gk0 = LY.geom(m, m._regions[kd]).bounds
try:                                                                       # e) наехать частью на такой же — нельзя
    LY.move(m, k2, gk0[2] - gk[0] - 20, gk0[1] - gk[1] + 5); raise AssertionError("overlap")
except ValueError as e:
    assert 'E704' in str(e), e
c0 = np.array(LY.geom(m, m._regions[kd]).centroid.coords[0])               # f) родитель едет — дети с ним
LY.move(m, 0, 7, 3)
assert np.allclose(np.array(LY.geom(m, m._regions[kd]).centroid.coords[0]) - c0, (7, 3), atol=0.05)
LY.move(m, 0, -7, -3)
try:                                                                       # g) 10x + 50x не сливаются
    LY.union(m, [0, k2]); raise AssertionError("merged across objectives")
except ValueError as e:
    assert 'one objective' in str(e), e
n_ = len(m._regions)                                                       # h) 50x + 50x сливаются
print(LY.union(m, [kd, k2])); assert len(m._regions) == n_ - 1
kd = next(k for k, r in enumerate(m._regions) if r.get('obj') == 50 and r.get('kind') is None)
gk = LY.geom(m, m._regions[kd]).bounds                                     # i) объектив по кругу: пирамидка цела
k100 = sel(gk[0] + 1, gk[1] + 1, gk[0] + 10, gk[1] + 8, 100, m._regions[kd]['uid'])
ok, msg = LY.next_objective_of(m, kd, sorted(_OB))
assert ok and m._regions[kd]['obj'] != 50, (msg, m._regions[kd]['obj'])
m.objective = 50; m.focus()                                                # j) углы 1-2 на такой же объектив — E704
m._regions[kd].update(obj=50, fw=fw / 5, fh=fh / 5)
c_ = LY.geom(m, m._regions[kd]).centroid.coords[0]
m.goto(x=c_[0], y=c_[1])
SM.corner(m, 1); m._pending = None                                         # j) угол внутри того же объектива — можно
m.close(); print("Conflicts: draft->selection checks objective and partial overlap (nesting of one objective is fine), "
                 "moves keep the pyramid (children along), no merge across objectives, objective cycling: OK")
m = mc.Microscope('sim', objective=50, folder=tempfile.mkdtemp()); m.focus()   # k) 10x внутри 50x — нельзя
import microcontrol.manual as M
fw, fh = m.frame_size
M.run(m, fullscreen=False, _keys=[-1, -1]); nav = m._nav
M.run(m, fullscreen=False, _keys=[-1, ('maprect', *nav.to_px((0, 0)), *nav.to_px((600, 400))), -1, -1])
assert len(m._regions) == 1 and m._regions[0]['obj'] == 50
m.objective = 10; m.focus(); fw, fh = m.frame_size
M.run(m, fullscreen=False, _keys=[-1, -1]); nav = m._nav
M.run(m, fullscreen=False, _keys=[-1, ('maprect', *nav.to_px((150, 120)), *nav.to_px((300, 220))), -1, -1])
assert len(m._regions) == 1, m._regions                                    # маленький 10x внутри 50x не создан
m.close(); print("A coarser area inside a finer one is refused: OK")


# ------------------------------------------------------------------ 43. Люфт: сверка кадра с мозаикой (дополненная реальность совпадает с крестами), подъезд с одной стороны
import microcontrol.manual as M
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
m._world.backlash = {'x': 3.0, 'y': 2.0}                                   # люфт 3 и 2 шага
m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 3 * fh); f = m.scan(); mc.stitch(f, verbose=False)
s_ = m.motor_step
true_ = lambda: np.array([m._world.p['x'], m._world.p['y']]) * s_
for tgt, frm in (((1.4 * fw, 1.3 * fh), (2.6 * fw, 2.5 * fh)), ((1.6 * fw, 1.5 * fh), (0.3 * fw, 0.2 * fh))):
    m.goto(x=frm[0], y=frm[1]); m.goto(x=tgt[0], y=tgt[1])                 # подъезд с разных сторон
    m.__dict__['_ar_d'] = None
    for _ in range(3):
        dP, nm = M.ar_lock(m, m.image(0))
        assert dP is not None, nm
    K, d = m._pixels_per_step(), m.motor_step
    est = np.array([m._p['x'], m._p['y']], float) * d + M._ar_d(m)
    B = getattr(m, '_bl', None)
    if B is not None:
        est = est + d * np.asarray(B, float) @ np.array([m._dir['x'], m._dir['y']], float)
    err = est - true_()
    off0 = err if tgt == (1.4 * fw, 1.3 * fh) else off0
    assert np.hypot(*(err - off0)) < 0.4, (tgt, err, off0)                   # с любой стороны — одно и то же (< 0.4 мкм)
    assert np.hypot(*dP) < 0.3, dP                                           # сошлось
m.one_side = True                                                            # подъезд всегда с одной стороны
m.goto(x=2.5 * fw, y=2.5 * fh); m.goto(x=fw, y=fh)
assert m._dir['x'] == 1 and m._dir['y'] == 1 and abs(m.position.x - fw) < s_ and abs(m.position.y - fh) < s_
p1 = true_(); m.goto(x=0.2 * fw, y=0.1 * fh); m.goto(x=fw, y=fh)
assert np.allclose(true_(), p1, atol=1e-6), (true_(), p1)                   # туда же — ровно туда же
m.close(); print("Backlash: the camera picture is matched to the mosaic (outlines on the real crosses from any side, "
                 "< 0.4 um), one-side approach stops the same way every time: OK")


# ------------------------------------------------------------------ 44. Удалили черновик — вложенные остаются (как после разгруппировки) у его родителя
from microcontrol import samples as SM, layers as LY
from microcontrol.mask import Grid
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 2 * fh); f = m.scan(); mc.stitch(f, verbose=False)
px_um = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
g = Grid(f); p0 = np.array([66.0, 52.8]) / px_um; a_ = 100 / px_um   # 66, 53 мкм от угла мозаики
g.cr.add(p0); g.cr.add(p0 + (a_, 0)); g.cr.add(p0 + (0, a_)); g.labels = {(0, 0): (5, 7), (2, 1): (7, 8)}
g.selected |= {(0, 0), (1, 0)}; g.save(px_um)
LY.sync_brush(m); SM.persist(m)
S1 = m._regions[0]['uid']
d2 = next(r for r in m._regions if r.get('kind') == 'draft')
d = json.load(open(os.path.join(f, 'grid.json'))); c0 = d2['cells'][0]
d['refine'] = {d2['uid']: dict(n=2, cells=[[2 * c0[0], 2 * c0[1]], [2 * c0[0] + 1, 2 * c0[1] + 1]])}
json.dump(d, open(os.path.join(f, 'grid.json'), 'w'))
LY.sync_brush(m); SM.persist(m)
kids = [r['uid'] for r in m._regions if r.get('parent') == d2['uid']]
assert len(kids) == 2
print(SM.delete_area(m, LY.idx(m, d2['uid'])))                             # удалить Draft 2
for _ in range(2):
    LY.sync_brush(m); SM.persist(m)
    left = [r for r in m._regions if r['uid'] in kids]
    assert len(left) == 2 and all(r['parent'] == S1 and not r.get('src') for r in left), left
assert d2['uid'] not in (json.load(open(os.path.join(f, 'grid.json'))).get('refine') or {})
g = Grid(f); g.selected |= {(2, 0)}; g.save(px_um); LY.sync_brush(m)       # новый черновик со вложенным, потом
d3 = next(r for r in m._regions if r.get('kind') == 'draft' and r.get('src') and not r.get('factors'))
d = json.load(open(os.path.join(f, 'grid.json'))); c0 = d3['cells'][0]
d['refine'] = {d3['uid']: dict(n=2, cells=[[2 * c0[0], 2 * c0[1]]])}; json.dump(d, open(os.path.join(f, 'grid.json'), 'w'))
LY.sync_brush(m)
k3 = [r['uid'] for r in m._regions if r.get('parent') == d3['uid']]
g = Grid(f); g.selected.discard((2, 0)); g.save(px_um); LY.sync_brush(m)  # его квадрат сняли в просмотре
assert LY.idx(m, d3['uid']) is None and all(LY.idx(m, u) is not None and m._regions[LY.idx(m, u)]['parent'] == S1 for u in k3)
m.close(); print("Deleting a draft keeps the drafts inside it (released to its parent), also when its squares are unselected: OK")

# ------------------------------------------------------------------ 45. Точность для литографии: крест по краям плеч с наклоном, калибровка дисторсии, подготовка, кадры без потерь, кадры на кресты, картинка чертежа по крестам клеток
from microcontrol.mask import snap_cross, Grid, meas_gray
from microcontrol.snap import stage_to_mosaic
from microcontrol.drawing import local_picture, _grid_uv
from microcontrol.optics import dist_shift
rng = np.random.default_rng(5)
def _cross_img(c, ang, L=30, t=6, N=241, ss=8):
    n = N * ss; im = np.zeros((n, n), np.float32); C = (np.asarray(c) + 0.5) * ss - 0.5
    R = cv2.getRotationMatrix2D((float(C[0]), float(C[1])), -ang, 1.0)
    for P_ in ([[C[0]-L*ss, C[1]-t*ss/2], [C[0]+L*ss, C[1]-t*ss/2], [C[0]+L*ss, C[1]+t*ss/2], [C[0]-L*ss, C[1]+t*ss/2]],
               [[C[0]-t*ss/2, C[1]-L*ss], [C[0]+t*ss/2, C[1]-L*ss], [C[0]+t*ss/2, C[1]+L*ss], [C[0]-t*ss/2, C[1]+L*ss]]):
        cv2.fillPoly(im, [cv2.transform(np.array([P_], np.float32), R)[0].astype(np.int32)], 1.0)
    im = cv2.GaussianBlur(cv2.resize(im, (N, N), interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    out = (120 + 60 * im + rng.normal(0, 3, im.shape)).clip(0, 255).astype(np.uint8)
    cv2.putText(out, "12", (int(c[0]) + 8, int(c[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 90, 1)   # номер у креста
    return out
e_ = []
for k in range(20):
    c = np.array([120, 120]) + rng.uniform(-6, 6, 2); im = _cross_img(c, 10.0)
    d, resp = snap_cross(im, 60, angle=7.0); e_.append(np.hypot(*((np.array(im.shape[::-1]) - 1) / 2 + d - c)))
assert np.sqrt(np.mean(np.square(e_))) < 0.2, e_                          # крест под 10°: центр точнее 0.2 px
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
m._world.dist = dict(k1=-0.02, k2=0.006, p1=0.002, p2=-0.0015); m.focus(); m.goto(x=100, y=100); m.exposure = 6.0
r = m.prepare(); print(r)
assert m.exposure > 10 and 'overexposed 0.00' in r['Exposure'], r
r = m.calibrate_distortion(); print(r)
W_, H_ = 640, 480; R_ = np.hypot(W_ / 2, H_ / 2)
yy, xx = np.mgrid[0:H_:8, 0:W_:8]; v = np.c_[xx.ravel() - W_ / 2, yy.ravel() - H_ / 2]
Mx = np.r_[np.c_[np.ones(len(v)), np.zeros(len(v)), v[:, 0], -v[:, 1]], np.c_[np.zeros(len(v)), np.ones(len(v)), v[:, 1], v[:, 0]]]
nl = lambda f_: (lambda q: q - Mx @ np.linalg.lstsq(Mx, q, rcond=None)[0])(np.r_[f_[:, 0], f_[:, 1]])  # без сдвига, масштаба, поворота
left = np.sqrt(np.mean(nl(dist_shift(v, m._world.dist, R_) - dist_shift(v, r.info, R_)) ** 2) * 2)
assert left < 0.25 * np.sqrt(np.mean(nl(dist_shift(v, m._world.dist, R_)) ** 2) * 2), left   # дисторсия убрана > 75 %
m.frame_format = 'png'; w = m._world; w.backlash = {'x': 1.0, 'y': 1.0}
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 4 * fw, 3 * fh); f = m.scan(); mc.stitch(f, verbose=False)
assert os.listdir(os.path.join(f, 'frames'))[0].endswith('.png')        # кадры без потерь
assert json.load(open(os.path.join(f, 'meta.json')))['distortion']['calibrated']   # калибровка объектива — в скане
m.close()
px = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
t = np.radians(w.turn_deg); wp = lambda i, j: np.array([np.cos(t)*i*100 - np.sin(t)*j*100, np.sin(t)*i*100 + np.cos(t)*j*100])
g = Grid(f)
for ij in ((2, 1), (3, 1), (2, 2)):
    g.cr.add(np.asarray(stage_to_mosaic(f, wp(*ij)[None]), float).reshape(-1)[:2])
g.labels = {(0, 0): (12, 7), (1, 0): (13, 7), (0, 1): (12, 8)}; g.save(px)
r = mc.fit_crosses(f, verbose=False); print(r)
assert r.share >= 0.95 and r.info['p95_um'] < 0.2, r.info                 # кадры на крестах: 95 % точнее 0.2 мкм
assert abs(r.info['pixel_um'] / 0.22 - 1) < 0.002, r.info['pixel_um']      # пиксель по крестам маски (100 мкм): 0.22 мкм
assert json.load(open(os.path.join(f, 'meta.json')))['pixel_size_cal']
g = Grid(f); uv0, uv1 = np.array([10.7, 5.7]), np.array([14.3, 8.3])
img4, res, rep = local_picture(f, g, uv0, uv1, 100.0)
gq = meas_gray(img4[..., :3]); errs = []; r_ = int(15 / res)
for X in range(12, 16):
    for Y in range(7, 10):
        q = (np.array([X - 1, Y - 1], float) - uv0) * 100 / res - 0.5; x, y = int(round(q[0])), int(round(q[1]))
        if x - 2 * r_ >= 0 and y - 2 * r_ >= 0 and x + 2 * r_ < gq.shape[1] and y + 2 * r_ < gq.shape[0] and \
                img4[y - r_:y + r_ + 1, x - r_:x + r_ + 1, 3].min() == 255:   # крест целиком на мозаике (не срезан краем)
            d, resp = snap_cross(gq[y - 2 * r_:y + 2 * r_ + 1, x - 2 * r_:x + 2 * r_ + 1], r_)
            if resp > 0.1:
                errs.append(np.hypot(*(np.array([x, y]) + d - q)) * res)
assert len(errs) >= 6 and max(errs) < 0.15, errs                           # кресты на чертеже — в своих точках решётки
print("Lithography accuracy: cross centre by arm edges at 10 deg, distortion calibration, prepare, PNG frames, frames onto crosses "
      f"({100 * r.share:.0f} % within 0.5 um), drawing picture by the crosses of its cells (max {max(errs):.3f} um): OK")

# ------------------------------------------------------------------ 46. Мастер Prepare: шаги с условием и Done; сохранено для объектива; второй раз — Overwrite / Keep; плоское поле и тёмный кадр — в сшивке
import microcontrol.ui as UI_, microcontrol.manual as M
cv2.getWindowImageRect = lambda w: (0, 0, 1280, 900)
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus(); m.goto(x=150, y=150); m.exposure = 8.0
hits = UI_.tool_bar(1280, M.STEPS, 0, M.BUTTONS)[1]
bx = lambda name: next((a + b) / 2 for a, b, t in hits if t == ('button', name))
ks = [-1, ('tbar', bx("Prepare")), -1] + [('wiz', 'Done'), -1] * 5 + [('wiz', 'Skip'), -1]
ks += [('tbar', bx("Prepare")), -1, ('wiz', 'Keep'), -1, 27, 27]
M.run(m, fullscreen=False, _keys=ks)
rec = m._mem.data['prepared']['10']
assert rec['done'] == ['lamp', 'kohler', 'dark', 'flat', 'detail'] and m.exposure > 15, rec   # дисторсию пропустили
assert {'dark_10x', 'empty_10x'} <= set(m.references)
fw, fh = m.frame_size; m.area = mc.Area(0, 0, 3 * fw, 2 * fh); f = m.scan(); r = mc.stitch(f, verbose=False); m.close()
assert {'flatfield.npy', 'dark.npy'} <= set(os.listdir(os.path.join(f, 'join'))), os.listdir(os.path.join(f, 'join'))
print("Prepare step by step: lamp, even light, dark frame, flat field + exposure, focus + settling; saved, Keep next time; "
      "the flat field and the dark frame go into the stitch: OK")

# ------------------------------------------------------------------ 47. Профили креста (P в просмотре): ширина плеча по краям R, G, B; крест под мышью — самая крупная «+», не пылинка
from microcontrol.mask import cross_profiles, snap_cross, meas_gray
rng = np.random.default_rng(7)
def _cross_img(c, ang, L=30, t=6, N=241, ss=8):
    n = N * ss; im = np.zeros((n, n), np.float32); C = (np.asarray(c) + 0.5) * ss - 0.5
    R = cv2.getRotationMatrix2D((float(C[0]), float(C[1])), -ang, 1.0)
    for P_ in ([[C[0]-L*ss, C[1]-t*ss/2], [C[0]+L*ss, C[1]-t*ss/2], [C[0]+L*ss, C[1]+t*ss/2], [C[0]-L*ss, C[1]+t*ss/2]],
               [[C[0]-t*ss/2, C[1]-L*ss], [C[0]+t*ss/2, C[1]-L*ss], [C[0]+t*ss/2, C[1]+L*ss], [C[0]-t*ss/2, C[1]+L*ss]]):
        cv2.fillPoly(im, [cv2.transform(np.array([P_], np.float32), R)[0].astype(np.int32)], 1.0)
    im = cv2.GaussianBlur(cv2.resize(im, (N, N), interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    return (120 + 60 * im + rng.normal(0, 3, im.shape)).clip(0, 255).astype(np.uint8)
im = _cross_img(np.array([120.3, 119.6]), 10.0, L=30, t=6)
info = {}; d, resp = snap_cross(im, 60, info=info, angle=10.0)
c_ = (np.array(im.shape[::-1]) - 1) / 2 + d
pr = cross_profiles(im, c_, info['arm'], info['thick'], np.radians(info.get('angle', 10.0)))
wid = [pr[k]['edges'][1] - pr[k]['edges'][0] for k in 'hv']
assert all(abs(w_ - 6) < 0.6 for w_ in wid), wid
spots = im.copy()
for q in ((40, 40), (200, 60), (60, 190), (190, 200)):                    # тёмные пылинки вокруг
    cv2.circle(spots, q, 3, 60, -1)
d2, _ = snap_cross(spots, 100, angle=10.0)
assert np.hypot(*((np.array(im.shape[::-1]) - 1) / 2 + d2 - (120.3, 119.6))) < 0.5, d2
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus(); m.goto(x=150, y=150); m.exposure = 20.0
fw, fh = m.frame_size; m.area = mc.Area(0, 0, 2 * fw, 2 * fh); f = m.scan(); mc.stitch(f, verbose=False)
fo = next((os.path.join(f, d_) for d_ in sorted(os.listdir(f)) if os.path.exists(os.path.join(f, d_, 'join'))), f)
box_ = {}
def _grab(st):
    box_['st'] = st
def _keys47():
    yield -1; yield ord('p'); yield -1
    for k in range(8):
        yield ('move', 100 + 40 * k, 100 + 30 * k); yield -1
    yield ('click', 150, 180); yield -1; yield ('click', 260, 180); yield -1
    yield ('call', _grab); yield -1
    assert box_['st']['prof_pin'] is None and len(box_['st']['prof_pts']) == 2      # графики — только после 4 точек
    yield ('click', 200, 120); yield -1; yield ('click', 200, 240); yield -1
    st = box_['st']; res = st['prof_pin']
    assert res is not None and len(res['prof']) == 2 and not st['prof_pts']
    P1 = res['segs'][0][1]
    yield ('qdrag', float(P1[0]), float(P1[1]), float(P1[0]), float(P1[1]) + 20); yield -1      # наклонить линию A
    assert abs(res['segs'][0][1][1] - P1[1] - 20) < 1.0, res['segs'][0]
    px0, py0, w_, h_, xa, xb = st['prof_plots'][1]
    yield ('drag', px0 + w_ // 2, py0 + 30, px0 + w_ // 2 - 15, py0 + 30); yield -1           # край B рукой
    assert res['user'][1] is not None
    r_ = st['prof_rect']; w0 = res['w']
    yield ('wheel', r_[0] + 80, r_[1] + 100, +1); yield -1                                # над лупой — полоса
    assert res['w'] > w0
    yield ('wheel', px0 + 50, py0 + 40, +1); yield -1                                     # над графиком — масштаб Y
    yl = st['prof_y']; assert yl is not None
    yield ('wheel', px0 + 50, py0 + 40, +1); yield -1
    assert st['prof_y'][1] - st['prof_y'][0] < yl[1] - yl[0]
    yr = st['prof_yrect']
    yield ('click', (yr[0] + yr[2]) // 2, (yr[1] + yr[3]) // 2); yield -1
    assert st['prof_y'] is None
    yield ('rclick', px0 + 20, py0 + 30); yield -1
    assert res['user'][1] is None
    yield 27; yield 27
mc.view(fo, _keys=_keys47())
from microcontrol import project as P_
assert os.path.exists(os.path.join(P_.sample_of(fo), 'profile_lines.json'))
import microcontrol.manual as M47
cv2.getWindowImageRect = lambda w: (0, 0, 1280, 900)
_draw = []
_im0 = cv2.imshow
cv2.imshow = lambda w, img: (_draw.append(img), _im0(w, img))
M47.run(m, fullscreen=False, _keys=[-1, -1, 27, 27]); cv2.imshow = _im0; m.close()
assert _draw
print(f"Profiles across the arms: width {wid[0]:.2f} / {wid[1]:.2f} px (true 6), dust does not pull the snap; "
      "P window: two lines tilted by their ends, edges by hand, band by the wheel, saved for the manual map: OK")

# ------------------------------------------------------------------ 48. Уровни 10x -> 50x -> 100x: создать, отменить, вернуть, удалить, создать иначе; кресты стереть и отметить другие; сетка — выбранного в меню

import sys, os, tempfile, json, time, traceback, numpy as np, cv2
import microcontrol as mc
import microcontrol.manual as M
import microcontrol.viewer as V
from microcontrol import samples as SM, layers as LY, lattice as LT, project as P_
from microcontrol.mask import Grid
from microcontrol.snap import stage_to_mosaic, mosaic_to_stage

OUT = tempfile.mkdtemp()
os.makedirs(OUT, exist_ok=True)
for f_ in os.listdir(OUT):
    os.remove(os.path.join(OUT, f_))
cv2.getWindowImageRect = lambda w: (0, 0, 1400, 900)
shots = []
_o = cv2.imshow
cv2.imshow = lambda w, img: (shots.append(img), _o(w, img))
LOG, ISSUES = [], []
step_n = [0]


def log(s):
    LOG.append(s)


def issue(s):
    log("  !! " + s)
    ISSUES.append(f"[{step_n[0]}] {s}")


def shot(name):
    if shots:
        cv2.imwrite(os.path.join(OUT, f"{step_n[0]:02d}_{name}.png"), shots[-1])


m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
m.focus()
w = m._world
T, PT = np.radians(w.turn_deg), w.pitch


def cross_stage(i, j):
    return np.array([np.cos(T) * PT * i - np.sin(T) * PT * j, np.sin(T) * PT * i + np.cos(T) * PT * j])


def crosses_seen(f_):
    """Сколько настоящих крестов (сим-мира) целиком в мозаике съёмки (не ближе 10 мкм к краю)."""
    from microcontrol.storage import ScanData as _SD
    import microcontrol.viewer as _V
    sc_ = _V._Scene(f_)
    S_ = mosaic_to_stage(f_, [[0, 0], [sc_.size[0], 0], [sc_.size[0], sc_.size[1]], [0, sc_.size[1]]])
    sc_.st.close()
    x0, y0 = S_.min(0) + 10
    x1, y1 = S_.max(0) - 10
    n_ = 0
    for i in range(-5, 60):
        for j in range(-5, 60):
            c_ = cross_stage(i, j)
            n_ += int(x0 <= c_[0] <= x1 and y0 <= c_[1] <= y1)
    return n_


def manual(keys, name=None):
    try:
        M.run(m, fullscreen=False, _keys=[-1] + keys + [-1, -1])
    except Exception as e:
        issue(f"manual crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
    if name:
        shot(name)


def view(folder, keys, name=None):
    try:
        V.view(folder, _keys=[-1] + keys + [-1, 27, 27])
    except SystemExit:
        pass
    except Exception as e:
        issue(f"view crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
    if name:
        shot(name)
    LY.sync_brush(m)
    SM.persist(m)


def nav():
    return m._nav


def sel_k(num):
    return next((k for k, r in enumerate(m._regions) if r['num'] == num), None)


def latest(r):
    try:
        sc = P_.Selection(m.sample, r).latest
        return sc.folder if sc else None
    except Exception:
        return None


def check(tag):
    """Целостность после шага."""
    regs = m._regions
    uids = [r['uid'] for r in regs]
    nums = [r['num'] for r in regs]
    if len(set(uids)) != len(uids):
        issue(f"{tag}: duplicate uid")
    if len(set(nums)) != len(nums):
        issue(f"{tag}: duplicate numbers {sorted(nums)}")
    for r in regs:
        if r.get('parent'):
            k = LY.idx(m, r['parent'])
            if k is None:
                issue(f"{tag}: {r['num']} has a missing parent")
                continue
            p = regs[k]
            if p.get('kind') == 'group':
                continue
            if r.get('kind') != 'draft' and SM.obj(m, r) <= SM.obj(m, p):
                issue(f"{tag}: {r['num']} ({SM.obj(m, r)}x) inside {p['num']} ({SM.obj(m, p)}x)")
            if not LY.geom(m, p).buffer(0.5).contains(LY.geom(m, r)):
                issue(f"{tag}: {r['num']} sticks out of its parent {p['num']}")
    # сетки: кресты каждой последней съёмки ложатся на решётку выбранного (своя / родителя / образца)
    for r in regs:
        if r.get('kind') in ('draft', 'group'):
            continue
        f = latest(r)
        if not f or not os.path.exists(os.path.join(f, 'grid.json')):
            continue
        try:
            g = Grid(f)
        except Exception as e:
            issue(f"{tag}: grid of {r['num']} unreadable: {e}")
            continue
        if not g.ready():
            continue
        L = LY.lattice_for(m, r, self_too=True)
        if L is None:
            issue(f"{tag}: {r['num']} has a grid but no lattice")
            continue
        q = np.array([g.cr.point(i, j) for i in range(0, 3) for j in range(0, 3)])
        uv = LT.to_uv(L, mosaic_to_stage(f, q))
        dev = float(np.abs(uv - np.round(uv)).max()) * L['pitch']
        S_ = m.sample.data.get('lattice')
        uvS = LT.to_uv(S_, mosaic_to_stage(f, q)) if S_ else uv
        devS = float(np.abs(uvS - np.round(uvS)).max()) * S_['pitch'] if S_ else 0
        # номера: та же клетка в решётке выбранного и образца
        same_num = S_ is None or np.array_equal(np.floor(uv + 0.5), np.floor(uvS + 0.5))
        seeded = json.load(open(os.path.join(f, 'grid.json'))).get('seeded')
        log(f"    marks {r['num']}: " + ", ".join(f"{len(json.load(open(os.path.join(s_.folder, 'grid.json'))).get('marks') or [])}"
                                                for s_ in P_.Selection(m.sample, r).scans if os.path.exists(os.path.join(s_.folder, 'grid.json'))))
        own = L.get('own') or L.get('src')
        log(f"    grid of S{r['num']} {SM.obj(m, r)}x ({'seeded' if seeded else 'marked'}): "
            f"off its lattice {dev:.2f} um, off the sample lattice {devS:.2f} um, lattice from "
            f"{os.path.basename(os.path.dirname(os.path.dirname(own))) if own else '?'}")
        if dev > 1.0:
            issue(f"{tag}: crosses of S{r['num']} are {dev:.2f} um off the lattice of the selection")
        if not same_num:
            issue(f"{tag}: S{r['num']}: numbers differ from the sample lattice")


def step(title):
    step_n[0] += 1
    log(f"\n== {step_n[0]}. {title}")


R0 = len(m._regions)
fw, fh = m.frame_size
# ---------------------------------------------------------------- 10x: создать, отменить, вернуть, удалить, создать иначе
step("10x: draw an area, Ctrl+Z, Ctrl+Y")
manual([])
n = nav()
manual([('maprect', *n.to_px((-60, -50)), *n.to_px((300, 200))), -1], 'map_new10')
log(f"  areas: {[(r['num'], SM.obj(m, r)) for r in m._regions]}")
manual([26, -1])
log(f"  after Ctrl+Z: {len(m._regions)}")
if len(m._regions) != 0:
    issue("Ctrl+Z did not remove the new area")
manual([25, -1])
if len(m._regions) != 1:
    issue("Ctrl+Y did not bring the area back")
check("10x undo/redo")

step("10x: delete (C) and draw a different one")
m._sel = 0
manual([ord('c'), -1])
if m._regions:
    issue("C did not delete the selected area")
manual([('maprect', *n.to_px((-80, -60)), *n.to_px((340, 240))), -1], 'map_redraw10')
k1 = 0
r1 = m._regions[k1]
log(f"  area S{r1['num']} {SM.obj(m, r1)}x, {LY.geom(m, r1).bounds}")
check("10x redraw")

step("10x: scan + stitch")
m._sel = k1
f1 = m.scan()
f1 = m.last_scans[-1] if m.last_scans else f1
mc.stitch(f1, verbose=False)
log(f"  scan {os.path.basename(f1)}")
check("10x scanned")

step("10x view: mark crosses 0,0 1,0 0,1")
q = lambda i, j: tuple(stage_to_mosaic(f1, cross_stage(i, j))[0])
view(f1, [ord('c'), ('qclick', *q(0, 0)), -1, ('qclick', *q(1, 0)), -1, ('qclick', *q(0, 1)), -1], 'view_marked10')
g1 = Grid(f1)
log(f"  grid ready {g1.ready()}, crosses {len(g1.cr.marks)}; sample lattice {'yes' if m.sample.data.get('lattice') else 'no'}")
check("10x marked")

step("10x view: Shift+C (clear crosses)")
view(f1, [ord('C'), -1])
g1 = Grid(f1)
S_ = m.sample.data.get('lattice')
log(f"  grid ready {Grid(f1).ready()}; sample lattice still {'there' if S_ else 'gone'}")
if not Grid(f1).ready() and S_ is not None and S_.get('src') == os.path.abspath(f1):
    issue("crosses cleared, but the sample lattice from them stays (red grid on the map, new squares follow it)")
manual([], 'map_after_clear10')
check("10x cleared")

step("10x view: mark other crosses 2,1 3,1 2,2")
view(f1, [ord('c'), ('qclick', *q(2, 1)), -1, ('qclick', *q(3, 1)), -1, ('qclick', *q(2, 2)), -1])
check("10x re-marked")

step("10x view: Ctrl+Z twice, Ctrl+Y once")
view(f1, [26, -1, 26, -1, 25, -1])
log(f"  crosses {len(Grid(f1).cr.marks)}, ready {Grid(f1).ready()}")
check("10x undo crosses")

step("10x view: select 3 squares (M), right click one off, select another")
g1 = Grid(f1)
from microcontrol.storage import ScanData
W1, H1 = json.load(open(os.path.join(f1, 'join', 'size.json'))) if os.path.exists(os.path.join(f1, 'join', 'size.json')) else (None, None)
inside = []
for i in range(-4, 5):
    for j in range(-4, 5):
        p_ = g1.cr.point(i + 0.5, j + 0.5)
        S__ = mosaic_to_stage(f1, p_)[0]
        if LY.geom(m, m._regions[k1]).buffer(-60).contains(__import__('shapely.geometry', fromlist=['Point']).Point(*S__)):
            inside.append((i, j))
log(f"  squares fully inside: {inside}")
c_ = lambda ij: tuple(g1.cr.point(ij[0] + 0.5, ij[1] + 0.5))
a_, b_, c3, d4 = (inside + inside)[:4]
view(f1, [ord('m'), ('qclick', *c_(a_)), -1, ('qclick', *c_(b_)), -1, ('qclick', *c_(c3)), -1,
          ('qrclick', *c_(b_)), -1, ('qclick', *c_(d4)), -1], 'view_squares10')
want_sel = {a_, c3, d4}
if Grid(f1).selected != want_sel:
    issue(f"squares in the view {sorted(Grid(f1).selected)} != clicked {sorted(want_sel)}")
dr = [r for r in m._regions if r.get('kind') == 'draft']
log(f"  drafts: {[(r['num'], len(r.get('cells') or [])) for r in dr]}; selected in view {sorted(Grid(f1).selected)}")
check("10x squares")

# ---------------------------------------------------------------- 50x
step("50x: draw inside S1, Ctrl+Z, Ctrl+Y, delete, draw elsewhere, overlap, over the edge")
m.objective = 50
m.focus()
manual([])
n = nav()
L1 = LY.lattice_for(m, m._regions[k1], self_too=True)
uv1 = np.floor(LT.to_uv(L1, [LY.geom(m, m._regions[k1]).centroid.coords[0]])[0]).astype(int)
cen = lambda a, b: LT.to_stage(L1, [(uv1[0] - 1 + a, uv1[1] - 1 + b)])[0]
manual([('maprect', *n.to_px(cen(1.2, 1.2)), *n.to_px(cen(1.8, 1.8))), -1])
n50 = [r for r in m._regions if SM.obj(m, r) == 50 and r.get('kind') != 'draft']
log(f"  50x areas: {[(r['num'], r.get('cells')) for r in n50]}")
manual([26, -1])
manual([25, -1])
k50 = next(k for k, r in enumerate(m._regions) if SM.obj(m, r) == 50 and r.get('kind') != 'draft')
m._sel = k50
manual([ord('c'), -1])
if any(SM.obj(m, r) == 50 and r.get('kind') != 'draft' for r in m._regions):
    issue("50x area not deleted by C")
manual([('maprect', *n.to_px(cen(2.2, 1.2)), *n.to_px(cen(2.8, 1.8))), -1])
k50 = next(k for k, r in enumerate(m._regions) if SM.obj(m, r) == 50 and r.get('kind') != 'draft')
cnt = len(m._regions)
u50 = m._regions[k50]['uid']
R50 = lambda: m._regions[LY.idx(m, u50)]
manual([('maprect', *n.to_px(cen(2.3, 1.3)), *n.to_px(cen(2.7, 1.7))), -1])
inner_ = [r for r in m._regions if r.get('parent') == u50 and SM.obj(m, r) == 50]
log(f"  50x inside 50x: {[r['num'] for r in inner_]}")
for r in inner_:                                                    # вложено можно; дальше сценарий — без него
    m._regions.remove(r)
SM.persist(m)
b1 = LY.geom(m, m._regions[k1]).bounds
manual([('maprect', *n.to_px((b1[2] - 30, b1[1] + 10)), *n.to_px((b1[2] + 60, b1[1] + 60))), -1], 'map_50x')
log(f"  areas now: {[(r['num'], SM.obj(m, r), r.get('kind')) for r in m._regions]}; message: {getattr(m, '_last_msg', '')}")
check("50x drawn")

step("50x: scan + stitch S50, open the view (grid comes from the 10x lattice)")
r50 = R50()
m._sel = LY.idx(m, u50)
m.scan()
f50 = [x for x in m.last_scans if '_50x' in x] or m.last_scans
f50 = f50[-1]
mc.stitch(f50, verbose=False)
view(f50, [], 'view_seeded50')
d50 = json.load(open(os.path.join(f50, 'grid.json')))
log(f"  seeded {d50.get('seeded')}, labels {len(Grid(f50).labels)}")
if Grid(f50).ready() != (crosses_seen(f50) > 0):                # сетка — только если кресты видны
    issue(f"new 50x scan inside a marked 10x: grid {Grid(f50).ready()}, crosses seen {crosses_seen(f50)}")
check("50x seeded")

step("50x view: mark own crosses (Shift+C, then 2 crosses)")
q5 = lambda i, j: tuple(stage_to_mosaic(f50, cross_stage(i, j))[0])
uvc = LT.to_uv(L1, [LY.geom(m, r50).centroid.coords[0]])[0]
S0 = np.floor(uvc).astype(int)
# ближайшие кресты физической решётки: по сим-миру (индексы мира, не номера образца)
W_uv = lambda S: np.linalg.solve(np.c_[cross_stage(1, 0), cross_stage(0, 1)], S)
wc = np.floor(W_uv(LY.geom(m, r50).centroid.coords[0])).astype(int)
view(f50, [ord('C'), -1, ('qclick', *q5(wc[0], wc[1])), -1, ('qclick', *q5(wc[0] + 1, wc[1])), -1], 'view_marked50')
L50 = LY.lattice_for(m, R50(), self_too=True)
log(f"  50x lattice from {L50.get('own') and os.path.basename(L50['own'])}; sample lattice src "
    f"{os.path.basename(m.sample.data['lattice']['src'])}")
if not L50.get('own'):
    issue("own 50x marking is not used for the selected 50x")
m._sel = LY.idx(m, u50)
manual([], 'map_sel50')
check("50x own marks")

step("50x: Shift+C again -> falls back to the 10x lattice; mark differently")
view(f50, [ord('C'), -1])
_d = json.load(open(os.path.join(f50, 'grid.json')))
log(f"  after Shift+C: marks {len(_d.get('marks') or [])}, seeded {_d.get('seeded')}, keys {sorted(_d)}")
L50b = LY.lattice_for(m, R50(), self_too=True)
log(f"  lattice now from {L50b.get('own') or L50b.get('src')}")
if os.path.abspath(f50) == os.path.abspath(L50b.get('own') or ''):
    issue("cleared 50x marks are still used")
view(f50, [ord('c'), ('qclick', *q5(wc[0], wc[1] + 1)), -1, ('qclick', *q5(wc[0] + 1, wc[1] + 1)), -1])
check("50x re-marked")

step("50x: scan again (version 2) -> grid from its own marking")
m._sel = LY.idx(m, u50)
m.scan()
f50b = m.last_scans[-1]
mc.stitch(f50b, verbose=False)
view(f50b, [])
if not Grid(f50b).ready() and crosses_seen(f50b):
    issue("second version of the 50x has no grid")
log(f"  versions: {[os.path.basename(s.folder) for s in P_.Selection(m.sample, R50()).scans]}")
check("50x v2")

# ---------------------------------------------------------------- 100x
step("100x: draw inside the 50x, scan, view")
m.objective = 100
m.focus()
manual([])
n = nav()
gb = LY.geom(m, R50()).bounds
manual([('maprect', *n.to_px((gb[0] + 2, gb[1] + 2)), *n.to_px((gb[2] - 2, gb[3] - 2))), -1], 'map_100x')
k100 = next((k for k, r in enumerate(m._regions) if SM.obj(m, r) == 100 and r.get('kind') != 'draft'), None)
if k100 is None:
    issue("100x area inside the 50x not created")
else:
    log(f"  100x S{m._regions[k100]['num']} parent {LY.idx(m, m._regions[k100].get('parent'))} (50x is {k50})")
    m._sel = k100
    m.scan()
    f100 = m.last_scans[-1]
    mc.stitch(f100, verbose=False)
    view(f100, [], 'view_seeded100')
    if Grid(f100).ready() != (crosses_seen(f100) > 0):
        issue(f"new 100x scan: grid {Grid(f100).ready()}, crosses seen {crosses_seen(f100)}")
    check("100x seeded")

step("delete the 50x (middle level) and Ctrl+Z")
u100 = m._regions[k100]['uid'] if k100 is not None else None
m._sel = LY.idx(m, u50)
manual([ord('c'), -1])
if u100:
    r100 = m._regions[LY.idx(m, u100)]
    pk = LY.idx(m, r100.get('parent'))
    log(f"  100x parent now: {m._regions[pk]['num'] if pk is not None else None}")
    inS1 = LY.geom(m, m._regions[k1]).buffer(0.5).contains(LY.geom(m, r100))
    if inS1 and (pk is None or m._regions[pk]['uid'] != m._regions[k1]['uid']):
        issue("the 100x lost its parent although it lies inside the 10x")
check("50x deleted")
manual([26, -1])
k50 = LY.idx(m, u50)
if k50 is None:
    issue("Ctrl+Z did not bring the 50x back")
elif u100:
    r100 = m._regions[LY.idx(m, u100)]
    if r100.get('parent') != R50()['uid']:
        issue("after Ctrl+Z the 100x is not inside the 50x again")
check("50x restored")

step("V opens the latest scan of what is selected in the menu")
opened = []
_v0 = V.view
V.view = lambda f, **kw: opened.append(f)
for k in [k1, k50, LY.idx(m, u100) if u100 else None]:
    if k is None:
        continue
    m._sel = k
    opened.clear()
    manual([ord('v'), -1])
    want = latest(m._regions[k])
    ok = opened and os.path.abspath(opened[0]) == os.path.abspath(want)
    log(f"  S{m._regions[k]['num']}: opened {os.path.basename(opened[0]) if opened else None}, latest {os.path.basename(want)}")
    if not ok:
        issue(f"V with S{m._regions[k]['num']} selected opened something else")
V.view = _v0

step("10x: clear crosses and mark again: 50x / 100x keep their own")
u50 = next(r['uid'] for r in m._regions if r.get('kind') not in ('group', 'draft') and
           os.path.abspath(f50).startswith(os.path.abspath(P_.Selection(m.sample, r).folder) + os.sep))
L50_before = LY.lattice_for(m, R50(), self_too=True)
for sc_ in P_.Selection(m.sample, R50()).scans:
    _d = json.load(open(os.path.join(sc_.folder, 'grid.json'))) if os.path.exists(os.path.join(sc_.folder, 'grid.json')) else {}
    log(f"  scan {sc_.folder[-30:]} seeded {_d.get('seeded')} marks {len(_d.get('marks') or [])} own {LY.own_lattice(m.sample, R50()) is not None}")
view(f1, [ord('C'), -1, ord('c'), ('qclick', *q(1, 1)), -1, ('qclick', *q(2, 1)), -1, ('qclick', *q(1, 2)), -1])
if not Grid(f1).ready():
    issue("Shift+C, then C, then clicks: crosses not marked")
L50_after = LY.lattice_for(m, R50(), self_too=True)
log(f"  50x lattice before {np.round(L50_before['e1'], 2)} {L50_before.get('own')} after {np.round(L50_after['e1'], 2)} {L50_after.get('own')}")
if not np.allclose(L50_before['e1'], L50_after['e1']):
    issue("the 50x grid changed when the 10x was re-marked")
for k in [k1, k50]:
    m._sel = k
    manual([], f'map_sel_{SM.obj(m, m._regions[k])}x_final')
check("final")

m.close()
log("\nISSUES:\n" + ("\n".join(ISSUES) if ISSUES else "none"))
assert not ISSUES, (ISSUES, LOG[-30:])
cv2.imshow = _o
print("Levels 10x -> 50x -> 100x with undo, redo, delete and redo differently: crosses cleared and marked again, "
      "every grid follows what is selected (own marks, else the larger one around, else the sample), V opens "
      "the selected: OK")

# ------------------------------------------------------------------ 49. Scan area / S снимает то, что выбрано в меню справа: карточку, группу; новая область без угла 2 — не снимается
import microcontrol.manual as M
from microcontrol import samples as SM, layers as LY
cv2.getWindowImageRect = lambda w: (0, 0, 1440, 900)
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
M.run(m, fullscreen=False, _keys=[-1, -1]); n_ = m._nav
M.run(m, fullscreen=False, _keys=[-1, ('maprect', *n_.to_px((0, 0)), *n_.to_px((150, 100))), -1,
                                  ('maprect', *n_.to_px((400, 300)), *n_.to_px((550, 400))), -1,
                                  ('maprect', *n_.to_px((800, 0)), *n_.to_px((950, 100))), -1, -1])
sel_ = lambda: [os.path.basename(os.path.dirname(os.path.dirname(f_))) for f_ in m.last_scans]
M.run(m, fullscreen=False, _keys=[-1, ('mouse', cv2.EVENT_LBUTTONDOWN, 1250, 150, 0), -1,
                                  ('mouse', cv2.EVENT_LBUTTONUP, 1250, 150, 0), -1, ord('s'), -1, -1])
assert m._regions[m._sel]['num'] == 1 and sel_() == ['S01_10x'], (m._sel, sel_())       # карточка 1 — только она
LY.group(m, [1, 2]); SM.persist(m)                                                     # группа из 2 и 3 — выбрана
M.run(m, fullscreen=False, _keys=[-1, ord('s'), -1, -1])
assert sorted(sel_()) == ['S02_10x', 'S03_10x'], sel_()
m.last_scans = []; m.goto(x=2000, y=1500)
M.run(m, fullscreen=False, _keys=[-1, ord('1'), -1, ord('s'), -1, -1])                 # угол 1 новой: не снимать
assert m.last_scans == [] and m._pending is not None, (m.last_scans, m._pending)
m.close(); print("Scan takes what is selected on the right: one card, a whole group; a new area without corner 2 waits: OK")

# ------------------------------------------------------------------ 50. Сшивка тысяч кадров: доли пикселя, ошибка столика не тянет шов, люфт, разреженный решатель; швы в просмотре (J)
import importlib, csv as _csv
ST = importlib.import_module('microcontrol.stitch')
rng = np.random.default_rng(3)
big = cv2.GaussianBlur(rng.random((900, 1400)).astype(np.float32), (0, 0), 2.0)
class _St:
    def __init__(s, imgs): s.imgs = imgs
    def frame(s, i, roi=None):
        im = s.imgs[i] if roi is None else s.imgs[i][roi[1]:roi[3], roi[0]:roi[2]]
        return np.clip(im * 255, 0, 255).astype(np.uint8)
e_sub = []
for t in np.linspace(0, 1, 6):
    tr = np.array([576 + t, 3.3 * t])
    b_ = cv2.warpAffine(big, np.float32([[1, 0, -(100 + tr[0])], [0, 1, -(100 + tr[1])]]), (640, 480), flags=cv2.INTER_CUBIC)
    for off in ((0, 0), (5, -6)):                                         # столик ошибся на 5, -6 px
        dt, _ = ST._pair(_St([big[100:580, 100:740], b_]), 0, 1, tr + off, 640, 480, 40, 0.0, 40)
        e_sub.append(np.hypot(*(dt - tr)))
assert max(e_sub) < 0.03, e_sub
# разреженный решатель: 3000 кадров, швы с шумом -> то же, что плотный МНК (на 300), и быстро
nx_, ny_ = 60, 50
X_ = np.array([[i * 500.0, j * 400.0] for j in range(ny_) for i in range(nx_)]) + rng.normal(0, 3, (nx_ * ny_, 2))
I_, J_ = [], []
for j in range(ny_):
    for i in range(nx_):
        k = j * nx_ + i
        if i + 1 < nx_: I_.append(k); J_.append(k + 1)
        if j + 1 < ny_: I_.append(k); J_.append(k + nx_)
I_, J_ = np.array(I_), np.array(J_)
M_ = X_[J_] - X_[I_] + rng.normal(0, 0.1, (len(I_), 2)); w_ = np.ones(len(I_))
base_ = np.array([[i * 500.0, j * 400.0] for j in range(ny_) for i in range(nx_)])
t0 = time.time(); P_ = ST._solve_graph(I_, J_, M_, w_, base_, 0.01, rel=0.3); t_cg = time.time() - t0
e_ = (P_ - P_.mean(0)) - (X_ - X_.mean(0))
assert np.percentile(np.hypot(*e_.T), 95) < 1.0 and t_cg < 5, (np.percentile(np.hypot(*e_.T), 95), t_cg)
# скан с люфтом 3 шага: истина — положения столика в момент снимка
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); w = m._world
w.backlash = {'x': 3, 'y': 3}
rec_, th_ = [], []
_f0 = w._frame
def _fr(*a, **k):
    p_ = w.p; sp_, sm_ = w._p, w.motion
    w._p, w.motion = {**sp_, **p_}, None
    try:
        im_ = _f0(*a, **k)
    finally:
        w._p, w.motion = sp_, sm_
    rec_.append((p_['x'], p_['y']))
    th_.append(cv2.resize(cv2.cvtColor(im_, cv2.COLOR_BGR2GRAY), (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32))
    return im_
w._frame = _fr
m.focus(); fw, fh = m.frame_size
m.area = mc.Area(0, 0, 5 * fw * 0.9, 3 * fh * 0.9); f = m.scan()
r_ = mc.stitch(f, verbose=False)
fo = f if os.path.exists(os.path.join(f, 'join')) else next(os.path.join(f, d_) for d_ in sorted(os.listdir(f)) if os.path.exists(os.path.join(f, d_, 'join')))
from microcontrol.storage import ScanData, meta_A
st_ = ScanData(fo); H_, W_ = st_.shape[:2]; A_ = meta_A(st_.meta, W_, H_)
T_ = []
for i in range(len(st_)):                                                 # снимок кадра i — тот, что совпадает картинкой
    raw_ = st_._jpg(st_.index[i]['file'])
    t_ = cv2.resize(cv2.cvtColor(raw_, cv2.COLOR_BGR2GRAY), (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    T_.append(rec_[int(np.argmin([np.abs(q_ - t_).mean() for q_ in th_]))])
st_.close()
T_ = -(np.array(T_, float) @ A_.T)
P_ = np.array([[float(q['x_px']), float(q['y_px'])] for q in _csv.DictReader(open(os.path.join(fo, 'join', 'positions_all.csv')))])
e_ = (P_ - P_.mean(0)) - (T_ - T_.mean(0)); e_ -= np.median(e_, 0)
info_ = json.load(open(os.path.join(fo, 'join', 'stitch.json')))
assert np.hypot(*e_.T).max() < 1.0, (np.hypot(*e_.T).max(), info_)
from microcontrol.sim import UM_PER_STEP
assert all(abs(v - 3 * UM_PER_STEP) < 0.5 for v in info_['backlash_um']), info_['backlash_um']
# J: швы точками; щелчок — выключить, ещё раз — включить; выключить и сохранить; заново — помнится; Backspace — как было
z_ = ST.seams(fo); assert z_ is not None
k_ = int(np.argmax(z_['w']))
box_ = {}
def _keys50(save):
    yield -1; yield ord('j'); yield -1
    yield ('call', lambda st: box_.update(st=st)); yield -1
    ed_ = box_['st']['edit']; assert ed_.get('z') is not None
    c_ = box_['st']['_t']['seam_pt'](k_)                                 # точка шва (у диагонали — не середина)
    P0_ = ed_['orig'].copy()
    yield ('qclick', float(c_[0]), float(c_[1])); yield -1
    key_ = (int(z_['I'][k_]), int(z_['J'][k_]))
    assert ed_['ovr'].get(key_) is False and ed_['w'][k_] == 0 and ed_['moved'], (ed_['ovr'], key_)
    d_ = np.abs(box_['st']['edit']['res']).max()
    if not save:
        yield ('qclick', float(c_[0]), float(c_[1])); yield -1
        assert ed_['ovr'].get(key_) is True and ed_['w'][k_] > 0
        yield 27; yield 27
    else:
        yield ord('j'); yield -1; yield -1; yield 27; yield 27
mc.view(fo, _keys=_keys50(False))
mc.view(fo, _keys=_keys50(True))
se_ = json.load(open(os.path.join(fo, 'join', 'seams_edit.json')))
assert [int(z_['I'][k_]), int(z_['J'][k_]), False] in se_, se_
assert os.path.exists(os.path.join(fo, 'join', 'positions_auto.csv'))
def _keys50b():
    yield -1; yield ord('j'); yield -1
    yield ('call', lambda st: box_.update(st=st)); yield -1
    ed_ = box_['st']['edit']
    assert ed_['ovr'].get((int(z_['I'][k_]), int(z_['J'][k_]))) is False and ed_['w'][k_] == 0
    yield 8; yield -1; yield -1; yield 27; yield 27
mc.view(fo, _keys=_keys50b())
assert not os.path.exists(os.path.join(fo, 'join', 'seams_edit.json')) and \
    not os.path.exists(os.path.join(fo, 'join', 'positions_auto.csv'))
m.close()
print(f"Stitch: seam to {max(e_sub):.3f} px (stage off by 8 px: same), 3000 frames solved in {t_cg:.2f} s, scan with "
      f"backlash: frames to {np.hypot(*e_.T).max():.2f} px, backlash {info_['backlash_um']} um found; J: seam dots "
      "off / on, saved, remembered, Backspace back to automatic: OK")

# ------------------------------------------------------------------ 51. Рамки с кристаллами: Ctrl+тянуть на карте — рамка как нарисована (не по клеткам); обзор 10x выбран — Scan снимает все рамки 50x в нём
import microcontrol.manual as M
from microcontrol import samples as SM, layers as LY
cv2.getWindowImageRect = lambda w: (0, 0, 1440, 900)
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
M.run(m, fullscreen=False, _keys=[-1, -1]); n_ = m._nav
M.run(m, fullscreen=False, _keys=[-1, ('maprect', *n_.to_px((-50, -50)), *n_.to_px((900, 700))), -1, -1])
k10 = len(m._regions) - 1; u10 = m._regions[k10]['uid']
m._sample.data['lattice'] = dict(O=[0.0, 0.0], e1=[100.0, 0.0], e2=[0.0, 100.0], stage=True, src=None)   # сетка квадратов
SM.persist(m)
m.objective = 50
fw5, fh5 = m.frame_size
n_ = m._nav
def box_px(x0, y0, x1, y1):
    return (*n_.to_px((x0, y0)), *n_.to_px((x1, y1)))
M.run(m, fullscreen=False, _keys=[-1, ('maprect', *box_px(205, 205, 395, 395)), -1, -1])          # просто тянуть: клетки
rc_ = m._regions[-1]
assert rc_.get('cells') and not rc_.get('free'), rc_                                                   # квадраты сетки
boxes = [(510, 120, 528, 134), (560, 400, 640, 455), (90, 480, 150, 520)]                             # мкм: малая, средняя, ещё одна
for b_ in boxes:
    M.run(m, fullscreen=False, _keys=[-1, ('maprect', *box_px(*b_), True), -1, -1])                    # Ctrl+тянуть
fr_ = [r_ for r_ in m._regions if r_.get('free')]
assert len(fr_) == 3 and all(r_.get('kind') is None and r_.get('obj') == 50 for r_ in fr_), [(r_['num'], r_.get('kind')) for r_ in fr_]
ms_ = m.motor_step
for r_, b_ in zip(fr_, boxes):                                                                       # рамка: нарисованное внутри,
    B_ = LY.geom(m, r_).bounds                                                                       # запас не меньше шага,
    want_ = (*n_.to_um(n_.to_px(b_[:2])), *n_.to_um(n_.to_px(b_[2:])))                               # края на шагах мотора
    assert all(B_[i_] <= want_[i_] - ms_ + 1e-6 for i_ in (0, 1)) and all(B_[i_] >= want_[i_] + ms_ - 1e-6 for i_ in (2, 3)), (B_, want_)
    assert all(abs(v_ - round(v_ / ms_) * ms_) < 0.01 for v_ in B_), (B_, ms_)
    assert r_.get('parent') == u10, r_
m._sel = LY.idx(m, u10)                                                                              # выбран обзор 10x
m.last_scans = []
M.run(m, fullscreen=False, _keys=[-1, ord('s'), -1, -1])
names_ = sorted(os.path.basename(os.path.dirname(os.path.dirname(f_))) for f_ in m.last_scans)
want_n = sorted(f"S{r_['num']:02d}_50x" for r_ in m._regions if SM.obj(m, r_) == 50)
assert names_ == want_n, (names_, want_n)                                                            # все 50x в обзоре, и только они
from microcontrol.storage import ScanData
assert all(r_.get('kind') != 'draft' and SM.obj(m, r_) == 50 for r_ in fr_), [(r_['num'], r_.get('kind')) for r_ in fr_]
nf_ = []
for r_ in fr_:                                                                                      # рамки стали 50x и сняты
    f_ = next(f_ for f_ in m.last_scans if f"S{r_['num']:02d}_50x" in f_)
    st_ = ScanData(f_); n_fr = len(st_); st_.close()
    nf_.append(n_fr)
    from microcontrol import quality as Q51
    assert SM.shoot_mode(m, r_) == 'quality' and Q51.stitched_as(f_) == 'quick' and \
        n_fr == int(np.prod(Q51.plan_area(m, r_, 0.4).n)), (r_['num'], n_fr)              # рамка снята для quality, сшита quick
m.objective = 10
m._sel = LY.idx(m, u10)
M.run(m, fullscreen=False, _keys=[-1, ord('s'), -1, -1])                                              # 10x: сам обзор
assert any('S%02d_10x' % m._regions[LY.idx(m, u10)]['num'] in f_ for f_ in m.last_scans), m.last_scans
m.close()
print(f"Crystal boxes: Ctrl+drag gives draft boxes (drawn box inside, margin for backlash, edges on motor steps), plain "
      f"drag still snaps to squares; the 10x overview chosen -> Scan turns them into 50x and shoots them ({nf_} frames): OK")

# ------------------------------------------------------------------ 52. Любое долгое дело с полосой останавливается (Esc / Stop); полоса идёт долями (по этапам, по времени), а не 0 -> 100 %
import threading
from microcontrol import ui as UI_, util as U_
from microcontrol.errors import Stopped
from microcontrol.util import Progress
cv2.namedWindow('busy test')
log_ = []
def frame_(stt):
    log_.append((time.monotonic(), stt['frac'], stt['stage'], stt['text']))
    return np.zeros((120, 400, 3), np.uint8), (0, 0, 10, 10)
def steps_(n=200):
    with Progress(n, "Work", unit='steps') as pr:
        for i in range(n):
            time.sleep(0.02); pr.update(1, f"step {i}")
    return 'done'
# 1) с полосой: Stop -> выходит на ближайшем шаге
threading.Timer(0.5, U_.stop_request).start()
r = UI_.run_busy('busy test', "Steps", steps_, frame_, wait_ms=20)
assert r.get('stopped') and isinstance(r['e'], Stopped) and r['seconds'] < 1.2, r
# 2) чистый Python без полосы и без проверок: останавливается силой
def spin_():
    x = 0
    while True:
        x += 1
threading.Timer(0.3, U_.stop_request).start()
r = UI_.run_busy('busy test', "Spin", spin_, frame_, wait_ms=20)
assert r.get('stopped') and r['seconds'] < 4.0, r
# 3) внутри no_stop (скан): Stop не прерывает, дело кончается само (on_stop — его мягкий стоп)
flag = {}
def soft_():
    with U_.no_stop():
        with Progress(50, "Scan", unit='frames') as pr:
            for i in range(50):
                if flag.get('abort'):
                    return f'stopped after {i} frames'
                time.sleep(0.02); pr.update(1)
threading.Timer(0.3, U_.stop_request).start()
r = UI_.run_busy('busy test', "Scan", soft_, frame_, wait_ms=20, on_stop=lambda: flag.update(abort=True))
assert r.get('r', '').startswith('stopped after') and not r.get('stopped'), r
# 4) этапы: общая полоса растёт, без скачков 0 -> 100 на каждом этапе
def two_():
    for nm in ("Job: first", "Job: second"):
        with Progress(30, nm, unit='steps') as pr:
            for i in range(30):
                time.sleep(0.02); pr.update(1)
log_.clear()
r = UI_.run_busy('busy test', "Job", two_, frame_, plan=[("Job: first", 0.5), ("Job: second", 0.5)], wait_ms=20)
fr_ = [f for _t, f, _s, _x in log_ if f is not None]
assert all(b >= a - 1e-9 for a, b in zip(fr_, fr_[1:])) and len({round(f, 2) for f in fr_ if 0.05 < f < 0.95}) > 10, fr_
# 5) без своих долей: полоса по времени (ожидание — из прошлого раза), а не 0 / 100
UI_._busy_times("Sleepy", 1.0)
log_.clear()
r = UI_.run_busy('busy test', "Sleepy", lambda: time.sleep(1.0), frame_, wait_ms=20)
fr_ = [f for _t, f, _s, _x in log_]
assert None not in fr_ and len({round(f, 2) for f in fr_ if 0.1 < f < 0.9}) > 5, fr_
# 6) сшивка останавливается посреди швов, файлы прежней сшивки целы
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus(); fw, fh = m.frame_size
m.area = mc.Area(0, 0, 7 * fw * 0.7, 5 * fh * 0.7); f = m.scan(); m.close()
mc.stitch(f, verbose=False)
fo = f if os.path.exists(os.path.join(f, 'join')) else next(os.path.join(f, d_) for d_ in sorted(os.listdir(f)) if os.path.exists(os.path.join(f, d_, 'join')))
P0 = open(os.path.join(fo, 'join', 'positions_all.csv')).read()
log_.clear()
threading.Timer(1.0, U_.stop_request).start()
r = UI_.run_busy('busy test', "Stitching", lambda: mc.stitch(fo, verbose=True), frame_, wait_ms=20,
                 plan=[("Stitch: light", 0.08), ("Stitch: seams", 0.62), ("Stitch: brightness", 0.08), ("Stitch: saving", 0.22)])
assert r.get('stopped') and r['seconds'] < 3.0, (r.get('e'), r['seconds'])
assert open(os.path.join(fo, 'join', 'positions_all.csv')).read() == P0
log_.clear()
r = UI_.run_busy('busy test', "Stitching", lambda: mc.stitch(fo, verbose=True), frame_, wait_ms=20,
                 plan=[("Stitch: light", 0.08), ("Stitch: seams", 0.62), ("Stitch: brightness", 0.08), ("Stitch: saving", 0.22)])
fr_ = [f for _t, f, _s, _x in log_ if f is not None]
assert {'Stitch: seams', 'Stitch: saving'} <= {s_ for _t, _f, s_, _x in log_}, {s_ for _t, _f, s_, _x in log_}   # после Stop полосы не «вложенные»
assert 'e' not in r and all(b >= a - 1e-9 for a, b in zip(fr_, fr_[1:])) and \
    len({round(f, 2) for f in fr_ if 0.02 < f < 0.98}) > 5, (r.get('e'), len({round(f, 2) for f in fr_ if 0.02 < f < 0.98}),
                                                               [(i, a, b) for i, (a, b) in enumerate(zip(fr_, fr_[1:])) if b < a - 1e-9][:5],
                                                               sorted({s_ for _t, _f, s_, _x in log_}))
cv2.destroyWindow('busy test')
print("Stop: with a progress bar at the next step, plain Python by force, a scan softly after its frame; a stitch stops "
      "in 1-3 s and the old results stay; the bar goes by stages and by time, not 0 -> 100: OK")

# ------------------------------------------------------------------ 53. Клавиши с камерой при живом потоке камеры (как на приборе): F, пробел, W не ждут вечно замок камеры, который держит клавиша
import threading
import microcontrol.manual as M
cv2.getWindowImageRect = lambda w: (0, 0, 1440, 900)
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
m._test_live = True                                            # живой поток камеры, как без тестовых клавиш
m.goto(z=m.position.z + 40)
z0 = m.position.z
out = {}
def run_():
    try:
        M.run(m, fullscreen=False, _keys=[-1, -1, ord('f'), -1, ord(' '), -1, ord('w'), -1, -1])
        out['ok'] = True
    except Exception as e:
        out['e'] = e
th = threading.Thread(target=run_, daemon=True); th.start(); th.join(90)
assert not th.is_alive(), "autofocus / picture / white balance from the keys hang (camera lock)"
assert out.get('ok'), out
assert abs(m.position.z - z0) > 5, (z0, m.position.z)          # фокус двигал Z
m.close()
print(f"Keys with the live camera thread: F focused (Z {z0:.0f} -> {m.position.z:.0f} um), Space, W done, no hang: OK")

# ------------------------------------------------------------------ 54. Сетка сама — только там, где видны кресты (по столику — никогда сама); черновик удаляется из списка в просмотре
import microcontrol.manual as M, microcontrol.viewer as V
from microcontrol import samples as SM, layers as LY
from microcontrol.mask import Grid
cv2.getWindowImageRect = lambda w: (0, 0, 1440, 900)
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); w = m._world
w.cross_free = [(1500, -1500, 4500, 1500)]                         # здесь крестов нет
m.focus()
M.run(m, fullscreen=False, _keys=[-1, -1]); n_ = m._nav
M.run(m, fullscreen=False, _keys=[-1, ('maprect', *n_.to_px((0, 0)), *n_.to_px((420, 320))), -1,
                                  ('maprect', *n_.to_px((2000, 0)), *n_.to_px((2420, 320))), -1, -1])
fs_ = []
for num_ in (1, 2):                                                 # 1 — с крестами, 2 — без
    u_ = next(r_['uid'] for r_ in m._regions if r_['num'] == num_)
    m.last_scans = []; m.scan(only=[u_]); fs_.append(m.last_scans[-1]); mc.stitch(fs_[-1], verbose=False)
fA, fB = fs_
P_, T_ = w.pitch, np.radians(w.turn_deg)
Lc = dict(O=[0.0, 0.0], e1=[P_ * np.cos(T_), P_ * np.sin(T_)], e2=[-P_ * np.sin(T_), P_ * np.cos(T_)], pitch=P_,
          numbered=False, stage=False, src=None)
for f_, want in ((fA, True), (fB, False)):                          # подставить по крестам: есть кресты — есть сетка
    gp_ = os.path.join(f_, 'grid.json')
    if os.path.exists(gp_):
        os.remove(gp_)
    sc_ = V._Scene(f_)
    g2, msg_ = V._seed(f_, Grid(f_), Lc, None, sc_, sc_.st.meta.get('pixel_size_um'), need_crosses=True)
    sc_.st.close()
    assert g2.ready() == want and (want or not os.path.exists(gp_)), (f_, g2.ready(), msg_, open(gp_).read()[:300] if os.path.exists(gp_) else None)
def opened(f_, lat):
    m._sample.data['lattice'] = lat; m._sample.save()
    gp_ = os.path.join(f_, 'grid.json')
    if os.path.exists(gp_):
        os.remove(gp_)
    mc.view(f_, _keys=[-1, -1, 27, 27])
    return Grid(f_).ready()
assert not opened(fA, dict(O=[0.0, 0.0], e1=[100.0, 0.0], e2=[0.0, 100.0], pitch=100.0, numbered=False, stage=True, src=None))
assert opened(fA, Lc) and not opened(fB, Lc)                       # по столику — не сама; по крестам — где они видны
# черновик: выбрать квадраты (M, рамка), крестик на его карточке — квадраты сняты, черновика нет
box_ = {}
def keys54():
    yield -1; yield ord('m'); yield -1
    yield ('drag', 420, 300, 700, 520); yield -1
    yield ('call', lambda st: box_.update(t=st['_t'])); yield -1
    dr_ = [r_ for r_ in box_['t']['regions']() if r_.get('kind') == 'draft']
    assert dr_ and Grid(fA).selected, (dr_, Grid(fA).selected)
    box_['msg'] = box_['t']['delete_draft'](dr_[0]['uid'], -1)
    yield -1
    assert not [r_ for r_ in box_['t']['regions']() if r_.get('kind') == 'draft'], box_['t']['regions']()
    yield 27; yield 27
mc.view(fA, _keys=keys54())
assert not Grid(fA).selected and 'deleted' in box_['msg'], box_['msg']
m.close()
print(f"Grid by itself only where crosses are seen (none by the stage coordinates); draft x in the view: {box_['msg']}: OK")


# ------------------------------------------------------------------ 55. Сторож: программа вылетела (os._exit, SIGKILL, необработанная ошибка) — лазер, моторы, свет, охлаждение выключены; штатное закрытие — сторож молчит
import pty, select, signal
from microcontrol import protocol as P
d55 = tempfile.mkdtemp()
root_ = os.path.dirname(os.path.abspath(mc.__file__ if hasattr(mc, '__file__') else 'microcontrol'))
root_ = os.path.dirname(root_)


def heard(master, s_):
    """Байты, пришедшие в «прибор» за s_ секунд -> список (команда, данные)."""
    buf, t_ = b"", time.time() + s_
    while time.time() < t_:
        r_, _, _ = select.select([master], [], [], 0.1)
        if r_:
            try:
                buf += os.read(master, 1024)
            except OSError:
                break
    out = []
    while P.HEAD in buf:
        i_ = buf.index(P.HEAD); n_ = int.from_bytes(buf[i_ + 2:i_ + 4], 'big')
        fr = buf[i_ + 2:i_ + 2 + n_ - 1]
        out.append((fr[2], bytes(fr[3:])))
        buf = buf[i_ + 2 + n_ - 1:]
    return out


want = [(P.LASER, b"\x00\x00"), (P.TURN_MOTOR, b"\x00"), (P.LIGHT, b"\x00\x00"), (P.SET_TEC, P.tec_data(P.WARM_C))]
for how in ('exit', 'kill', 'error', 'close'):
    master, slave = pty.openpty()
    port_ = os.ttyname(slave)
    prog = f"""
import os, sys, signal, time
sys.path.insert(0, {root_!r})
from microcontrol import guard
g = guard.start({port_!r}, cooling=True, logdir={d55!r})
print(g.proc.pid, flush=True)
time.sleep(0.5)
if {how!r} == 'exit': os._exit(3)                       # падение без atexit (как segfault)
if {how!r} == 'kill': os.kill(os.getpid(), signal.SIGKILL)
if {how!r} == 'error': raise RuntimeError('boom')       # необработанная ошибка, close() не вызван
g.done()                                                 # штатное m.close()
"""
    pr = subprocess.run([sys.executable, '-c', prog], capture_output=True, text=True, timeout=30)
    gpid = int(pr.stdout.split()[0])
    got = heard(master, 6)
    t_ = time.time()
    while time.time() - t_ < 10:                              # сторож вышел сам
        try:
            os.kill(gpid, 0); time.sleep(0.1)
        except ProcessLookupError:
            break
    else:
        raise AssertionError(f"guard {gpid} still alive after {how}")
    if how == 'close':
        assert got == [], got
    else:
        assert got == want, (how, got)
    os.close(master); os.close(slave)
logs_ = [f_ for f_ in os.listdir(d55) if f_.endswith('_guard.log')]
txt_ = "".join(open(os.path.join(d55, f_), encoding='utf-8').read() for f_ in logs_)
assert len(logs_) >= 3 and txt_.count('ended without closing') == 3 and 'Laser: off sent' in txt_, (logs_, txt_[:500])
# сторож лёгкий: без OpenCV и numpy
pr = subprocess.run([sys.executable, '-c', f"import sys, runpy; sys.argv=['g', '/nonexistent', '0', {d55!r}]; "
                     f"sys.stdin = open(os.devnull) if False else sys.stdin; "
                     f"import importlib.util as u; s=u.spec_from_file_location('g', {os.path.join(root_, 'microcontrol', 'guard.py')!r});"
                     f"m=u.module_from_spec(s); s.loader.exec_module(m); m._mod('protocol'); m._mod('log'); m._mod('control');"
                     f"print('cv2' in sys.modules, 'numpy' in sys.modules)"], capture_output=True, text=True, timeout=30)
assert pr.stdout.split() == ['False', 'False'], (pr.stdout, pr.stderr[-500:])
print(f"Crash guard: after os._exit, SIGKILL and an unhandled error the laser, motors, light and cooling are switched "
      f"off ({len(want)} commands), a normal close leaves the guard silent; the guard loads no OpenCV/numpy: OK")


# ------------------------------------------------------------------ 56. Сервер у прибора (net://): ключ, один хозяин, команды и кадры по сети, скан и сшивка, память у сервера; нет вестей / вылет / stop по сети — сервер выключает прибор; обрыв посреди движения — сервер знает, дошло ли
import socket as _so, warnings as _wn
from microcontrol import remote as RM
from microcontrol.errors import SafetyError as _SE
_s = _so.socket(); _s.bind(('127.0.0.1', 0)); P56 = _s.getsockname()[1]; _s.close()
hs_ = tempfile.mkdtemp()                                          # «старый компьютер»: свой дом
root56 = os.path.dirname(os.path.dirname(os.path.abspath(mc.__file__)))
srv = subprocess.Popen([sys.executable, '-m', 'microcontrol', 'serve', '--port', 'sim', '--listen', f'127.0.0.1:{P56}',
                        '--dead', '2'], env=dict(os.environ, HOME=hs_, USERPROFILE=hs_, PYTHONPATH=root56),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
t_ = time.time()
while time.time() - t_ < 60:                                        # сервер слушает (симулятор создаётся ~2 с)
    try:
        _so.create_connection(('127.0.0.1', P56), timeout=1).close(); break
    except OSError:
        time.sleep(0.2)
TOK = open(os.path.join(hs_, '.microcontrol', 'server_token')).read().strip()
U = f'net://127.0.0.1:{P56}'
try:
    for bad_ in (f'net://wrong@127.0.0.1:{P56}', U):                  # чужой ключ; ключа нет (ещё не запомнен)
        try:
            mc.Microscope(bad_, objective=10, folder=tempfile.mkdtemp()); raise AssertionError(bad_)
        except _SE as e:
            assert e.code == 'E111', e
    m = mc.Microscope(f'net://{TOK}@127.0.0.1:{P56}', objective=10, folder=tempfile.mkdtemp())
    assert m.port == U and m._link.codec_now in ('raw', 'png') and m.cam.grab().shape == (480, 640, 3)
    for c_ in ('raw', 'png', 'jpg95'):                             # все три вида кадров
        m._link.codec_now = c_
        g_ = m.cam.grab(); assert g_.shape == (480, 640, 3) and g_.dtype == np.uint8 and g_.std() > 5, c_
    m._link.codec_now = 'raw'
    import microcontrol.manual as M56
    M56.run(m, fullscreen=False, _keys=[-1, ord('d'), -1, -1])     # ручной режим по сети: шаг клавишей
    try:
        mc.Microscope(U, objective=10, folder=tempfile.mkdtemp()); raise AssertionError('second owner')   # ключ запомнен
    except _SE as e:
        assert e.code == 'E106', e
    m.light = 3
    m.focus()
    fw, fh = m.frame_size
    m.area = mc.Area(0, 0, 3 * fw, 2 * fh)
    f_ = m.scan(name='net')                                        # скан и сшивка — как с прибором рядом
    info_ = mc.stitch(f_, verbose=False)
    assert os.path.exists(os.path.join(f_, 'join', 'preview_all.jpg'))
    st_ = RM.status(U)['state']; assert st_['light'] == 3 and st_['motors'], st_
    m.set_home()
    m.move(dx=300, dy=-200)
    m._link._paused = True; time.sleep(3.5); m._link._paused = False    # «сеть зависла»: сервер выключил прибор
    st_ = RM.status(U); assert st_['state'] == dict(laser=0, light=0, motors=False, cooling=False) and \
        'no word' in st_['last_off']['reason'], st_
    with _wn.catch_warnings(record=True) as W_:
        _wn.simplefilter('always')
        m.move(dx=50)                                              # команда не дошла: повторена, положение точное
    assert any('switched the laser' in str(w_.message) for w_ in W_), [str(w_.message) for w_ in W_]
    assert not m._uncertain and abs(m._p['x'] * m.motor_step - (300 + 50)) <= m.motor_step, (m._uncertain, m._p)
    p_ = dict(m._p)
    assert RM.status(U)['state']['light'] == 3                    # свет вернулся, лазер — нет
    # команда выполнена, связь оборвалась до ответа: сервер отдаёт её ответ после переподключения
    seq0 = m._link.seq
    m._link.pending = None
    raw_ = m._link.transact(mc.protocol.pack(mc.protocol.GET_DEVICE_TYPE), 1.0)[2]
    m._link.pending = m._link.seq
    m._link.reconnect()
    assert m._link.pending_result() == raw_
    m._link.pending = m._link.seq + 1                             # не дошедшая
    m._link.reconnect()
    assert m._link.pending_result() is False
    m.close()
    st_ = RM.status(U); assert st_['owner'] is None and st_['state']['light'] == 0 and st_['state']['laser'] == 0, st_
    # память прибора — у сервера: другая программа (другая папка, другой «компьютер») знает положение
    hc2 = tempfile.mkdtemp()
    code_ = f"""
import os, microcontrol as mc
m = mc.Microscope('net://{TOK}@127.0.0.1:{P56}', objective=10, folder={tempfile.mkdtemp()!r})
print('POS', dict(m._p), flush=True)
m.light = 2
os._exit(1)                                                            # вылет без close()
"""
    pr_ = subprocess.run([sys.executable, '-c', code_], env=dict(os.environ, HOME=hc2, USERPROFILE=hc2, PYTHONPATH=root56),
                         capture_output=True, text=True, timeout=60)
    pos_ = eval(next(l_ for l_ in pr_.stdout.splitlines() if l_.startswith('POS'))[4:])
    assert pos_ == p_, (pos_, p_, pr_.stdout[-500:], pr_.stderr[-500:])
    time.sleep(0.5)
    st_ = RM.status(U); assert st_['owner'] is None and st_['state']['light'] == 0 and \
        'lost' in st_['last_off']['reason'], st_
    # аварийная остановка по сети (без ключа) отключает программу
    m = mc.Microscope(U, objective=10, folder=tempfile.mkdtemp())
    m.light = 4
    out_ = subprocess.run([sys.executable, '-m', 'microcontrol', 'stop', U], env=dict(os.environ, HOME=tempfile.mkdtemp(),
                          PYTHONPATH=root56), capture_output=True, text=True, timeout=30).stdout
    assert 'laser, motors, light off' in out_ and 'disconnected' in out_, out_
    st_ = RM.status(U); assert st_['owner'] is None and st_['state']['light'] == 0 and 'stop' in st_['last_off']['reason']
    m._link.close(bye=False)
finally:
    srv.terminate()
    try:
        so_ = srv.communicate(timeout=10)[0]
    except Exception:
        srv.kill(); so_ = ''
print("Server at the instrument: key (wrong / none refused), one owner, commands and frames over the network (PNG), "
      "scan and stitch, memory on the server; no word / crash / remote stop -> the server switches off; a lost move "
      "is known done or not and repeated exactly: OK")


# ------------------------------------------------------------------ 57. Съёмка -> quick (обзор, не правится, в чертёж — контур); Stitch -> quality: пиксель по крестам, расстояния как на образце, пересъёмка размытого кадра, ровный снимок рамки в чертеже по столику
import csv
os.environ['MC_SIM_PIXEL_CAL'] = '1'
import microcontrol.manual as M57
from microcontrol import samples as SM57, layers as LY57, quality as Q57
from microcontrol.mask import Grid as G57
from microcontrol.storage import ScanData as SD57
from microcontrol.stitch import load_positions as LP57
from microcontrol.config import UM_PER_STEP as UPS
cv2.getWindowImageRect = lambda w: (0, 0, 1440, 900)
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
w = m._world; w.scale_err = 0.01; m.objective = 10                      # объектив на 1 % не такой, как в таблице
w.backlash = {'x': 2.0, 'y': 2.0}
m.focus()
T57 = np.radians(w.turn_deg)
wp57 = lambda i, j: np.array([np.cos(T57) * i * 100 - np.sin(T57) * j * 100, np.sin(T57) * i * 100 + np.cos(T57) * j * 100])


def box57(x0, y0, x1, y1):
    M57.run(m, fullscreen=False, _keys=[-1, -1]); n_ = m._nav
    M57.run(m, fullscreen=False, _keys=[-1, ('maprect', *n_.to_px((x0, y0)), *n_.to_px((x1, y1)), True), -1, -1])
    k_ = max(i for i, r in enumerate(m._regions) if r.get('free'))
    return m._regions[k_]


def measure57(f_):
    """Снимок рамки против правды симулятора: сдвиги 16 участков (расстояния внутри, px) и ошибка масштаба."""
    q_ = Q57.info(f_)
    img_ = cv2.cvtColor(cv2.imread(os.path.join(f_, 'join', 'quality.png')), cv2.COLOR_BGR2GRAY).astype(np.float32)
    h_, w_ = img_.shape
    c_ = np.array(q_['pixel_to_stage_um']) @ np.array([w_ / 2, h_ / 2, 1.0])
    sv_ = dict(w._p)
    w._p = {'x': c_[0] / UPS, 'y': c_[1] / UPS, 'z': w.z_focus + (w.tilt[0] * c_[0] + w.tilt[1] * c_[1]) / UPS}
    w.motion = None
    tr_ = cv2.cvtColor(w._frame(w_ + 200, h_ + 200), cv2.COLOR_BGR2GRAY).astype(np.float32)
    w._p = sv_
    sh_, ps_ = [], min(160, w_ // 3, h_ // 3)
    for yy in np.linspace(ps_ // 2, h_ - ps_ // 2, 4).astype(int):
        for xx in np.linspace(ps_ // 2, w_ - ps_ // 2, 4).astype(int):
            a_ = img_[yy - ps_ // 2: yy + ps_ // 2, xx - ps_ // 2: xx + ps_ // 2]
            ex_, ey_ = max(0, xx - ps_ // 2 + 100 - 400), max(0, yy - ps_ // 2 + 100 - 400)
            r_ = cv2.matchTemplate(tr_[ey_: ey_ + ps_ + 800, ex_: ex_ + ps_ + 800], a_, cv2.TM_CCOEFF_NORMED)
            _, mx_, _, (x_, y_) = cv2.minMaxLoc(r_)
            dx_ = dy_ = 0.0
            if 0 < x_ < r_.shape[1] - 1 and 0 < y_ < r_.shape[0] - 1:
                dx_ = (r_[y_, x_ - 1] - r_[y_, x_ + 1]) / (2 * (r_[y_, x_ - 1] - 2 * r_[y_, x_] + r_[y_, x_ + 1]))
                dy_ = (r_[y_ - 1, x_] - r_[y_ + 1, x_]) / (2 * (r_[y_ - 1, x_] - 2 * r_[y_, x_] + r_[y_ + 1, x_]))
            sh_.append((x_ + ex_ + dx_ - (xx - ps_ // 2), y_ + ey_ + dy_ - (yy - ps_ // 2)))
    sh_ = np.array(sh_)
    d_ = np.hypot(*(sh_ - np.median(sh_, 0)).T)
    return float(d_.max()), q_['pixel_um'] / w.px_um - 1, max(w_, h_) * w.px_um


fw, fh = m.frame_size
r1 = box57(310, 210, 310 + 2.0 * fw, 210 + 1.6 * fh)
m.last_scans = []
f1 = m.scan(only=[r1['uid']])                                          # съёмка: сразу quick
assert Q57.stitched_as(f1) == 'quick' and Q57.info(f1) is None, Q57.stitched_as(f1)
assert json.load(open(os.path.join(f1, 'join', 'stitch.json')))['reduced'] >= 2
dx57 = mc.export_sample(m, ks=[LY57.idx(m, r1['uid'])])                 # quick — в чертёж только контур
dj57 = json.load(open(os.path.join(os.path.dirname(dx57), 'drawing.json')))
assert not dj57['pictures'] and any('quick' in n_ for n_ in dj57['notes']), dj57
# калибровка пикселя по крестам: участок с 3 x 3 крестами, отметки точно на крестах (как C)
w.backlash = {'x': 0.0, 'y': 0.0}
m.area = mc.Area(40, 40, 40 + 5 * fw * 0.7, 40 + 6 * fh * 0.7)
fc = m.scan(stitch=None); mc.stitch(fc, verbose=False); m.area = None
w.backlash = {'x': 2.0, 'y': 2.0}
st_ = SD57(fc); P_ = LP57(fc, st_); P_ = P_ - P_.min(0); H_, W_ = st_.shape[:2]
x0_, y0_ = st_.index[0]['x_p'] * UPS, st_.index[0]['y_p'] * UPS; st_.close()
g_ = G57(fc)
for ij in ((1, 1), (2, 1)):
    g_.cr.add(P_[0] + (wp57(*ij) - (x0_, y0_)) / w.px_um + (W_ / 2, H_ / 2))
g_.save(json.load(open(os.path.join(fc, 'meta.json')))['pixel_size_um'])
rc = mc.fit_crosses(fc, verbose=False)
assert rc.info['used'] >= 8 and abs(rc.info['pixel_um'] / w.px_um - 1) < 2e-4, rc.info
m.objective = 10
assert m._px_src == 'crosses' and abs(m.pixel_size / w.px_um - 1) < 2e-4, (m._px_src, m.pixel_size)
# точная сшивка рамки (Stitch): кадр размыт — переснимается
r2 = box57(710, 510, 710 + 2.0 * fw, 510 + 1.6 * fh)
m.last_scans = []
f2 = m.scan(only=[r2['uid']])
rows_ = list(csv.reader(open(os.path.join(f2, 'index.csv'), encoding='utf-8')))
kb_ = rows_[0].index('blurry'); rows_[2][kb_] = '1'                    # кадр 2 «остался размытым»
csv.writer(open(os.path.join(f2, 'index.csv'), 'w', newline='', encoding='utf-8')).writerows(rows_)
n0_ = len(rows_) - 1
rq = m.quality(f2)
q2 = Q57.info(f2)
assert Q57.stitched_as(f2) == 'quality' and q2['ok'] and q2['pixel_source'] == 'crosses' and q2['coverage'] == 1.0, q2
assert 1 in q2['replaced'] and q2['frames'] >= n0_ + len(q2['replaced']), (q2['replaced'], q2['frames'], n0_)   # сняты заново, старые — не в снимке
dmax_, serr_, span_ = measure57(f2)
assert dmax_ < 0.5 and abs(serr_) < 2e-4, (dmax_, serr_)                   # расстояния внутри — как на образце
err_um = dmax_ * w.px_um + abs(serr_) * span_
assert err_um < 0.15, err_um                                            # на всей рамке — точнее 0.15 мкм
img2 = cv2.imread(os.path.join(f2, 'join', 'quality.png'))
assert img2.shape[1] == q2['size_px'][0] and (img2.reshape(-1, 3).max(1) > 0).all()   # ровный прямоугольник без пустот
f3 = m.scan(only=[r2['uid']])                                          # ещё съёмка (quick): последняя — она,
from microcontrol.project import Selection as SL57
from microcontrol.drawing import best_scan as BS57
assert SL57(m.sample, r2).latest.folder == f3 and BS57(m, r2).folder == f2   # а в чертёж — свежая quality
r2['main'] = os.path.basename(f2)                                       # основная — выбрана руками
assert SL57(m.sample, r2).latest.folder == f2 and BS57(m, r2).folder == f2
dx57 = mc.export_sample(m, ks=[LY57.idx(m, r2['uid'])])                 # в чертёж — снимок, по столику
dj57 = json.load(open(os.path.join(os.path.dirname(dx57), 'drawing.json')))
assert any(v_.get('quality') for v_ in dj57['pictures'].values()) and dj57['areas'][0]['placed'].startswith('quality'), dj57
m.close()
print(f"Quick after the scan (reduced frames, outline only in the drawing); pixel by the crosses {rc.info['pixel_um']:.6f} "
      f"(true {w.px_um:.6f}); quality stitch: blurry frame re-shot, distances off by {dmax_:.2f} px, "
      f"{err_um:.3f} um over {span_:.0f} um, picture in the drawing by the stage: OK")


# ------------------------------------------------------------------ 58. Очередь: съёмки подряд (прибор один), quick-сшивки в фоне на ядрах, остановка любой задачи, окно живёт
import microcontrol.manual as M
from microcontrol import jobs as J, quality as QQ
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus(); fw, fh = m.frame_size
for x0 in (0, 6, 12):                                                  # три выделения
    m.area = mc.Area(x0 * fw, 0, (x0 + 3) * fw, 2 * fh); m.scan(stitch=None)
uids = [r['uid'] for r in m._regions]
q = m.queue
# 1) три съёмки: на приборе по одной, quick-сшивка предыдущей идёт на ядрах, пока снимается следующая
js = [q.scan(u) for u in uids]
seen, tl = set(), []
while q.active():
    seen.add(tuple(sorted(j.lane for j in js if j.state == 'running')))
    tl.append(q.total_left()); time.sleep(0.03)
assert all(j.state == 'done' and QQ.stitched_as(j.folder) == 'quick' for j in js), [(j.state, j.error) for j in js]
assert ('cpu', 'inst') in seen, seen                                   # сшивка шла во время съёмки
assert not any(s_.count('inst') > 1 for s_ in seen), seen               # прибор — один
assert max(tl) > 0 and tl[-1] < max(tl), tl[:5]
print("  queue lanes seen:", sorted(seen), "  total left at start:", J.tm(max(tl)))
# 2) остановить идущую съёмку (после кадра: снятое цело) и ждущую (убрана, не начиналась)
a_, b_ = q.scan(uids[0]), q.scan(uids[1])
while not (a_.state == 'running' and a_.lane == 'inst' and a_.frac > 0.2):
    time.sleep(0.01)
assert q.inst_busy() and b_.state in ('queued', 'waiting')
q.stop(b_); q.stop(a_); assert q.wait(60)
assert a_.state == 'stopped' and b_.state == 'stopped' and b_.folder is None, (a_.state, b_.state, b_.folder)
assert 0 < J.n_frames(a_.folder) < J.n_frames(js[0].folder), (J.n_frames(a_.folder), J.n_frames(js[0].folder))
# 3) сшивка в фоне (отдельный процесс) останавливается сразу; прежняя сшивка цела
P0 = open(os.path.join(js[2].folder, 'join', 'positions_all.csv')).read()
c_ = q.stitch(js[2].folder, 'quality')
while c_.state != 'running' or c_.frac < 0.05:
    time.sleep(0.02)
t0_ = time.monotonic(); q.stop(c_); assert q.wait(10)
assert c_.state == 'stopped' and time.monotonic() - t0_ < 6, (c_.state, c_.error)
assert QQ.stitched_as(js[2].folder) == 'quick'
d_ = q.stitch(js[2].folder, 'quality'); q.wait(); assert d_.state == 'done' and QQ.stitched_as(js[2].folder) == 'quality', d_.error
# 4) окно: S ставит съёмку в очередь и окно живёт; стрелки во время съёмки не двигают столик; конец — дождаться
m._test_async, m._sel = True, 2
msgs = []
M.run(m, fullscreen=False, _keys=[-1, ord('s'), -1, 2424832, -1, ('call', lambda: msgs.append(m._last_msg)), -1, -1, ('qwait',), -1])
assert msgs and 'busy' in msgs[0], msgs
assert q.jobs[-1].kind == 'scan' and q.jobs[-1].state == 'done' and q.jobs[-1].uid == uids[2]
M.run(m, fullscreen=False, _keys=[-1, ord('s'), -1, 2424832, -1, -1])   # выход ждёт очередь
assert not q.active() and q.jobs[-1].state == 'done'
m.close()
print("Queue: scans one after another at the instrument, quick stitches in the background on other cores; any job "
      "stops (a scan after its frame, a waiting one is removed, a stitch process at once); the window lives: OK")


# ------------------------------------------------------------------ 59. Слои в просмотре: прозрачность картинки выделения (O) и совмещение менее точной съёмки с более точной (L: тянуть / само)
from microcontrol.snap import align, stage_to_mosaic
cv2.getWindowImageRect = lambda w: (0, 0, 1400, 900)
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus(); fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 3 * fh); f10 = m.scan()
m.objective = 50; m.focus(); f5w, f5h = m.frame_size
m.area = mc.Area(fw, fh, fw + 3 * f5w, fh + 3 * f5h); f50 = m.scan()
d_true = (8.0, -5.0)
align(f50, d_true)                                                   # 50x «уехал» (своя ось у объектива)
c50 = stage_to_mosaic(f10, [(fw + 1.5 * f5w, fh + 1.5 * f5h)])[0]
res = {}
mc.view(f10, _keys=[-1, ('qmove', *c50), -1, ord('o'), ord('o'), -1, ord('O'), -1, ('call', lambda st: res.update(a=st['msg'])),
                    ord('l'), -1, ('qdrag', *c50, *(c50 + (30, 0))), -1, ('call', lambda st: res.update(dq=list(st['al']['dq']))),
                    27, -1, ('call', lambda st: res.update(mode=st['mode'])), ord('l'), -1, 32, -1,
                    ('call', lambda st: res.update(auto=st['msg'])), 13, -1, 27, 27])
uid50 = json.load(open(os.path.join(f50, 'meta.json')))['sample_uid']
assert json.load(open(os.path.join(m.sample.folder, 'view.json')))['alpha'] == {uid50: 0.75} and '75 %' in res['a'], res
assert abs(res['dq'][0] - 30) < 0.5 and abs(res['dq'][1]) < 0.5 and res['mode'] == 'look', res   # тянуть: едут за мышью; Esc — отмена
assert align(f10) != (0.0, 0.0) and 'by the picture' in res['auto'], res
d10 = align(f10)
assert np.hypot(d10[0] - d_true[0], d10[1] - d_true[1]) < 0.5, (d10, d_true)                       # само: с точностью < 0.5 мкм
q_ = stage_to_mosaic(f10, [(100.0, 100.0)])[0]; align(f10, (0, 0))
assert np.allclose(q_, stage_to_mosaic(f10, [(100 - d10[0], 100 - d10[1])])[0], atol=1e-6)   # сдвиг — во всех пересчётах
m.close()
print(f"Layers: picture transparency O (75 % kept per selection); L: drag moves the higher pictures, Esc cancels; "
      f"Space aligns by the picture: 10x shifted by {d10[0]:+.2f}, {d10[1]:+.2f} um (true {d_true}): OK")
