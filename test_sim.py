"""Проверка MicroControl на симуляторе: python test_sim.py (для окон OpenCV нужен экран)."""
import os, subprocess, sys, tempfile, time, numpy as np
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
m.manual(fullscreen=False, _keys=[ord('1'), ord('d'), ord('0'), ord('2'), ord('b'), ord('r'), ord('c'), 27])
assert not any(m._off.values()); m.goto(x=0, y=0)
area = m.manual(fullscreen=False, _keys=[ord('F'), ord('1')] + [65363] * 4 + [65364] * 3 + [ord('2'), 27])
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
m = mc.Microscope('sim', objective=10, folder=hd)                 # новый сеанс: положение восстановлено
assert m.position.x == round(54 * mc.config.UM_PER_STEP, 1) and m._uncertain == {'x', 'y', 'z'}
refused(lambda: m.goto(x=0), 'E211')
m.confirm_position()
r = m.go_home(); print(r); assert 'matches' in r['Picture']
home_true = dict(m._world.p)
m._world.motion = None; m._world._p["x"] += 3; m._world._p["y"] -= 2                       # «сбой»: столик не там, где думает программа
r = m.go_home(); print(r); assert 'shifted' in r['Picture']
r = m.go_home(correct=True); print(r)
assert all(abs(m._world.p[a] - home_true[a]) < 0.5 for a in 'xy'), m._world.p   # на самом деле дома
m._remember(moving=True); m._ser.close(); m._lock.release(); m._lock = None; m._ser = None   # «убили» во время движения
m = mc.Microscope('sim', objective=10, folder=hd)
assert m._uncertain == {'x', 'y', 'z'}
m.set_home(replace=True); assert not m._uncertain
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
st = m._frame_step(1 - mc.config.SCAN_OVERLAP)
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
    s = min(1280 / fr.size[0], 900 / fr.size[1]) * 0.95
    v = np.array([640, 450]) + s * (M - fr.size / 2)
    return int(v[0]) + 3, int(v[1]) - 2
sq = {n: mos_xy(world(*ij)) for n, ij in (('a', (2.5, 2.5)), ('b', (4.5, 3.5)), ('c', (3.5, 2.5)), ('e', (2.5, 4.5)))}
fr.close()
typed = lambda t: [ord(ch) for ch in t] + [13, -1]
clk = lambda ij: [('click', *view_xy(world(*ij))), -1]
# два креста в ряду -> наклон, полотно само докручивается (R — вернуть как снято, чтобы щёлкать по расчёту)
keys = [ord('g')] + clk((2, 2)) + clk((3, 2)) + [ord('r'), -1] + clk((9, 2)) + clk((8, 6))   # дальние уточняют
keys += [ord('r'), -1, ord('r'), -1, ord('s'), -1, ('drag', 560, 380, 720, 520), -1]
keys += [ord('n'), -1, ('click', *view_xy(world(2.5, 2.5))), -1] + typed('12 7') + \
        [('click', *view_xy(world(4.5, 3.5))), -1] + typed('14 6')      # номера литографии: X вдоль i, Y — против j
keys += [ord('y'), -1, ord('e'), -1, 27, 27]
mc.view(f, _keys=keys)
grid = json.load(open(os.path.join(f, 'grid.json')))
assert abs(grid['pitch_um'][0] - 100) < 1 and abs(grid['pitch_um'][1] - 100) < 1, grid['pitch_um']
assert abs(grid["turn_deg"] - 3.5) < 0.2 and grid['selected'], grid
assert {'selection.png', 'selection.dxf', 'selection.json'} <= set(os.listdir(os.path.join(f, 'export')))
from microcontrol.mask import Grid
gg = Grid(f)
assert len(gg.labels) == 2 and 'fixed' in gg.numbering()[2], (gg.labels, gg.numbering())
assert gg.name(*gg.cell_at(sq['c'])) == (13, 7) and gg.name(*gg.cell_at(sq['e'])) == (12, 5), \
    (gg.name(*gg.cell_at(sq['c'])), gg.name(*gg.cell_at(sq['e'])))
sj = json.load(open(os.path.join(f, 'export', 'selection.json')))
assert len(sj['names']) == len(sj['squares']) and 'S13,7' not in open(os.path.join(f, 'export', 'selection.dxf')).read()
print("Square numbers:", gg.numbering()[2], "  hints:", len(gg.useful), "of", len(gg.scores) if hasattr(gg, 'scores') else '?')
hint_ok = {gg.cell_at(mos_xy(world(3.5, 2.5))), gg.cell_at(mos_xy(world(5.5, 3.5)))}
assert hint_ok <= set(gg.useful) and len(gg.useful) <= 4, (hint_ok, gg.useful)    # жёлтая подсказка нашла чешуйки
n0 = len(json.load(open(os.path.join(f, 'grid.json')))['selected'])
mc.view(f, _keys=[ord('s'), -1, ('rclick', 640, 450), -1, 27, 27])      # правый щелчок по зелёному — убрать область
n1 = len(json.load(open(os.path.join(f, 'grid.json')))['selected'])
assert n1 < n0, (n0, n1)
mc.clean(f, confirm=True)
assert not os.path.exists(os.path.join(f, 'frames'))
print("Stitch / mask / export: OK")
