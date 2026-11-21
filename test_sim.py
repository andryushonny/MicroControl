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
    s = min(Wsc / fr.size[0], 900 / fr.size[1]) * 0.95
    v = np.array([Wsc / 2, 450]) + s * (M - fr.size / 2)
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
keys += [ord('y'), -1, ord('e'), -1, 27, 27]
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
print("Square numbers:", gg.numbering()[2], "  hints:", len(gg.useful), "of", len(gg.scores) if hasattr(gg, 'scores') else '?')
hint_ok = {gg.cell_at(mos_xy(world(3.5, 2.5))), gg.cell_at(mos_xy(world(5.5, 3.5)))}
# жёлтая подсказка (поиск аномалий) нашла чешуйки; тёмные пятна текстуры симулятора — тоже «что-то»,
# но пустые квадраты не отмечены все подряд, и чешуйки — среди самых заметных
assert hint_ok <= set(gg.useful), (hint_ok, gg.useful)
med = sorted(gg.useful.values())[len(gg.useful) // 2]
assert len(gg.useful) <= 0.6 * gg.hint_squares and all(gg.useful[c] >= med for c in hint_ok), (hint_ok, gg.useful)
# простой щелчок ставит крест ровно в точку щелчка (без притяжения), отметку можно перетащить мышью
Wsc_ = 1280 - mc.samples.CAR_W
n0 = len(gg.cr.marks)
u0, v0 = view_xy(world(5, 5)); u0, v0 = u0 - 3, v0 + 2
mc.view(f, _keys=[ord('c'), -1, ('click', u0, v0), -1, ('drag', u0, v0, u0 + 10, v0 + 4), -1, 27, 27])
g2 = Grid(f)
assert len(g2.cr.marks) == n0 + 1, (n0, len(g2.cr.marks))
fr = Frames(f); size_ = np.array(fr.size, float); fr.close()
sv = min(Wsc_ / size_[0], 900 / size_[1]) * 0.95
exp_ = size_ / 2 + (np.array([u0 + 10.5, v0 + 4.5]) - (Wsc_ / 2, 450)) / sv
assert np.hypot(*(g2.cr.marks[-1] - exp_)) < 0.6 / sv + 0.5, (g2.cr.marks[-1], exp_)
u1, v1 = view_xy(world(6, 5)); u1, v1 = u1 - 3, v1 + 2
mc.view(f, _keys=[ord('c'), -1, ('click', u1, v1), -1, ('drag', u1, v1, u1 + 6, v1), -1, 26, -1, 26, -1, 25, -1,
                  26, -1, ord('l'), -1, ('click', 300, 300), -1, ('click', 420, 380), -1, 27, 27])
assert len(Grid(f).cr.marks) == n0 + 1, (n0, len(Grid(f).cr.marks))              # крест, сдвиг -> Ctrl+Z x2, Ctrl+Y, Ctrl+Z
print("Cross marks: exactly at the click, draggable, Ctrl+Z / Ctrl+Y: OK")
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
subs = mc.samples.samples(ses)
assert [os.path.basename(x) for x in subs] == ['selection_01', 'selection_02'], subs
assert all(json.load(open(os.path.join(x, 'meta.json')))['sample_index'] == i + 1 for i, x in enumerate(subs))
r = mc.stitch(ses, verbose=False)
assert len(r) == 2, r
Wsc = 1280 - mc.samples.CAR_W
mc.view(ses, _keys=[-1, ord('.'), -1, -1, ('click', 1280 - 25, 64), -1, ('click', 1280 - 25, 64), -1, -1, 27])
left = mc.samples.samples(ses)
assert [os.path.basename(x) for x in left] == ['selection_02'] and os.path.isdir(os.path.join(ses, '_removed', 'selection_01'))   # x у 1-го
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
r = C('area_corner', corner=1); assert not r['ok'] and r['error']['code'] == 'E704', r
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
r = ok(C('find_hints', folder=folder)); print('hints', r['hints'], r['strongest'][:3])
r = ok(C('list_squares', folder=folder, which='hints', limit=5)); print('sq', r['squares'][:2])
ok(C('select_squares', folder=folder, action='add_hints'))
r = ok(C('select_squares', folder=folder, action='add', squares=[[12, 8]])); print('sel', r)
r = C('select_squares', folder=folder, action='add', squares=[[12]]); assert r['error']['code'] == 'BAD_ARGS', r
r = ok(C('look', target='mosaic', folder=folder)); print('mosaic2', r['image'])
r = ok(C('export', folder=folder)); print('export', r['result'][:3])
th = mc.train_hints(os.path.join(m.dirs['scans'], folder)); assert os.path.exists(th['File']), th
hh = mc.hints(os.path.join(m.dirs['scans'], folder), model=os.path.join(m.dirs['scans'], folder)); print(hh)
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
assert len(R[1]['result']['tools']) == 26 and R[3]['result']['content'][1]['type'] == 'image'
print('mcp ok', len(R[1]['result']['tools']), 'tools; asked', len(asked))
m.close()
print("Agent: tools, approval, errors, scan, grid, hints, export, MCP: OK")

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

# ------------------------------------------------------------------ 18. Постоянные номера образцов, uid
import json, microcontrol.manual as M
from microcontrol import samples as SM
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
m.focus()
R_, D_ = 65363, 65364
k = [ord('1'), -1, R_, -1, ord('2'), -1, R_, -1, R_, -1, ord('1'), -1, R_, -1, ord('2'), -1, -1]
M.run(m, fullscreen=False, _keys=k)
assert [r['num'] for r in m._regions] == [1, 2] and len({r['uid'] for r in m._regions}) == 2
SM.delete_area(m, 0)                                   # удалить образец 1: у второго номер остаётся 2
assert [r['num'] for r in m._regions] == [2]
M.run(m, fullscreen=False, _keys=[R_, -1, R_, -1, ord('1'), -1, R_, -1, ord('2'), -1, -1])
assert [r['num'] for r in m._regions] == [2, 3], [r['num'] for r in m._regions]
ses = m.scan(name='uid')
subs = sorted(os.listdir(ses)); print(subs)
assert 'selection_02' in subs and 'selection_03' in subs and 'selection_01' not in subs
info = json.load(open(os.path.join(ses, 'session.json')))
assert [s['index'] for s in info['samples']] == [2, 3] and all(len(s['uid']) == 32 for s in info['samples'])
u2 = SM.sample_uid(os.path.join(ses, 'selection_02')); assert u2 == m._regions[0]['uid']
d1 = SM.remove_sample(os.path.join(ses, 'selection_02'))
os.makedirs(os.path.join(ses, 'selection_02')); json.dump({'sample_uid': 'x' * 32}, open(os.path.join(ses, 'selection_02', 'meta.json'), 'w'))
d2 = SM.remove_sample(os.path.join(ses, 'selection_02'))
assert os.path.isdir(d1) and os.path.isdir(d2) and d1 != d2, (d1, d2)   # ранее убранное не стёрто
m.close(); print("Sample numbers stay, uid, remove without overwrite: OK")

# ------------------------------------------------------------------ 19. То же имя скана — та же папка
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, fw, fh)
f1 = m.scan(name='Same')
f2 = m.scan(name='Same')
f3 = m.scan()
print(os.path.basename(f1), os.path.basename(f2), os.path.basename(f3))
assert f1 == f2 and f3 != f1 and len(os.listdir(m.dirs['scans'])) == 2
m.close(); print("Same name -> same folder: OK")

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
try:                                                                               # тот же объектив внутри — нельзя
    SM.corner(m, 1); raise AssertionError("same objective overlap allowed")
except mc.MicroControlError as e:
    assert 'E704' in str(e), e
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
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp())
m.focus()
p0 = m.position
R_ = 65363
M.run(m, fullscreen=False, _keys=[ord('1'), -1, R_, -1, ord('2'), -1, -1, 26, -1, 26, -1, -1, 26, -1, -1, 27])
assert not m._regions and m._pending is None and m.position == p0, (m._regions, m._pending, m.position, p0)
M.run(m, fullscreen=False, _keys=[25, -1, 25, -1, -1, 25, -1, -1, 27])                # всё вернуть
assert len(m._regions) == 1 and m.position != p0, (m._regions, m.position)
M.run(m, fullscreen=False, _keys=[ord('l'), -1, (600, 300), -1, (700, 300), -1, -1, ord('l'), -1, ord('l'), -1,
                                  (600, 300), -1, R_, -1, (600, 300), -1, -1])     # линейка; точка 2 — после сдвига
d_ = np.hypot(*(np.array(m._ruler['pts'][1]) - m._ruler['pts'][0]))
assert len(m._ruler['pts']) == 2 and abs(d_ - m.frame_size[0]) < 0.1 * m.frame_size[0], (m._ruler, d_, m.frame_size)
M.run(m, fullscreen=False, _keys=[ord('1'), -1, R_, -1, ord('2'), -1, ord('s'), -1, ord('s'), -1, -1, ord('v'), 27, 27, -1])
sf = m._last_view
assert os.path.exists(os.path.join(sf, 'join', 'positions_all.csv')), sf              # S, S: снято и сшито в окне
m.close(); print("Undo / redo in manual mode (selections and moves), ruler, scan + stitch (S) and view (V) from the window: OK")
