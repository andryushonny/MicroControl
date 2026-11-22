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
sv = min(Wsc_ / size_[0], (900 - mc.ui.MENU_H) / size_[1]) * 0.95
exp_ = size_ / 2 + (np.array([u0 + 10.5, v0 + 4.5]) - (Wsc_ / 2, (900 - mc.ui.MENU_H) / 2)) / sv
assert np.hypot(*(g2.cr.marks[-1] - exp_)) < 0.6 / sv + 0.5, (g2.cr.marks[-1], exp_)
u1, v1 = view_xy(world(6, 5)); u1, v1 = u1 - 3, v1 + 2
mc.view(f, _keys=[ord('c'), -1, ('click', u1, v1), -1, ('drag', u1, v1, u1 + 6, v1), -1, 26, -1, 26, -1, 25, -1,
                  26, -1, ord('r'), -1, ('click', 300, 300), -1, ('click', 420, 380), -1, 27, 27])
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
M.run(m, fullscreen=False, _keys=[ord('1'), -1, R_, -1, ord('2'), -1, ord('s'), -1, -1, ord('v'), 27, 27, -1])
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
g = Grid(f); p0 = np.array([150.0, 120.0]); a_ = 100 / px_um
g.cr.add(p0); g.cr.add(p0 + (a_, 0)); g.cr.add(p0 + (0, a_))
g.selected |= {(0, 0), (1, 0), (3, 1), (3, 2)}                     # две фигуры по 2 квадрата
g.save(px_um)
print(LY.sync_brush(m))
kids = [k for k, r in enumerate(m._regions) if r.get('src')]
assert len(kids) == 2 and all(m._regions[k]['parent'] == m._regions[0]['uid'] and m._regions[k]['kind'] == 'mask'
                              for k in kids), m._regions
assert abs(LY.geom(m, m._regions[kids[0]]).area - 2 * 100 * 100) < 50      # 2 квадрата 100 x 100 мкм, контур без клеток
assert [d for k, d in SM.tree(m)] == [0, 1, 1]                      # матрёшка: карта 10x > две маски
assert LY.sync_brush(m) is None                                     # ничего не изменилось
g = Grid(f); g.selected |= {(2, 0)}; g.save(px_um)                  # кисть дорисовала квадрат — выделение растёт
LY.sync_brush(m)
assert len([r for r in m._regions if r.get('src')]) == 2 and \
    abs(LY.geom(m, [r for r in m._regions if r.get('src')][0]).area - 3e4) < 80
ks = [k for k, r in enumerate(m._regions) if r.get('src')]
print(LY.union(m, ks))                                              # слить: одно выделение, отпущено от кисти
assert len([r for r in m._regions if r.get('kind') == 'mask']) == 1 and not any(r.get('src') for r in m._regions)
assert LY.sync_brush(m) is None                                     # кисть его не пересоздаёт
k20 = next(k for k, r in enumerate(m._regions) if r.get('kind') == 'mask')
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
m.close(); print("Brush -> masks (one per figure), merge / minus / group / ungroup, outline in sample.json: OK")

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
g = Grid(f); p0 = np.array([150.0, 120.0]); a_ = 100 / px_um
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
g = Grid(f); q0 = np.array([150.0, 120.0]); a_ = 100 / px_um
g.cr.add(q0); g.cr.add(q0 + (a_, 0)); g.cr.add(q0 + (0, a_)); g.selected |= {(0, 0)}; g.save(px_um)
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


# ------------------------------------------------------------------ 33. Маска — черновик; участок рисуется в просмотре (D) и на карте внутри маски; чертёж образца
import glob as _glob
import ezdxf
import microcontrol.manual as M
import microcontrol.viewer as V_
from microcontrol import samples as SM, layers as LY
from microcontrol.mask import Grid
m = mc.Microscope('sim', objective=10, folder=tempfile.mkdtemp()); m.focus()
fw, fh = m.frame_size
m.area = mc.Area(0, 0, 3 * fw, 2 * fh)
f = m.scan(); mc.stitch(f, verbose=False)
px_um = json.load(open(os.path.join(f, 'meta.json')))['pixel_size_um']
g = Grid(f); p0 = np.array([150.0, 120.0]); a_ = 100 / px_um
g.cr.add(p0); g.cr.add(p0 + (a_, 0)); g.cr.add(p0 + (0, a_)); g.selected |= {(1, 0), (2, 0)}; g.save(px_um)
LY.sync_brush(m); SM.persist(m)
km = next(k for k, r in enumerate(m._regions) if r.get('kind') == 'mask')
assert SM.title(m, km).startswith('Mask') and SM.frames_of(m, km) == (0, 0)
assert len(m.scan(name='again') and m.last_scans) == 1            # маска не снимается
f = mc.project.Selection(m.sample, m._regions[0]).latest.folder; mc.stitch(f, verbose=False)
g = Grid(f); g.cr.add(p0); g.cr.add(p0 + (a_, 0)); g.cr.add(p0 + (0, a_)); g.selected |= {(1, 0), (2, 0)}; g.save(px_um)
LY.sync_brush(m); SM.persist(m)
km = next(k for k, r in enumerate(m._regions) if r.get('kind') == 'mask')
gm = LY.geom(m, m._regions[km]); c_ = np.array(gm.centroid.coords[0])
m.objective = 50; m.focus()                                         # 50x: участок внутри маски — на карте
M.run(m, fullscreen=False, _keys=[-1, -1])
nav = m._nav
M.run(m, fullscreen=False, _keys=[-1, ('maprect', *nav.to_px(c_ - 30), *nav.to_px(c_ + 30)), -1, -1])
k50 = next(k for k, r in enumerate(m._regions) if r['obj'] == 50)
assert m._regions[k50]['parent'] == m._regions[km]['uid'], m._regions      # ребёнок маски, перекрытие разрешено
V_.view(f, _keys=[-1, ord('d'), ('drag', 300, 250, 380, 310), -1, 27])   # D в просмотре: участок по картинке
assert json.load(open(os.path.join(f, 'grid.json')))['areas'], "no drawn area"
print(LY.sync_drawn(m)); SM.persist(m)
assert len([r for r in m._regions if r['obj'] == 50]) == 2 and not json.load(open(os.path.join(f, 'grid.json')))['areas']
m.scan(); [mc.stitch(x, verbose=False) for x in m.last_scans]
m._regions[next(k for k, r in enumerate(m._regions) if r['obj'] == 50 and k != k50)]['export'] = False
dx = mc.export_sample(m)                                            # 10x + один 50x (второй снят с экспорта)
doc = ezdxf.readfile(dx if dx.endswith('.dxf') else dx[:-4] + '.dxf')
ims = doc.modelspace().query('IMAGE')
assert len(ims) == 2 and len(doc.modelspace().query('LINE[layer=="CROSSES"]')) > 10, (len(ims),)
im50 = next(i for i in ims if '50x' in i.dxf.layer)
pl = next(p for p in doc.modelspace().query('LWPOLYLINE') if p.dxf.layer == im50.dxf.layer)
V = np.array([p[:2] for p in pl.get_points()])
ins, u, v = np.array(im50.dxf.insert)[:2], np.array(im50.dxf.u_pixel)[:2], np.array(im50.dxf.v_pixel)[:2]
w_, h_ = im50.dxf.image_size.x, im50.dxf.image_size.y
uv = np.linalg.solve(np.c_[u, v], (V - ins).T).T                   # углы контура — в пикселях картинки
tol = 0.6 * m.motor_step / np.hypot(*u)                           # кадры стоят на целых шагах мотора (5.6 мкм)
assert (uv[:, 0] > -tol).all() and (uv[:, 0] < w_ + tol).all() and (uv[:, 1] > -tol).all() and (uv[:, 1] < h_ + tol).all(), uv
f10 = mc.project.Selection(m.sample, m._regions[0]).latest.folder     # чертёж по квадратам: общая грань — на каждом слое
g = Grid(f10); g.selected = {(1, 0), (2, 0)}; g.save(px_um)
mc.export(f10)
d2 = ezdxf.readfile(os.path.join(f10, 'export', 'drawing.dxf')).modelspace()
sq = d2.query('LWPOLYLINE[layer=="SQUARES"]'); ol = d2.query('LINE[layer=="OUTLINE"]')
assert len(sq) == 2 and all(len(p_) == 4 and p_.closed for p_ in sq) and len(ol) == 6, (len(sq), len(ol))
m.close(); print("Mask is a draft, areas drawn in the view (D) and on the map inside it, one drawing of chosen areas;"
                 " every square has all 4 sides, the outline repeats the shared edges: OK")
