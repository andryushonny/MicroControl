"""Сторож: безопасное отключение прибора, если программа вылетела.

При подключении к прибору (не к симулятору) запускается отдельный маленький процесс-сторож. Он держит трубу от
программы и ничего не делает, пока программа жива. Штатное m.close() пишет в трубу «bye» — сторож тихо выходит.
Если программа умерла, не закрыв прибор (необработанная ошибка в потоке, падение OpenCV/драйвера, закрытое окно
консоли, перезапуск ядра Jupyter, kill), система закрывает трубу без «bye». Тогда сторож:
  1. ждёт, пока система освободит порт и блокировку прибора (до 20 с), и берёт блокировку сам;
  2. выключает лазер, моторы X, Y, свет и (если включали) охлаждение ПЗС — как m.close();
  3. пишет, что сделал, в журнал logs/MC_..._guard.log и отпускает прибор.
Сторож запускается отдельно от консоли и группы процессов программы, поэтому закрытие окна или Ctrl+C его не
задевают. Зависшую (но живую) программу сторож не трогает: порт держит она — для этого python -m microcontrol stop.
"""
import os, subprocess, sys, time

BYE = b"bye\n"


class Guard:
    """Ручка сторожа в программе: done() — штатное отключение (сторож выходит, ничего не трогая)."""
    def __init__(self, proc):
        self.proc = proc

    def done(self):
        try:
            self.proc.stdin.write(BYE)
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc = None

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None


def start(port, cooling=False, logdir=None):
    """Запустить сторожа для порта. -> Guard или None (не удалось — программа работает и без него)."""
    cmd = [sys.executable, os.path.abspath(__file__), str(port), "1" if cooling else "0", os.path.abspath(logdir or "logs")]
    kw = dict(stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    if os.name == "nt":                     # своя консоль/группа: закрытие окна и Ctrl+C программы его не задевают;
        flags = [0x00000008 | 0x00000200 | 0x01000000, 0x00000008 | 0x00000200]   # + выход из задания (job), если можно
        for fl in flags:                    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP [| CREATE_BREAKAWAY_FROM_JOB]
            try:
                return Guard(subprocess.Popen(cmd, creationflags=fl, **kw))
            except OSError:
                continue
        return None
    try:
        return Guard(subprocess.Popen(cmd, start_new_session=True, **kw))
    except OSError:
        return None


def _mod(name):
    """Модуль пакета без его __init__ (сторож лёгкий: без OpenCV, numpy и прочего — только порт и журнал)."""
    import importlib, types
    if __package__:
        return importlib.import_module("." + name, __package__)
    if "microcontrol" not in sys.modules:
        pk = types.ModuleType("microcontrol")
        pk.__path__ = [os.path.dirname(os.path.abspath(__file__))]
        sys.modules["microcontrol"] = pk
    return importlib.import_module("microcontrol." + name)


def switch_off(port, cooling=False, answer_s=0.5):
    """Открыть порт и выключить лазер, моторы X, Y, свет, охлаждение. -> список сделанного."""
    import serial
    P = _mod("protocol")
    done = []
    with serial.Serial(port, 115200, timeout=answer_s, write_timeout=2.0) as s:
        steps = [(P.LASER, b"\x00\x00", "Laser"), (P.TURN_MOTOR, b"\x00", "Motors X, Y"), (P.LIGHT, b"\x00\x00", "Light")]
        if cooling:
            steps.append((P.SET_TEC, P.tec_data(P.WARM_C), "Cooling"))
        for cmd, data, name in steps:
            try:
                s.reset_input_buffer()
                s.write(P.pack(cmd, data))
                s.flush()
                try:
                    P.read_frame(s, answer_s)
                    done.append(f"{name} off.")
                except Exception:
                    done.append(f"{name}: off sent, no answer.")
                time.sleep(0.05)
            except Exception as e:
                done.append(f"{name}: could not send ({e}).")
    return done


def _wait_bye():
    """Ждать конца трубы. True — штатное отключение (bye), False — программа умерла."""
    got = b""
    try:
        while True:
            b = sys.stdin.buffer.read(64)
            if not b:
                break
            got += b
            if BYE in got:
                return True
    except Exception:
        pass
    return BYE in got


def main(argv):
    port, cooling, logdir = argv[0], argv[1] == "1", argv[2]
    if _wait_bye():
        return 0
    L, ControlLock = _mod("log"), _mod("control").ControlLock
    os.makedirs(logdir, exist_ok=True)
    L.start(logdir, suffix="_guard")
    import logging
    L.event('SAF-001', f"The program ended without closing {port} (crash or killed). Switching the instrument off.",
            logging.WARNING)
    lock, t_end = None, time.time() + 20
    while lock is None and time.time() < t_end:   # блокировку снимает система, когда процесс программы умер
        try:
            lock = ControlLock(port)
        except OSError:
            time.sleep(0.2)
    if lock is None:
        L.event('SAF-001', f"Control of {port} is still held by another process; switching off anyway.", logging.WARNING)
    done, err = [], None
    while time.time() < t_end + 5:
        try:
            done = switch_off(port, cooling)
            break
        except Exception as e:                    # порт ещё не отпущен системой
            err = e
            time.sleep(0.3)
    else:
        L.event('SAF-001', f"Could not open {port}: {err}. Switch the laser off by hand or run "
                           "python -m microcontrol stop.", logging.ERROR)
    for d in done:
        L.event('SAF-001', d)
    if lock is not None:
        lock.release()
    return 0


if __name__ == "__main__":
    if sys.path and os.path.abspath(sys.path[0] or ".") == os.path.dirname(os.path.abspath(__file__)):
        sys.path.pop(0)                       # папка пакета не заслоняет стандартные модули (log, profile ...)
    sys.exit(main(sys.argv[1:]))
