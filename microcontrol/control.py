"""Один хозяин у прибора и аварийная остановка.

Блокировка: при подключении берётся системная блокировка файла ~/.microcontrol/control_<порт>.lock.
Пока она у одной программы (ноутбука), второй подключиться нельзя — ошибка E106 с номером процесса.
Если программа завершилась или зависла и была убита, система снимает блокировку сама.

Аварийная остановка (из любой командной строки): python -m microcontrol stop
  1. создаётся файл ~/.microcontrol/STOP — работающая программа видит его перед каждой командой
     и в паузах (каждые 0.05 с): выключает лазер и свет и прекращает работу (ошибка E110);
  2. если через 3 с программа не отпустила прибор (зависла), её процесс завершается принудительно;
  3. порт открывается заново, лазер и свет выключаются. Если порт занят чужим процессом (Access denied) —
     завершаются все, кто может его держать: процессы MicroControl (Jupyter), штатное ПО Optosky;
     если на это не хватает прав — повторить команду в командной строке администратора.
Мотор дорабатывает уже отправленный кусок движения (не больше 1 мм): команды «стоп» у контроллера нет.
"""
import json, os, sys, time
from datetime import datetime

DIR = os.path.join(os.path.expanduser("~"), ".microcontrol")
STOP_FILE = os.path.join(DIR, "STOP")


def _lock_path(port):
    return os.path.join(DIR, f"control_{str(port).replace(os.sep, '_').replace(':', '')}.lock")


class ControlLock:
    """Системная блокировка: один процесс — один хозяин прибора."""
    def __init__(self, port):
        os.makedirs(DIR, exist_ok=True)
        self.path = _lock_path(port)
        self.f = open(self.path, 'a+')
        try:
            if os.name == 'nt':
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            self.f = None
            raise
        info = dict(pid=os.getpid(), since=datetime.now().isoformat(timespec='seconds'),
                    program=os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else '', folder=os.getcwd())
        self.f.seek(0)
        self.f.truncate()
        self.f.write(json.dumps(info))
        self.f.flush()

    @staticmethod
    def owner(port):
        """Кто держит прибор: {'pid', 'since', ...} или None."""
        try:
            with open(_lock_path(port)) as f:
                return json.loads(f.read() or 'null')
        except (OSError, ValueError):
            return None

    def release(self):
        if self.f:
            try:
                if os.name == 'nt':
                    import msvcrt
                    self.f.seek(0)
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
            finally:
                self.f.close()
                self.f = None


def stop_requested_after(t):
    """True, если аварийная остановка запрошена после момента t (time.time())."""
    try:
        return os.path.getmtime(STOP_FILE) >= t
    except OSError:
        return False


def _is_locked(port):
    try:
        l = ControlLock(port)
    except OSError:
        return True
    l.release()
    return False


VENDOR_EXE = ('optoskyramanmicroscope', 'ramanmicroscope', 'optosky')    # штатное ПО Optosky


def _alive(pid):
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except ImportError:
        pass
    if os.name == 'nt':
        import subprocess
        r = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"], capture_output=True, text=True)
        return str(int(pid)) in r.stdout
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _kill(pid):
    """Завершить процесс (с дочерними). True — процесса больше нет; False — не дали (нужны права администратора)."""
    pid = int(pid)
    if os.name == 'nt':
        import subprocess
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        import signal
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    for _ in range(20):
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


def _mine():
    """Этот процесс и его родители (их не трогаем: это само окно остановки)."""
    ids = {os.getpid()}
    try:
        import psutil
        ids |= {p.pid for p in psutil.Process().parents()}
    except Exception:
        ids.add(os.getppid())
    return ids


def port_holders():
    """Кто может держать порт прибора: [(pid, имя, причина)].
    1) процессы MicroControl — у них открыт журнал logs/MC_*.log или файл блокировки (нужен psutil);
    2) штатное ПО Optosky (OptoskyRamanMicroscope.exe, RamanMicroscope.exe);
    3) без psutil — процессы Python, в командной строке которых microcontrol, ipykernel или jupyter."""
    out, skip = [], _mine()
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil:
        for p in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
            if p.pid in skip:
                continue
            name = (p.info['name'] or '').lower()
            if any(v in name for v in VENDOR_EXE):
                out.append((p.pid, p.info['name'], "Optosky software"))
                continue
            if 'python' not in name:
                continue
            why = None
            try:
                for f in p.open_files():
                    fp = f.path.lower().replace('\\', '/')
                    if '/logs/mc_' in fp or '.microcontrol' in fp:
                        why = "MicroControl (open log / lock file)"
                        break
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
            if why is None:
                cl = ' '.join(p.info['cmdline'] or []).lower()
                if 'microcontrol' in cl and ' stop' not in cl:
                    why = "MicroControl (command line)"
            if why:
                out.append((p.pid, p.info['name'], why))
        return out
    if os.name == 'nt':                                   # без psutil: список процессов через PowerShell
        import subprocess, json as _j
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Get-CimInstance Win32_Process | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json"],
                           capture_output=True, text=True)
        try:
            rows = _j.loads(r.stdout or '[]')
        except ValueError:
            rows = []
        for row in rows if isinstance(rows, list) else [rows]:
            pid, name, cl = row.get('ProcessId'), (row.get('Name') or ''), (row.get('CommandLine') or '').lower()
            if pid in skip:
                continue
            if any(v in name.lower() for v in VENDOR_EXE):
                out.append((pid, name, "Optosky software"))
            elif 'python' in name.lower() and any(k in cl for k in ('microcontrol', 'ipykernel', 'jupyter')) \
                    and ' stop' not in cl:
                out.append((pid, name, "Python with MicroControl / Jupyter"))
    return out


def _open_port(p):
    import serial
    return serial.Serial(p, 115200, timeout=1.0)


def free_port(p, done):
    """Порт занят (Access denied): завершить всех, кто может его держать. True — порт открылся."""
    denied = []
    for pid, name, why in port_holders():
        if _kill(pid):
            done.append(f"Process {name} PID {pid} ({why}) was terminated.")
        else:
            denied.append(pid)
            done.append(f"Process {name} PID {pid} ({why}) could NOT be terminated: no rights.")
    time.sleep(0.5)
    try:
        _open_port(p).close()
        return True
    except Exception as e:
        done.append(f"Port {p} is still busy: {e}")
        if denied or os.name == 'nt':
            done.append("Run 'python -m microcontrol stop' again in an administrator console, "
                        "or close the program that holds the port.")
        return False


def emergency_stop(port='auto', wait=3.0):
    """Аварийная остановка (см. описание модуля). Возвращает список сделанного."""
    os.makedirs(DIR, exist_ok=True)
    with open(STOP_FILE, 'w') as f:
        f.write(datetime.now().isoformat(timespec='seconds'))
    done = ["Stop signal sent."]
    from .config import VID, PID
    ports = []
    if port == 'auto':
        try:
            from serial.tools import list_ports
            ports = [p.device for p in list_ports.comports() if (p.vid, p.pid) == (VID, PID)]
        except ImportError:
            pass
        ports += [f[8:-5] for f in os.listdir(DIR) if f.startswith('control_') and f.endswith('.lock')]
        ports = list(dict.fromkeys(ports))
    else:
        ports = [port]
    for p in ports:
        own = ControlLock.owner(p)
        t_end = time.time() + wait
        while _is_locked(p) and time.time() < t_end:
            time.sleep(0.1)
        if _is_locked(p):
            if own and own.get('pid') and own['pid'] != os.getpid():
                ok = _kill(own['pid'])
                done.append(f"Program PID {own['pid']} ({own.get('program', '')}) did not stop and "
                            + ("was terminated." if ok else "could NOT be terminated (no rights)."))
                time.sleep(0.5)
        elif own:
            done.append(f"Program PID {own.get('pid')} stopped by itself.")
        if p != 'sim':
            try:
                _open_port(p).close()
            except Exception as e:
                if 'denied' in str(e).lower() or 'отказано' in str(e).lower() or 'PermissionError' in repr(e) \
                        or 'busy' in str(e).lower():
                    done.append(f"Port {p} is held by another process: {e}")
                    free_port(p, done)
            try:
                from . import protocol as P
                with _open_port(p) as s:
                    for cmd, name in ((P.LASER, "Laser"), (P.LIGHT, "Light")):
                        s.reset_input_buffer()
                        s.write(P.pack(cmd, b"\x00\x00"))
                        try:
                            P.read_frame(s, 1.0)
                            done.append(f"{name} is off on {p}.")
                        except Exception:
                            done.append(f"{name}: no answer on {p}.")
            except Exception as e:
                done.append(f"Port {p}: {e}")
    done.append("Reconnect with mc.Microscope(...) when it is safe. An old stop signal does not block new connections.")
    return done
