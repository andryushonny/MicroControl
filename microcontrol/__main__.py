"""Командная строка MicroControl.

  python -m microcontrol stop      аварийная остановка (из любого окна cmd, даже если ноутбук завис)
  python -m microcontrol status    кто сейчас управляет прибором
  python -m microcontrol holders   какие процессы могут держать порт (MicroControl, Jupyter, ПО Optosky)
  python -m microcontrol probe     проверка связи с контроллером: только чтение, без камеры и движения
"""
import sys
from .control import emergency_stop, ControlLock, DIR
import os


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'
    if cmd == 'stop':
        print("MicroControl: EMERGENCY STOP")
        lines = emergency_stop(sys.argv[2] if len(sys.argv) > 2 else 'auto')
        for line in lines:
            print("  " + line)
        if any("could NOT" in l or "still busy" in l for l in lines):
            sys.exit(2)                       # не хватило прав: повторить в консоли администратора
    elif cmd == 'holders':
        from .control import port_holders
        rows = port_holders()
        print("\n".join(f"PID {pid}  {name}  ({why})" for pid, name, why in rows) or "Nobody else holds the port.")
    elif cmd == 'status':
        locks = [f for f in os.listdir(DIR) if f.endswith('.lock')] if os.path.isdir(DIR) else []
        if not locks:
            print("Nobody controls the microscope.")
        for f in locks:
            port = f[8:-5]
            own = ControlLock.owner(port)
            try:
                ControlLock(port).release()
                print(f"{port}: free.")
            except OSError:
                print(f"{port}: controlled by PID {own.get('pid')} since {own.get('since')} ({own.get('program')}).")
    elif cmd == 'probe':
        from .probe import run
        run(sys.argv[2] if len(sys.argv) > 2 else 'auto')
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
