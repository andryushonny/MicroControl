"""Командная строка MicroControl.

  python -m microcontrol stop      аварийная остановка (из любого окна cmd, даже если ноутбук завис)
  python -m microcontrol status    кто сейчас управляет прибором
  python -m microcontrol holders   какие процессы могут держать порт (MicroControl, Jupyter, ПО Optosky)
  python -m microcontrol probe     проверка связи с контроллером: только чтение, без камеры и движения
  python -m microcontrol serve     сервер у прибора: программа на другом компьютере — Microscope('net://адрес')
                                   (--port COM5|auto|sim, --listen 0.0.0.0:5150, --no-camera, --dead 10)
  python -m microcontrol stop net://адрес     аварийная остановка прибора за сервером
  python -m microcontrol status net://адрес   сервер: что включено, кто подключён
  python -m microcontrol find      серверы MicroControl в локальной сети (адрес можно не знать: net://auto)
  python -m microcontrol mcp [net://auto | COM3 | sim | (ничего: только сканы)] [--objective 10]
                                   MCP-сервер (stdio) для ИИ-агента: Claude Desktop, DeepSeek через любой MCP-клиент
"""
import sys
from .control import emergency_stop, ControlLock, DIR
import os


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'
    if cmd == 'serve':
        from .remote import serve
        serve(sys.argv[2:])
        return
    if cmd == 'find':
        from .remote import find_servers
        got = find_servers(2.0)
        if not got:
            print("No MicroControl server answered in this network.\n  On the instrument PC: python -m microcontrol "
                  "serve; both PCs on one network (Wi-Fi 'Private' in Windows); port 5150 open in its firewall.")
        for g in got:
            print(f"net://{g['host']}:{g['port']}   {g.get('name')}   controller {g.get('controller')}   camera "
                  f"{g.get('camera') or 'none'}   v{g.get('version')}" + (f"   busy: {g['busy']}" if g.get('busy') else "   free"))
        return
    if cmd == 'mcp':
        from .agent import main as _am
        rest = sys.argv[2:]
        dev = [a for a in rest if a.startswith(('net://', 'COM', '/dev/')) or a in ('sim', 'auto')][:1]
        _am(['--mcp'] + (['--port', dev[0]] if dev else []) + [a for a in rest if a not in dev])
        return
    if cmd in ('stop', 'status') and len(sys.argv) > 2 and sys.argv[2].startswith('net://'):
        from . import remote as R
        if cmd == 'stop':
            print("MicroControl: EMERGENCY STOP")
            print("  " + R.emergency_stop(sys.argv[2]))
        else:
            import json
            print(json.dumps(R.status(sys.argv[2]), indent=1, ensure_ascii=False))
        return
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
