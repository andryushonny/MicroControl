"""Проверка связи с контроллером без камеры и без движения: python -m microcontrol probe [COM3].

Отправляет только запросы на чтение — те же, что штатное ПО при подключении и печатает все байты туда и обратно
с временем ответа. Сначала — как штатное ПО (линии DTR и RTS выключены), если ответа нет — с включёнными.
Ничего не включает и не двигает.
"""
import time
from . import protocol as P
from .config import VID, PID


def _hex(b):
    return b.hex(' ').upper() if b else '(nothing)'


def run(port='auto'):
    import serial
    from serial.tools import list_ports
    out = []
    say = lambda s='': (print(s, flush=True), out.append(s))
    found = list_ports.comports()
    say("Ports:")
    for p in found:
        mark = "  <- ATR8300 controller (0483:5740)" if (p.vid, p.pid) == (VID, PID) else ""
        say(f"  {p.device:<7} VID:PID {p.vid or 0:04X}:{p.pid or 0:04X}  {p.description}{mark}")
    if port == 'auto':
        atr = [p.device for p in found if (p.vid, p.pid) == (VID, PID)]
        if not atr:
            say("No ATR8300 controller found. Check the USB cable and the driver (STM32 Virtual COM Port).")
            return out
        port = atr[0]
    for dtr in (False, True):
        say()
        say(f"{port}, 115200 8N1, DTR/RTS {'ON' if dtr else 'OFF (as the Optosky software)'}:")
        try:
            ser = serial.Serial(baudrate=115200, timeout=1.0, write_timeout=2)
            ser.dtr = ser.rts = dtr
            ser.port = port
            ser.open()
        except Exception as e:
            say(f"  Cannot open: {e}")
            say("  The port is busy: close the Optosky software and other notebooks "
                "(python -m microcontrol status).")
            return out
        answered = 0
        try:
            time.sleep(0.2)
            junk = ser.read(ser.in_waiting or 0)
            if junk:
                say(f"  Bytes before any request: {_hex(junk)}")
            for cmd, data in P.HANDSHAKE:
                frame = P.pack(cmd, data)
                ser.reset_input_buffer()
                t0 = time.perf_counter()
                ser.write(frame)
                dl = time.monotonic() + 1.0                 # свой срок: старый драйвер не соблюдает тайм-аут
                reply = P._read_exact(ser, 4, dl)
                if len(reply) == 4 and reply[:2] == P.HEAD:
                    n = int.from_bytes(reply[2:4], 'big')
                    reply += P._read_exact(ser, max(0, n - 2), dl + 1.0)
                else:
                    reply += P._read_exact(ser, 256, time.monotonic() + 0.3)
                ms = (time.perf_counter() - t0) * 1000
                say(f"  -> {P.NAMES[cmd]:<9} {_hex(frame)}")
                say(f"  <- {ms:7.1f} ms {_hex(reply)}")
                if reply[:2] == P.HEAD:
                    answered += 1
                    data = reply[5:-1]
                    if data and all(32 <= c < 127 or c == 0 for c in data):
                        say(f"     text: {data.decode('ascii', 'replace').strip(chr(0))!r}")
        finally:
            ser.close()
        if answered:
            say(f"  The controller answers ({answered} of {len(P.HANDSHAKE)}).")
            break
        say("  No answer.")
    return out
