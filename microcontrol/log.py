"""Журнал: logs/MC_ГГГГММДД_ЧЧММСС.log, одна строка на событие, по времени.

    2026-11-20 01:22:46.290  INFO   MOV-001  Moved by dX +200.0, dY 0.0, dZ 0.0 um. Now X 200.0, Y 0.0, Z 0.0 um.
    2026-11-20 01:22:46.291  DEBUG  HW-TX    MOTOR_X  AA 55 00 07 2E 00 00 24 59
    2026-11-20 01:22:46.300  ERROR  E201     Step too large: dX 2000 um, the limit is 1000 um.

Код события — в EVENTS, коды ошибок — errors.CODES.
"""
import logging, os, platform, sys
from datetime import datetime

EVENTS = {
    'SYS-001': "Session start",    'SYS-002': "Session end",
    'CON-001': "Connect",          'CON-002': "Disconnect",        'CON-003': "Identify",
    'MOV-001': "Move",             'MOV-002': "Go to",             'MOV-003': "Offset",
    'MOV-004': "Homing",           'MOV-005': "Objective",         'MOV-006': "Motor step",
    'LGT-001': "Light",            'LSR-001': "Laser",
    'CAM-001': "Picture",          'CAM-002': "Camera setting",    'CAM-003': "Reference frame",
    'AF-001':  "Autofocus",        'AF-002':  "Focus plane",     'AF-003': "Focus at limit", 'AF-004': "Focus window moved",
    'MAN-001': "Manual start",     'MAN-002': "Manual end",        'MAN-004': "Manual step", 'MAN-005': "Cross marked", 'MAN-006': "Focus point", 'CEL-001': "Cells scan", 'STI-001': "Stitched", 'STI-002': "Stitch engine", 'MSK-001': "Export", 'MSK-002': "Clean", 'CEL-002': "Cells join", 'MAN-007': "Cross label", 'ORI-001': "Orientation", 'MAN-003': "Area corner",
    'GC-001':  "G-code start",     'GC-002':  "G-code line",       'GC-003':  "G-code end",
    'SCN-001': "Scan frame",       'SCN-002': "Refocus", 'SCN-003': "Stage still moving",
    'JOI-001': "Join",             'STO-001': "Batch closed", 'STO-005': "JPG not saved",      'SET-001': "Setting changed",
    'MOV-007': "Speed measured",   'MOV-008': "Position confirmed",
    'MOV-012': "Limit code while leaving", 'MOV-013': "Resend, still moving", 'MOV-011': "Motors on",        'CON-005': "Cooling",           'CON-006': "Reconnect",
    'MOV-009': "Home set",         'MOV-010': "Go home",           'CON-004': "Position restored",
    'STP-001': "Emergency stop",   'LCK-001': "Control lock",
    'HW-TX':   "Bytes sent",       'HW-RX':   "Bytes received",
}

logger = logging.getLogger("microcontrol")
logger.setLevel(logging.DEBUG)
logger.propagate = False
_file = None


def _ascii(s):
    return s.replace("…", "...").replace("µ", "u").replace("—", "-").replace("×", "x")


class _Fmt(logging.Formatter):
    def format(self, r):
        t = datetime.fromtimestamp(r.created).strftime("%Y-%m-%d %H:%M:%S.") + f"{int(r.msecs):03d}"
        return f"{t}  {r.levelname:<5}  {getattr(r, 'code', '-'):<7}  {_ascii(r.getMessage())}"


def start(folder="logs", level="DEBUG"):
    """Открыть новый файл журнала. Возвращает путь."""
    global _file
    os.makedirs(folder, exist_ok=True)
    for h in list(logger.handlers):
        logger.removeHandler(h); h.close()
    _file = os.path.abspath(os.path.join(folder, datetime.now().strftime("MC_%Y%m%d_%H%M%S.log")))
    h = logging.FileHandler(_file, encoding="utf-8")
    h.setFormatter(_Fmt())
    h.setLevel(level)
    logger.addHandler(h)
    return _file


def event(code, msg="", level=logging.INFO):
    logger.log(level, msg or EVENTS.get(code, ""), extra=dict(code=code))


def debug(code, msg):
    event(code, msg, logging.DEBUG)


def error(code, msg):
    event(code, msg, logging.ERROR)


def header(**info):
    from . import __version__, AUTHORS
    event('SYS-001', f"MicroControl {__version__}. Authors: Lyapeikov A., Domenko S. "
                     f"Python {sys.version.split()[0]}, {platform.platform()}. " +
                     ", ".join(f"{k} {v}" for k, v in info.items()) + ".")


def path():
    return _file
