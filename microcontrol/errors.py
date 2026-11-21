"""Ошибки MicroControl. Текст на английском, у каждой есть код: str(e) -> "[E201] Step too large: ..."."""

CODES = {
    'E101': "Controller port not found",
    'E102': "Port is busy or unavailable",
    'E103': "Port is not an ATR8300 controller",
    'E104': "Microscope is closed",
    'E105': "Unknown setting",
    'E106': "The microscope is already controlled by another program",
    'E107': "The device does not answer like an ATR8300",
    'E108': "Port does not open",
    'E110': "Emergency stop",
    'E201': "Step too large",
    'E202': "Target outside the allowed travel",
    'E203': "Z above the ceiling (objective collision risk)",
    'E204': "Limit switch hit",
    'E205': "Direction blocked by a limit switch",
    'E206': "Invalid number",
    'E207': "Unknown objective",
    'E208': "Invalid setting value",
    'E209': "Homing failed or locked",
    'E210': "Suspicious move",
    'E211': "Position is uncertain",
    'E212': "No motion seen",
    'E213': "Nothing to do",
    'E214': "Move smaller than one motor step",
    'E215': "Home is already set",
    'E216': "Home is not set",
    'E301': "No reply from the controller",
    'E302': "Malformed reply",
    'E303': "Checksum error",
    'E304': "Reply to another command",
    'E306': "Motor command not done",
    'E307': "Connection to the controller lost",
    'E305': "Command not allowed",
    'E401': "Laser is locked",
    'E402': "Laser power above the limit",
    'E501': "Camera not found",
    'E502': "Camera SDK (MVCAMSDK_X64.dll) not found",
    'E503': "Camera not connected",
    'E505': "Camera does not answer",
    'E601': "No room for autofocus",
    'E602': "Unknown focus metric",
    'E603': "No clear focus anywhere in the area",
    'E701': "Scan area not set",
    'E702': "Not enough disk space",
    'E703': "Invalid scan parameters",
    'E801': "Scan folder not found or empty",
    'E901': "G-code syntax error",
    'E902': "Unsupported G-code command",
}


class MicroControlError(Exception):
    """Базовая ошибка: .code (E101...) и сообщение на английском. Пишется в журнал."""
    def __init__(self, code, detail=""):
        self.code, self.detail = code, detail
        msg = CODES.get(code, "Error") + (f": {detail}" if detail else ".")
        super().__init__(f"[{code}] {msg}")
        from .log import error
        error(code, msg)


class SafetyError(MicroControlError, ValueError):
    """Команда отклонена защитой."""


class LimitError(MicroControlError, RuntimeError):
    """Столик упёрся в концевик."""


class CommError(MicroControlError, IOError):
    """Нет связи с контроллером."""


class CameraError(MicroControlError, RuntimeError):
    """Проблема с камерой."""


class FocusError(MicroControlError, RuntimeError):
    """Проблема автофокуса."""


class StoreError(MicroControlError, RuntimeError):
    """Проблема с папкой скана."""


class StopError(MicroControlError, RuntimeError):
    """Аварийная остановка: python -m microcontrol stop. После неё нужно подключиться заново."""


class GcodeError(MicroControlError, ValueError):
    """Ошибка в G-коде (программа не запускается, пока ошибок нет)."""
