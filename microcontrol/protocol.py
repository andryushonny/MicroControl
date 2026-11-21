"""Протокол контроллера ATR8300: AA 55 | LEN(2, BE) | CMD | DATA | SUM8.
LEN = 4 + len(DATA); SUM8 = (LEN_hi + LEN_lo + CMD + sum(DATA)) & 0xFF.
Ответ: тот же формат, CMD = эхо, DATA[0] = статус (0 ок, 1 предел хода)."""
from .errors import CommError

HEAD = b"\xAA\x55"
GET_PN, GET_SN = 0x03, 0x04
LASER = 0x20            # DATA = мощность мВт (2 байта BE), 0 = выкл
MOTOR = {'x': 0x2E, 'y': 0x2F, 'z': 0x2B}   # DATA = [dir, n_hi, n_lo]
LIGHT = 0x2D            # DATA = уровень (2 байта BE), 0 = выкл
GET_MAX_LASER = 0x66
GET_DEVICE_TYPE = 0x26
GET_CALIBRATION = 0x31
TURN_MOTOR = 0x24       # DATA = [1] включить / [0] выключить моторы X, Y (штатное ПО: перед первым движением X/Y)
SET_TEC = 0x12          # DATA = [00 или FF (знак), |t| °C] — охлаждение ПЗС спектрометра (вентилятор)
GET_TEC = 0x13
NAMES = {GET_PN: 'GET_PN', GET_SN: 'GET_SN', LASER: 'SET_LASER', LIGHT: 'SET_LIGHT',
         GET_MAX_LASER: 'GET_MAX_LASER', GET_DEVICE_TYPE: 'GET_TYPE', GET_CALIBRATION: 'GET_CALIB',
         0x2E: 'MOTOR_X', 0x2F: 'MOTOR_Y', 0x2B: 'MOTOR_Z', TURN_MOTOR: 'TURN_MOTOR', SET_TEC: 'SET_TEC',
         GET_TEC: 'GET_TEC'}
# Ответ на движение: DATA[0] — как перечисления MotorXResult/MotorYResult/MotorZResult в OptoSky.dll
MOTOR_OK, MOTOR_LIMIT_A, MOTOR_LIMIT_B, MOTOR_FAIL, MOTOR_MOVING = 0, 1, 2, 3, 4
# Какой концевик: X 1 = вперёд (байт 00), Y 1 = влево (00), Z 1 = вверх (FF). В шагах программы: код 1 — сторона +, код 2 — сторона −.
MOTOR_STATUS = {0: 'success', 1: 'limit switch (forward / left / top)', 2: 'limit switch (backward / right / bottom)',
                3: 'fail', 4: 'still moving'}
COOL_C, WARM_C = -5, 20  # штатное ПО: при подключении CoolTemperature = -5 (usercfg.ini), при отключении 20


def tec_data(t):
    """Данные команды SET_TEC, как в штатном ПО (Protocol.SetTecTEMP)."""
    return bytes([0xFF if t < 0 else 0x00, abs(int(t)) & 0xFF])


# Подключение — как в штатном ПО (DeviceService.Initialize): запросы на чтение, свет 5, охлаждение -5 °C.
# Отключение (DeviceService, ветка закрытия): лазер 0, моторы выкл, свет 0, охлаждение 20 °C.
# (команда, данные): калибровка запрашивается с индексом 9, как в штатном ПО (GetCalibrationDataInteger(9));
# без этого байта контроллер ждёт его и не отвечает.
HANDSHAKE = ((GET_SN, b''), (GET_PN, b''), (GET_MAX_LASER, b''), (GET_CALIBRATION, b'\x09'), (GET_DEVICE_TYPE, b''))
DEVICE_TYPES = {0: 'AutoFocus', 1: 'Basic', 2: 'Mapping'}
# Только эти команды вообще могут уйти в прибор (калибровки, тип устройства,
# старт спектра и т.п. заблокированы — их запись может испортить прибор).
ALLOWED = set(NAMES)


def pack(cmd: int, data: bytes = b"") -> bytes:
    """Собрать кадр команды."""
    body = (4 + len(data)).to_bytes(2, "big") + bytes([cmd]) + bytes(data)
    return HEAD + body + bytes([sum(body) & 0xFF])


def _read_exact(ser, n, deadline):
    """Прочитать n байт к сроку deadline. Не полагается на тайм-аут драйвера порта (старый драйвер STM32
    его не соблюдает и висит минутами): опрашивает, сколько байт пришло, и читает только их."""
    import time
    buf = b""
    while len(buf) < n:
        k = ser.in_waiting
        if k:
            buf += ser.read(min(k, n - len(buf)))
            continue
        if time.monotonic() > deadline:
            break
        time.sleep(0.001)
    return buf


def read_frame(ser, timeout: float):
    """Прочитать один ответ: вернуть (cmd, data, raw). Проверяет заголовок и сумму."""
    import time
    deadline = time.monotonic() + timeout
    head = _read_exact(ser, 4, deadline)
    if len(head) < 4 or head[:2] != HEAD:
        raise CommError('E301' if not head else 'E302', f"got: {head.hex(' ') or 'nothing'}")
    n = int.from_bytes(head[2:4], "big")
    if not 4 <= n <= 8200:
        raise CommError('E302', f"length {n}")
    rest = _read_exact(ser, n - 2, deadline + 1.0)
    if len(rest) != n - 2:
        raise CommError('E302', "truncated reply")
    if sum(head[2:] + rest[:-1]) & 0xFF != rest[-1]:
        raise CommError('E303', (head + rest).hex(' '))
    return rest[0], bytes(rest[1:-1]), head + rest
