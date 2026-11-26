"""Тонкий сервер у прибора: микроскоп и камера на одном компьютере, программа (Jupyter) — на другом.

    у прибора:   python -m microcontrol serve                      (порт контроллера сам, камера, порт сети 5150)
    в Jupyter:   m = mc.Microscope('net://192.168.50.2', objective=10)

Сервер держит порт контроллера и камеру и выполняет то, что просит программа: команда контроллеру — одна
транзакция «кадр туда, ответ обратно» (один сетевой круг на команду), кадр камеры — без сжатия по гигабитной сети,
PNG (без потерь) по сети похуже, JPEG 95 по медленной (выбор — по замеру при подключении; MC_NET_CODEC=raw|png|jpg95). Считает всё программа; старому компьютеру нечего считать.

Безопасность — на стороне прибора, а не программы:
  * программа шлёт «я жив» раз в секунду; нет вестей dead_s секунд (сеть упала, компьютер завис) или связь
    оборвалась без штатного close() — сервер сам выключает лазер, моторы X, Y, свет и (если включали) охлаждение;
  * python -m microcontrol stop net://адрес — аварийная остановка по сети (без ключа): то же выключение и
    отключение программы;
  * хозяин один: вторая программа не подключится (E106), пока первая не закрылась;
  * ключ: сервер создаёт его при первом запуске (~/.microcontrol/server_token) и печатает; программа
    передаёт его один раз (net://КЛЮЧ@адрес или MC_TOKEN) и запоминает. Ключ по сети не ходит (HMAC);
    сам обмен не шифруется — для интернета нужен VPN (Tailscale, WireGuard);
  * сервер пропускает только разрешённые команды (protocol.ALLOWED), как и сама программа;
  * если упадёт сам сервер — его сторож (guard.py) выключит прибор, как у обычной программы.
Память прибора (положение столика, дом, последний кадр) живёт на сервере — общая для программ на любом
компьютере.
"""
import hashlib, hmac, json, os, secrets, socket, struct, threading, time

DEFAULT_PORT = 5150
PROTO = 1
DIR = os.path.join(os.path.expanduser("~"), ".microcontrol")


# ---------------------------------------------------------------- обмен сообщениями
def _send(sock, hdr, blob=b""):
    h = json.dumps(hdr).encode()
    sock.sendall(struct.pack(">II", len(h), len(blob)) + h + bytes(blob))


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        b = sock.recv(min(n - len(buf), 1 << 20))
        if not b:
            raise ConnectionError("connection closed")
        buf += b
    return bytes(buf)


def _recv(sock):
    nh, nb = struct.unpack(">II", _recv_exact(sock, 8))
    if nh > 1 << 20 or nb > 1 << 28:
        raise ConnectionError("bad message")
    return json.loads(_recv_exact(sock, nh)), (_recv_exact(sock, nb) if nb else b"")


def _mac(token, nonce):
    return hmac.new(token.encode(), bytes.fromhex(nonce), hashlib.sha256).hexdigest()


def parse_url(url):
    """net://[ключ@]адрес[:порт] -> (адрес, порт, ключ или None)."""
    rest = url[len("net://"):] if url.startswith("net://") else url
    token = None
    if "@" in rest:
        token, rest = rest.rsplit("@", 1)
    rest = rest.strip("/")
    if rest.startswith("["):                                  # [IPv6]:порт
        host, _, p = rest[1:].partition("]")
        port = int(p[1:]) if p.startswith(":") else DEFAULT_PORT
    elif rest.count(":") == 1:
        host, p = rest.split(":")
        port = int(p)
    else:
        host, port = rest, DEFAULT_PORT
    return host, port, token or None


DISCOVER = b"MICROCONTROL?"                                  # поиск серверов в сети: UDP на тот же порт


def find_servers(timeout=1.0, port=DEFAULT_PORT):
    """Серверы MicroControl в локальной сети (широковещательный UDP; Wi-Fi тоже). -> [dict(host, port, name,
    controller, camera, busy, version)] — без ключа: ключ всё равно нужен для подключения."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.settimeout(0.1)
    targets = {"255.255.255.255", "127.0.0.1"}                # и этот же компьютер
    for ip in _my_ips():                                   # и широковещательный адрес каждой сети (/24)
        if not ip.startswith("127."):
            targets.add(ip.rsplit(".", 1)[0] + ".255")
    found, t_end, t_send = {}, time.monotonic() + timeout, 0.0
    try:
        while time.monotonic() < t_end:
            if time.monotonic() - t_send > 0.3:             # UDP по Wi-Fi теряется: спросить ещё раз
                t_send = time.monotonic()
                for t in targets:
                    try:
                        s.sendto(DISCOVER, (t, port))
                    except OSError:
                        pass
            try:
                data, (ip, _p) = s.recvfrom(4096)
            except (socket.timeout, OSError):
                continue
            try:
                d = json.loads(data.decode("utf-8"))
            except ValueError:
                continue
            if d.get("hello") == "microcontrol":                # один сервер — одна запись (ответил по двум
                k = d.get("id") or (ip, d.get("port"))           # адресам: сеть важнее 127.0.0.1)
                if k not in found or found[k]["host"].startswith("127."):
                    found[k] = dict(host=ip, **{k2: v for k2, v in d.items() if k2 not in ("hello", "id")})
    finally:
        s.close()
    return sorted(found.values(), key=lambda d: (d["host"], d["port"]))


def resolve(url, timeout=1.5):
    """net://[ключ@]auto[:порт] -> тот же адрес с найденным сервером (один в сети — он; несколько — какой)."""
    host, port, token = parse_url(url)
    if host not in ("auto", "*"):
        return url
    from .errors import CommError
    got = find_servers(timeout, port)
    if len(got) != 1:
        lst = "; ".join(f"net://{g['host']}:{g['port']} ({g.get('name')}, {g.get('controller')})" for g in got)
        raise CommError("E112", ("No MicroControl server answered in this network (is 'python -m microcontrol serve' "
                                 "running on the instrument PC, both on one network, port "
                                 f"{port} open in its firewall?)") if not got else
                        f"Several servers in the network, give one: {lst}")
    g = got[0]
    return f"net://{token + '@' if token else ''}{g['host']}:{g['port']}"


def server_token():
    """Ключ сервера (создаётся при первом запуске)."""
    p = os.path.join(DIR, "server_token")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            t = f.read().strip()
        if t:
            return t
    os.makedirs(DIR, exist_ok=True)
    t = secrets.token_hex(8)
    with open(p, "w", encoding="utf-8") as f:
        f.write(t)
    return t


def _saved_tokens():
    p = os.path.join(DIR, "net_tokens.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_token(key, token):
    d = _saved_tokens()
    if d.get(key) != token:
        d[key] = token
        os.makedirs(DIR, exist_ok=True)
        with open(os.path.join(DIR, "net_tokens.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)


# ---------------------------------------------------------------- сервер (у прибора)
class Server:
    """Сервер прибора. port: 'auto', 'COM5' или 'sim' (симулятор — для проверки по сети)."""

    def __init__(self, port="auto", listen=("0.0.0.0", DEFAULT_PORT), camera=True, sdk_folder=None, dead_s=10.0,
                 token=None, logs=None, verbose=True, guard=True):
        from . import log as L
        from . import protocol as P
        self.L, self.P, self.verbose = L, P, verbose
        self.dead_s = float(dead_s)
        self.token = token or server_token()
        L.start(logs or os.path.join(DIR, "server_logs"), suffix="_server")
        self.dev = threading.RLock()                 # порт контроллера: по одной транзакции
        self.camlock = threading.RLock()
        self.owner = None                            # сеанс хозяина: dict(sid, who, last, socks, done, ...)
        self.ended = {}                              # прошлые сеансы: sid -> сеанс (что выполнено последним)
        self.last_off = None                         # последнее выключение сервером: dict(time, reason)
        self.state = dict(laser=0, light=0, motors=False, cooling=False)
        self._guard, self._lock, self.cam, self.world = None, None, None, None
        self.sim = port == "sim"
        if self.sim:
            from .sim import SimWorld, SimPort, SimCamera
            self.world = SimWorld()
            self.ser = SimPort(self.world)
            self.port = "sim"
            if camera:
                self.cam = SimCamera(self.world)
        else:
            from serial.tools import list_ports
            from .config import VID, PID
            from .control import ControlLock
            from .core import _open_serial
            if port == "auto":
                atr = [p.device for p in list_ports.comports() if (p.vid, p.pid) == (VID, PID)]
                if len(atr) != 1:
                    raise RuntimeError(f"Controller port: found {atr or 'none'}. Give it: serve --port COM5")
                port = atr[0]
            self.port = port
            self._lock = ControlLock(port)           # пока сервер работает, прибор — его
            self.ser = _open_serial(port)
            if guard:
                from . import guard as G
                self._guard = G.start(port, False, logs or os.path.join(DIR, "server_logs"))
            if camera:
                from .camera import MVCamera
                self.cam = MVCamera(sdk_folder)
        self.cam_info = self._cam_info()
        self.sock = socket.create_server(listen, reuse_port=False)
        self.addr = self.sock.getsockname()
        self.udp, self.uid = None, secrets.token_hex(6)
        try:                                          # ответ на поиск (net://auto, python -m microcontrol find)
            self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp.bind((listen[0], self.addr[1]))
            self.udp.settimeout(0.5)
        except OSError:
            self.udp = None
        self._stop = threading.Event()
        self._last_tx = time.monotonic()
        L.event("NET-001", f"Server for {self.port} on {self.addr[0]}:{self.addr[1]}, camera "
                           f"{self.cam_info.get('model') if self.cam else 'none'}, safe off after {self.dead_s:g} s "
                           "without the program.")

    def _cam_info(self):
        if self.cam is None:
            return {}
        try:
            d = dict(self.cam.info())
            img = self.cam.grab(0)
            d["size"] = [int(img.shape[1]), int(img.shape[0])]
            return d
        except Exception as e:
            return dict(error=str(e))

    def say(self, s):
        if self.verbose:
            print(time.strftime("%H:%M:%S ") + s, flush=True)

    # ----- прибор
    def _tx(self, frame, timeout):
        """Кадр контроллеру -> ответ (raw) или CommError."""
        P = self.P
        with self.dev:
            self.ser.reset_input_buffer()
            self.ser.write(frame)
            try:
                _, _, raw = P.read_frame(self.ser, timeout)
            finally:
                self._last_tx = time.monotonic()
            return raw

    def safe_off(self, reason):
        """Выключить лазер, моторы X, Y, свет, охлаждение (если включали)."""
        P = self.P
        done = []
        steps = [(P.LASER, b"\x00\x00", "laser"), (P.TURN_MOTOR, b"\x00", "motors"), (P.LIGHT, b"\x00\x00", "light")]
        if self.state["cooling"]:
            steps.append((P.SET_TEC, P.tec_data(P.WARM_C), "cooling"))
        for cmd, data, name in steps:
            try:
                self._tx(P.pack(cmd, data), 1.0)
                done.append(name)
            except Exception as e:
                done.append(f"{name} (no answer: {e})")
        self.state.update(laser=0, light=0, motors=False, cooling=False)
        self.last_off = dict(time=time.time(), reason=reason)
        self.L.event("NET-003", f"Switched off ({reason}): " + ", ".join(done) + ".", 30)
        self.say(f"SAFE OFF ({reason}): " + ", ".join(done))

    def _track(self, cmd, data):
        """Что включено (для выключения и состояния)."""
        P = self.P
        if cmd == P.LASER:
            self.state["laser"] = int.from_bytes(data[:2], "big") if len(data) >= 2 else 0
        elif cmd == P.LIGHT:
            self.state["light"] = int.from_bytes(data[:2], "big") if len(data) >= 2 else 0
        elif cmd == P.TURN_MOTOR:
            self.state["motors"] = bool(data[:1] == b"\x01")
        elif cmd == P.SET_TEC:
            cool = data == P.tec_data(P.COOL_C)
            if cool != self.state["cooling"] and self._guard is not None:   # сторож сервера — с охлаждением
                from . import guard as G
                self._guard.done()
                self._guard = G.start(self.port, cool, os.path.dirname(self.L.path()))
            self.state["cooling"] = cool

    # ----- сеансы
    def _end(self, sess):
        self.ended[sess["sid"]] = sess
        while len(self.ended) > 50:
            self.ended.pop(next(iter(self.ended)))

    def _kick(self, reason):
        ow, self.owner = self.owner, None
        if ow is None:
            return
        self._end(ow)
        for s in list(ow["socks"]):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.L.event("NET-002", f"Program {ow['who'].get('host')} PID {ow['who'].get('pid')} disconnected: {reason}.")

    def _handle(self, conn, peer):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        role, sess = None, None
        try:
            nonce = secrets.token_hex(16)
            _send(conn, dict(hello="microcontrol", proto=PROTO, nonce=nonce))
            conn.settimeout(10)
            hdr, _ = _recv(conn)
            role = hdr.get("role")
            if role == "stop":                       # аварийная остановка — без ключа
                with self.dev:
                    ow = self.owner
                    self.safe_off(f"remote stop from {peer[0]}")
                    self._kick("emergency stop")
                _send(conn, dict(ok=True, was=ow["who"] if ow else None))
                return
            if not hmac.compare_digest(str(hdr.get("mac", "")), _mac(self.token, nonce)):
                self.L.event("NET-002", f"Wrong key from {peer[0]}.", 30)
                _send(conn, dict(err="E111", msg="wrong key: give the key printed by the server "
                                                 "(net://KEY@address or MC_TOKEN)"))
                return
            if role == "status":
                _send(conn, dict(ok=True, **self._status()))
                return
            with self.dev:
                ow = self.owner
                if role == "ctl":
                    if ow is not None and time.monotonic() - ow["last"] <= self.dead_s:
                        _send(conn, dict(err="E106", msg=f"{ow['who'].get('host')} PID {ow['who'].get('pid')} "
                                                        f"since {ow['since']}"))
                        return
                    if ow is not None:
                        self._kick("replaced by a new program")
                    old = self.ended.get(hdr.get("prev_sid") or "")
                    prev = (dict(seq=old["done"][0], raw=old["done"][1]) if old.get("done") else dict(seq=None)) \
                        if old else None                     # прошлый сеанс этой программы: что выполнено последним
                    sess = dict(sid=secrets.token_hex(8), who=hdr.get("who", {}), last=time.monotonic(),
                                socks={conn}, since=time.strftime("%Y-%m-%d %H:%M:%S"), clean=False, done=None)
                    sess["who"]["ip"] = peer[0]
                    self.owner = sess
                    from .device import DIR as MEM
                    _send(conn, dict(ok=True, sid=sess["sid"], port=self.port, sim=self.sim, camera=self.cam_info,
                                     dead_s=self.dead_s, last_off=self.last_off, mem_dir=MEM, proto=PROTO,
                                     prev=prev))
                    self.L.event("NET-002", f"Program {sess['who'].get('host')} PID {sess['who'].get('pid')} "
                                            f"({peer[0]}) connected.")
                    self.say(f"connected: {sess['who'].get('host')} PID {sess['who'].get('pid')} ({peer[0]})")
                elif role == "cam":
                    if ow is None or hdr.get("sid") != ow["sid"]:
                        _send(conn, dict(err="E106", msg="no session"))
                        return
                    sess = ow
                    sess["socks"].add(conn)
                    _send(conn, dict(ok=True))
                else:
                    _send(conn, dict(err="E111", msg=f"unknown role {role}"))
                    return
            conn.settimeout(None)
            while True:
                hdr, blob = _recv(conn)
                if self.owner is not sess:
                    raise ConnectionError("session ended")
                sess["last"] = time.monotonic()
                op = hdr.get("op")
                try:
                    if op == "bye":
                        sess["clean"] = True
                        _send(conn, dict(ok=True))
                        return
                    rep, out = self._op(op, hdr, blob, sess)
                except Exception as e:
                    code = getattr(e, "code", None)
                    rep, out = dict(err=code or "E307", msg=str(getattr(e, "msg", e)), cls=type(e).__name__), b""
                sess["last"] = time.monotonic()
                _send(conn, rep, out)
        except (ConnectionError, OSError, ValueError, struct.error):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass
            if role == "ctl" and sess is not None:
                with self.dev:
                    if self.owner is sess:
                        if not sess["clean"]:
                            self.safe_off("connection to the program lost")
                        else:
                            self.L.event("NET-002", "Program closed the microscope.")
                            self.say("closed by the program")
                        self.owner = None
                        self._end(sess)
                        for s in list(sess["socks"]):
                            try:
                                s.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass

    def _status(self):
        ow = self.owner
        return dict(port=self.port, sim=self.sim, state=dict(self.state), last_off=self.last_off,
                    owner=dict(ow["who"], since=ow["since"]) if ow else None)

    def _op(self, op, hdr, blob, sess=None):
        P = self.P
        if op == "ping":
            return dict(ok=True), b""
        if op == "tx":
            cmd, data = blob[4], blob[5:-1]
            if blob[:2] != P.HEAD or cmd not in P.ALLOWED or P.pack(cmd, data) != blob:
                raise PermissionError(f"command 0x{cmd:02X} is not allowed")
            if cmd == P.SET_TEC and data not in (P.tec_data(P.COOL_C), P.tec_data(P.WARM_C)):
                raise PermissionError("cooling only as the Optosky software")
            with self.dev:                           # выполнено — запомнено сразу (переподключение узнает)
                raw = self._tx(blob, float(hdr.get("timeout", 1.0)))
                self._track(cmd, data)
                if sess is not None and "seq" in hdr:
                    sess["done"] = (int(hdr["seq"]), raw.hex())
            return dict(ok=True), raw
        if op == "status":
            return dict(ok=True, **self._status()), b""
        if op == "bench":
            return dict(ok=True), os.urandom(min(int(hdr.get("n", 1 << 20)), 1 << 24))
        if op == "mem_get":
            return self._mem_get(hdr.get("prefix", ""))
        if op == "mem_put":
            return self._mem_put(hdr.get("name", ""), blob)
        if self.cam is None and op in ("grab", "set", "freeze", "info"):
            raise RuntimeError("no camera on the server")
        if op == "grab":
            import cv2
            with self.camlock:
                img = self.cam.grab(int(hdr.get("flush", 2)))
            codec = hdr.get("codec", "png")
            if codec == "raw":
                return dict(ok=True, shape=list(img.shape), dtype=str(img.dtype)), img.tobytes()
            if codec.startswith("jpg"):
                ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(codec[3:] or 95)])
            else:
                ok, b = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            return dict(ok=True), b.tobytes()
        if op == "set":
            with self.camlock:
                kw = {k: hdr[k] for k in ("exposure_ms", "gain", "white_balance") if k in hdr}
                return dict(ok=True, info=_jsonable(self.cam.set(**kw))), b""
        if op == "freeze":
            with self.camlock:
                if hasattr(self.cam, "freeze"):
                    self.cam.freeze()
            return dict(ok=True), b""
        if op == "info":
            with self.camlock:
                return dict(ok=True, info=_jsonable(self.cam.info())), b""
        raise ValueError(f"unknown op {op}")

    def _mem_get(self, prefix):
        from .device import DIR as MEM
        names = sorted(f for f in (os.listdir(MEM) if os.path.isdir(MEM) else [])
                       if prefix and f.startswith(prefix) and not f.endswith(".tmp"))
        blobs = []
        for f in names:
            with open(os.path.join(MEM, f), "rb") as fh:
                blobs.append(fh.read())
        return dict(ok=True, names=names, sizes=[len(b) for b in blobs], dir=MEM), b"".join(blobs)

    def _mem_put(self, name, blob):
        from .device import DIR as MEM
        name = os.path.basename(name)
        if not name or name.startswith(".") or not name.endswith((".json", ".png", ".jpg")):
            raise PermissionError(f"bad name {name!r}")
        os.makedirs(MEM, exist_ok=True)
        tmp = os.path.join(MEM, name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, os.path.join(MEM, name))
        return dict(ok=True), b""

    # ----- главный цикл
    def _watch(self):
        """Сторож сеанса: нет вестей от программы dead_s — выключить. Простой порта — безобидный запрос (USB не спит)."""
        while not self._stop.is_set():
            time.sleep(0.25)
            ow = self.owner
            if ow is not None and time.monotonic() - ow["last"] > self.dead_s:
                with self.dev:
                    if self.owner is ow and time.monotonic() - ow["last"] > self.dead_s:
                        self.safe_off(f"no word from the program for {self.dead_s:g} s")
                        self._kick("no heartbeat")
            if not self.sim and time.monotonic() - self._last_tx > 2.0:
                try:
                    self._tx(self.P.pack(self.P.GET_DEVICE_TYPE), 1.0)
                except Exception:
                    self._last_tx = time.monotonic()

    def _discover(self):
        """Отвечать на поиск серверов в сети: кто я (без ключа)."""
        from . import __version__
        while not self._stop.is_set() and self.udp is not None:
            try:
                data, peer = self.udp.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break
            if data.strip() != DISCOVER:
                continue
            ow = self.owner
            rep = dict(hello="microcontrol", id=self.uid, port=self.addr[1], name=socket.gethostname(), controller=self.port,
                       camera=self.cam_info.get("model") if self.cam else None, version=__version__,
                       busy=(ow or {}).get("who", {}).get("host") if ow else None)
            try:
                self.udp.sendto(json.dumps(rep).encode("utf-8"), peer)
            except OSError:
                pass

    def serve_forever(self):
        threading.Thread(target=self._watch, daemon=True).start()
        threading.Thread(target=self._discover, daemon=True).start()
        firewall(self.addr[1], self.say)
        self.say(f"MicroControl server: controller {self.port}, camera "
                 f"{(self.cam_info.get('model') or '?') if self.cam else 'none'}, listening on port {self.addr[1]}")
        for ip in _my_ips():
            self.say(f"  in Jupyter: mc.Microscope('net://{self.token}@{ip}:{self.addr[1]}', objective=...)")
        self.say(f"  or simply: mc.Microscope('net://{self.token}@auto', objective=...)  (finds this server in the network)")
        self.say(f"  AI agent (MCP client on that PC): python -m microcontrol mcp net://{self.token}@auto")
        self.say(f"  safe off after {self.dead_s:g} s without the program; Ctrl+C stops the server")
        try:
            self.sock.settimeout(0.5)                 # Windows: accept() иначе не прерывается Ctrl+C
            while not self._stop.is_set():
                try:
                    conn, peer = self.sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                conn.settimeout(None)
                _keepalive(conn)
                threading.Thread(target=self._handle, args=(conn, peer), daemon=True).start()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        self._stop.set()
        for s_ in (self.sock, self.udp):
            try:
                s_.close()
            except (OSError, AttributeError):
                pass
        with self.dev:
            if self.owner is not None:
                self.safe_off("server stopped")
                self._kick("server stopped")
            elif self.state["laser"] or self.state["light"] or self.state["motors"]:
                self.safe_off("server stopped")
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception:
                pass
            self.cam = None
        try:
            self.ser.close()
        except Exception:
            pass
        if self._guard is not None:
            self._guard.done()
            self._guard = None
        if self._lock is not None:
            self._lock.release()
            self._lock = None
        self.L.event("NET-001", "Server stopped.")


def firewall(port, say=print):
    """Windows: разрешить входящие на порт сервера (TCP — связь, UDP — поиск). Без прав администратора —
    сказать, какой командой (один раз, в консоли администратора)."""
    if os.name != "nt":
        return
    import subprocess
    name = f"MicroControl server {port}"
    try:
        r = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule", f"name={name}"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and "MicroControl" in r.stdout:
            return
        ok = True
        for proto in ("TCP", "UDP"):
            r = subprocess.run(["netsh", "advfirewall", "firewall", "add", "rule", f"name={name}", "dir=in",
                                "action=allow", f"protocol={proto}", f"localport={port}", "profile=private,domain"],
                               capture_output=True, text=True, timeout=10)
            ok = ok and r.returncode == 0
        if ok:
            say(f"  firewall: port {port} opened for the local network (rule '{name}')")
            return
    except (OSError, subprocess.SubprocessError):
        pass
    say(f"  firewall: could not open port {port} (not an administrator). Once, in an administrator console:\n"
        f"    netsh advfirewall firewall add rule name=\"{name}\" dir=in action=allow protocol=TCP localport={port}\n"
        f"    netsh advfirewall firewall add rule name=\"{name}\" dir=in action=allow protocol=UDP localport={port}\n"
        "  and make the Wi-Fi / Ethernet network 'Private' in Windows settings")


def _jsonable(d):
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in dict(d or {}).items()}


def _keepalive(s):
    """TCP keepalive: мёртвое соединение (Wi-Fi ушёл, компьютер уснул) замечается за ~15 с, а не часы."""
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        if hasattr(socket, "TCP_KEEPIDLE"):
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 5)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 2)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
        elif hasattr(socket, "SIO_KEEPALIVE_VALS"):        # Windows
            s.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 5000, 2000))
    except OSError:
        pass


def _my_ips():
    ips = []
    try:
        for fam, _, _, _, sa in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            if not sa[0].startswith("127.") and sa[0] not in ips:
                ips.append(sa[0])
    except OSError:
        pass
    try:                                              # адрес, через который идёт наружу (без отправки данных)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        if ip not in ips and not ip.startswith("127."):
            ips.insert(0, ip)
    except OSError:
        pass
    return ips or ["127.0.0.1"]


# ---------------------------------------------------------------- клиент (в программе)
class Link:
    """Связь программы с сервером: транзакции с контроллером, камера, память прибора, «я жив»."""

    def __init__(self, url, token=None, heartbeat_s=1.0):
        url = resolve(url)                             # net://auto — найти сервер в сети
        self.url = url
        self.host, self.port, t_url = parse_url(url)
        self.key = f"{self.host}:{self.port}"
        self.token = token or t_url or os.environ.get("MC_TOKEN") or _saved_tokens().get(self.key)
        self.heartbeat_s = heartbeat_s
        self.lock = threading.RLock()                  # канал команд
        self.camlock = threading.RLock()               # канал камеры (кадры не задерживают команды)
        self.ctl = self.cam_sock = None
        self.codec = os.environ.get("MC_NET_CODEC") or "auto"
        self.mem = None
        self._closed = False
        self.seq, self.pending, self.sid = 0, None, None
        self.t_session = time.time()
        self._connect()
        self.was_reset = False
        threading.Thread(target=self._beat, daemon=True).start()

    def _open(self, role, extra=None):
        from .errors import CommError, SafetyError
        try:
            s = socket.create_connection((self.host, self.port), timeout=10)
        except OSError as e:
            raise CommError("E112", f"{self.host}:{self.port}: {e}. Is 'python -m microcontrol serve' running "
                                    "there, and the address right?") from None
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _keepalive(s)
        hello, _ = _recv(s)
        if hello.get("hello") != "microcontrol":
            s.close()
            raise CommError("E112", f"{self.host}:{self.port} is not a MicroControl server")
        if not self.token and role not in ("stop",):
            s.close()
            raise SafetyError("E111", f"no key for {self.key}: net://KEY@{self.host} (the server prints the key) "
                                      "or the MC_TOKEN variable")
        hdr = dict(role=role, mac=_mac(self.token, hello["nonce"]) if self.token else "",
                   who=dict(host=socket.gethostname(), pid=os.getpid()), **(extra or {}))
        _send(s, hdr)
        rep, _ = _recv(s)
        if rep.get("err"):
            s.close()
            cls = SafetyError if rep["err"] in ("E106", "E111") else CommError
            raise cls(rep["err"], f"{self.key}: {rep.get('msg', '')}")
        s.settimeout(self.TIMEOUT)                     # Wi-Fi пропал молча: запрос не висит вечно — переподключение
        return s, rep

    TIMEOUT = 30.0                                     # с: ответа нет — связь считается оборванной

    def _connect(self):
        self.ctl, self.info = self._open("ctl", dict(prev_sid=self.sid) if self.sid else None)
        _save_token(self.key, self.token)
        self.sid = self.info["sid"]
        self.last = time.monotonic()
        if self.codec == "auto":                       # быстрая сеть — без потерь, медленная — JPEG 95
            t = time.monotonic()
            self.call("bench", n=4 << 20)
            mbs = 4 / max(1e-3, time.monotonic() - t)
            self.mbs = mbs
            self.codec_now = "raw" if mbs >= 90 else "png" if mbs >= 25 else "jpg95"   # гигабит: без сжатия (старому
                                                                                   # компьютеру не кодировать 5 Мп)
        else:
            self.mbs, self.codec_now = None, self.codec
        if self.cam_sock is not None:
            try:
                self.cam_sock.close()
            except OSError:
                pass
            self.cam_sock = None

    def _cam(self):
        if self.cam_sock is None:
            self.cam_sock, _ = self._open("cam", dict(sid=self.sid))
        return self.cam_sock

    def call(self, op, blob=b"", _cam=False, **hdr):
        """Запрос серверу -> (ответ, данные). Обрыв — ConnectionError (OSError), ошибка прибора — CommError."""
        from .errors import CommError, CameraError
        lk = self.camlock if _cam else self.lock
        with lk:
            if self._closed:
                raise ConnectionError("closed")
            s = self._cam() if _cam else self.ctl
            try:
                tmo = hdr.get("timeout")                   # команда контроллеру может идти долго (дом, автофокус)
                s.settimeout(self.TIMEOUT + (float(tmo) if isinstance(tmo, (int, float)) else 0.0))
                _send(s, dict(op=op, **hdr), blob)
                rep, out = _recv(s)
            except (OSError, struct.error, ValueError) as e:
                if _cam:
                    self.cam_sock = None
                raise ConnectionError(f"connection to {self.key} lost: {e}") from None
            if not _cam:
                self.last = time.monotonic()
        if rep.get("err"):
            raise (CameraError if _cam else CommError)(rep["err"], rep.get("msg", ""))
        return rep, out

    def _beat(self):
        """«Я жив» раз в heartbeat_s, если команд не было; заодно — память прибора на сервер."""
        while not self._closed:
            time.sleep(self.heartbeat_s / 2)
            if self._closed or getattr(self, "_paused", False):
                continue
            try:
                if self.mem is not None and self.mem.dirty:
                    self.mem.push()
                elif time.monotonic() - self.last >= self.heartbeat_s:
                    self.call("ping")
            except Exception:
                pass

    # ----- то, что видит core
    def transact(self, frame, timeout):
        with self.lock:
            self.seq += 1
            seq = self.seq
            try:
                rep, raw = self.call("tx", frame, timeout=timeout, seq=seq)
            except ConnectionError:
                self.pending = seq                    # дошло ли — скажет сервер после переподключения
                raise
        return raw[4], bytes(raw[5:-1]), raw

    def pending_result(self):
        """После reconnect: команда, на которой оборвалась связь. -> raw (выполнена, вот ответ), False (не дошла до
        контроллера — повторять можно), None (неизвестно: сервер перезапускали)."""
        seq, self.pending = self.pending, None
        if seq is None:
            return None
        prev = self.info.get("prev")
        if prev is None:                              # сервер не знает прошлый сеанс (перезапускали)
            return None
        if prev["seq"] == seq:
            return bytes.fromhex(prev["raw"])
        return False                                  # запросы идут по очереди: последняя выполненная — раньше

    def reconnect(self):
        """Связь пропала: подключиться заново (новый сеанс). was_reset — сервер успел выключить прибор."""
        with self.lock, self.camlock:
            for s in (self.ctl, self.cam_sock):
                try:
                    s.close()
                except Exception:
                    pass
            self.ctl = self.cam_sock = None
            self._connect()
        lo = self.info.get("last_off")
        self.was_reset = bool(lo and lo["time"] >= self.t_session - 1)
        return self.was_reset

    def close(self, bye=True):
        if self._closed:
            return
        if self.mem is not None and self.mem.dirty:
            try:
                self.mem.push()
            except Exception:
                pass
        if bye:
            try:
                self.call("bye")
            except Exception:
                pass
        self._closed = True
        for s in (self.ctl, self.cam_sock):
            try:
                s.close()
            except Exception:
                pass

    @property
    def is_open(self):
        return not self._closed


class RemoteSerial:
    """Порт контроллера через сервер: core зовёт transact (одна транзакция — один сетевой круг)."""
    def __init__(self, link):
        self.link = link

    def transact(self, frame, timeout):
        return self.link.transact(frame, timeout)

    def reset_input_buffer(self):
        pass

    @property
    def is_open(self):
        return self.link.is_open

    def close(self):
        self.link.close()


class RemoteCamera:
    """Камера через сервер: те же grab / set / freeze / info / close, что у MVCamera."""
    def __init__(self, link):
        self.link = link
        if not link.info.get("camera"):
            from .errors import CameraError
            raise CameraError("E501", f"no camera on the server {link.key}")
        self.name = link.info["camera"].get("model", "remote")

    def grab(self, flush=2):
        import cv2, numpy as np
        rep, b = self.link.call("grab", _cam=True, flush=int(flush), codec=self.link.codec_now)
        if "shape" in rep:
            return np.frombuffer(b, rep.get("dtype", "uint8")).reshape(rep["shape"]).copy()
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            from .errors import CameraError
            raise CameraError("E503", "bad frame from the server")
        return img

    def set(self, exposure_ms=None, gain=None, white_balance=False, **kw):
        h = dict(white_balance=bool(white_balance))
        if exposure_ms is not None:
            h["exposure_ms"] = exposure_ms
        if gain is not None:
            h["gain"] = gain
        return self.link.call("set", _cam=True, **h)[0]["info"]

    def freeze(self):
        self.link.call("freeze", _cam=True)

    def info(self):
        return self.link.call("info", _cam=True)[0]["info"]

    def close(self):
        pass


def remote_memory(link, model, serial):
    """Память прибора с сервера: копия в ~/.microcontrol/remote/<адрес>/devices, изменения — обратно на сервер."""
    from .device import DeviceMemory
    import re

    class RemoteMemory(DeviceMemory):
        def __init__(self):
            name = re.sub(r'[^A-Za-z0-9._-]+', '-', f"{model}_{serial}").strip('-') or 'device'
            cache = os.path.join(DIR, "remote", re.sub(r"[^A-Za-z0-9._-]+", "_", link.key), "devices")
            os.makedirs(cache, exist_ok=True)
            rep, blob = link.call("mem_get", prefix=name)
            self.server_dir, self.cache, self.pushed = rep.get("dir", ""), cache, {}
            for f in os.listdir(cache):
                if f.startswith(name) and f not in rep["names"]:
                    os.remove(os.path.join(cache, f))
            o = 0
            for f, n in zip(rep["names"], rep["sizes"]):
                b = blob[o:o + n]
                o += n
                if f.endswith(".json"):
                    b = json.dumps(_paths(json.loads(b), self.server_dir, cache), ensure_ascii=False, indent=1).encode()
                with open(os.path.join(cache, f), "wb") as fh:
                    fh.write(b)
                self.pushed[f] = os.path.getmtime(os.path.join(cache, f))
            self.prefix, self.dirty = name, False
            super().__init__(model, serial, folder=cache)

        def save(self):
            """Записать и сразу отдать серверу (положение не теряется, если программа вылетит)."""
            super().save()
            self.dirty = True
            try:
                self.push()
            except Exception:
                self.dirty = True                     # связи нет — отдаст «я жив» или close()

        def push(self):
            """Изменённые файлы памяти — на сервер (пути — в папку сервера)."""
            self.dirty = False
            for f in sorted(os.listdir(self.cache)):
                p = os.path.join(self.cache, f)
                if not f.startswith(self.prefix) or f.endswith(".tmp"):
                    continue
                t = os.path.getmtime(p)
                if self.pushed.get(f) == t and not f.endswith(".json"):
                    continue
                with open(p, "rb") as fh:
                    b = fh.read()
                if f.endswith(".json"):
                    b = json.dumps(_paths(json.loads(b), self.cache, self.server_dir), ensure_ascii=False,
                                   indent=1).encode()
                link.call("mem_put", b, name=f)
                self.pushed[f] = t

    return RemoteMemory()


def _paths(o, a, b):
    """Пути файлов памяти: папка a -> папка b (сервер хранит свои, копия — свои)."""
    if isinstance(o, dict):
        return {k: _paths(v, a, b) for k, v in o.items()}
    if isinstance(o, list):
        return [_paths(v, a, b) for v in o]
    if isinstance(o, str) and a and b and o.startswith(a):
        import re
        sep = "\\" if "\\" in b else "/"
        return b.rstrip("/\\") + sep + sep.join(x for x in re.split(r"[\\/]", o[len(a):]) if x)
    return o


def emergency_stop(url):
    """Аварийная остановка по сети: сервер выключает прибор и отключает программу. -> строка."""
    host, port, _ = parse_url(resolve(url))
    try:
        s = socket.create_connection((host, port), timeout=5)
        hello, _ = _recv(s)
        _send(s, dict(role="stop"))
        rep, _ = _recv(s)
        s.close()
    except OSError as e:
        return f"Server {host}:{port} does not answer ({e}): switch the laser off at the instrument."
    was = rep.get("was")
    return (f"Server {host}:{port}: laser, motors, light off" +
            (f"; program {was.get('host')} PID {was.get('pid')} disconnected." if was else "; no program was connected."))


def status(url, token=None):
    """Состояние сервера: прибор, что включено, кто подключён, последнее выключение."""
    host, port, t_url = parse_url(resolve(url))
    tok = token or t_url or os.environ.get("MC_TOKEN") or _saved_tokens().get(f"{host}:{port}")
    s = socket.create_connection((host, port), timeout=5)
    hello, _ = _recv(s)
    _send(s, dict(role="status", mac=_mac(tok, hello["nonce"]) if tok else ""))
    rep, _ = _recv(s)
    s.close()
    return rep


def serve(argv=None):
    """python -m microcontrol serve [--port COM5|auto|sim] [--listen 0.0.0.0:5150] [--no-camera] [--sdk DIR]
    [--dead 10] [--logs DIR]"""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m microcontrol serve")
    ap.add_argument("--port", default="auto")
    ap.add_argument("--listen", default=f"0.0.0.0:{DEFAULT_PORT}")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--sdk", default=None)
    ap.add_argument("--dead", type=float, default=10.0)
    ap.add_argument("--logs", default=None)
    ap.add_argument("--token", default=None)
    a = ap.parse_args(argv)
    h, _, p = a.listen.rpartition(":")
    srv = Server(a.port, (h or "0.0.0.0", int(p)), camera=not a.no_camera, sdk_folder=a.sdk, dead_s=a.dead,
                 token=a.token, logs=a.logs)
    srv.serve_forever()
