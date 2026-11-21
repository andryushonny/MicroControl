"""Interface for AI agents: every function of MicroControl as a tool with a strict JSON schema.

A local language model (Ollama, LM Studio, any MCP client) can drive the microscope and the scan viewer
through these tools. Everything that moves the stage or throws away the operator's work is shown to the
operator first and runs only after "Allow". The laser is never available to the agent.

    ag = mc.Agent(m)                        # m = mc.Microscope(...) or None (only scans on disk)
    ag.call('get_status')                   # -> dict, always JSON-safe
    ag.tools()                              # tool list (OpenAI / Ollama format)
    mc.agent.chat(m, model='qwen3:8b')      # chat with a local model through Ollama
    python -m microcontrol.agent --mcp      # MCP server (stdio) for any MCP client

The manual for the model is agent_manual.md (tool get_manual).
"""
import base64
import datetime as _dt
import difflib
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import cv2

from . import samples as SM
from .errors import MicroControlError

MANUAL = os.path.join(os.path.dirname(__file__), 'agent_manual.md')
IMG_SIDE = 1024                        # pictures for the model: longest side, px

# settings the agent may change (with approval); everything else only the operator changes in Python
AGENT_SETTINGS = {'light', 'exposure', 'objective', 'focus_method', 'focus_range', 'focus_step', 'focus_every',
                  'refocus', 'frame_scale', 'jpg_quality', 'photo_pause', 'frame_format', 'still_check'}


class AgentError(Exception):
    def __init__(self, code, message, hint=''):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


# ------------------------------------------------------------------ registry
_TOOLS = {}


def tool(name, desc, params=None, required=(), risk='read', ask=None):
    """risk: 'read' — looks only; 'work' — writes files / changes marks (no approval);
    'act' — moves the stage, changes the instrument or removes work: the operator must allow it."""
    def deco(fn):
        _TOOLS[name] = dict(name=name, desc=desc, params=params or {}, required=list(required), risk=risk,
                            ask=ask, fn=fn)
        return fn
    return deco


REASON = {'type': 'string', 'description': "Why you need this, one short sentence. Shown to the operator."}
FOLDER = {'type': 'string', 'description': "Scan folder: its name from list_scans (or a full path). "
                                           "For a session with several samples: the sample folder, e.g. '<session>/sample_01'."}
SQUARE = {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 2, 'maxItems': 2,
          'description': "Square name [X, Y] as printed by the lithography (or lattice index if no labels yet)."}


def _check_args(t, args):
    """Strict check of the arguments against the schema: unknown, missing, wrong type, out of range."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise AgentError('BAD_ARGS', f"Arguments of {t['name']} must be an object (JSON dict).")
    unknown = [k for k in args if k not in t['params']]
    if unknown:
        raise AgentError('BAD_ARGS', f"Unknown argument(s) {unknown} for {t['name']}.",
                         f"Allowed: {sorted(t['params']) or 'none'}.")
    miss = [k for k in t['required'] if args.get(k) in (None, '')]
    if miss:
        raise AgentError('BAD_ARGS', f"Missing argument(s) {miss} for {t['name']}.")
    out = {}
    for k, v in args.items():
        s = t['params'][k]
        ty = s.get('type')
        if v is None:
            continue
        if ty == 'number':
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    raise AgentError('BAD_ARGS', f"{k} must be a number, got {v!r}.")
            if not math.isfinite(v):
                raise AgentError('BAD_ARGS', f"{k} must be a finite number.")
        elif ty == 'integer':
            if isinstance(v, bool):
                raise AgentError('BAD_ARGS', f"{k} must be an integer, got {v!r}.")
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            if isinstance(v, str) and v.strip().lstrip('-').isdigit():
                v = int(v)
            if not isinstance(v, int):
                raise AgentError('BAD_ARGS', f"{k} must be an integer, got {v!r}.")
        elif ty == 'boolean':
            if isinstance(v, str) and v.lower() in ('true', 'false'):
                v = v.lower() == 'true'
            if not isinstance(v, bool):
                raise AgentError('BAD_ARGS', f"{k} must be true or false, got {v!r}.")
        elif ty == 'string':
            if not isinstance(v, str):
                raise AgentError('BAD_ARGS', f"{k} must be a string, got {v!r}.")
        elif ty == 'array':
            if not isinstance(v, list):
                raise AgentError('BAD_ARGS', f"{k} must be a list, got {v!r}.")
        if 'enum' in s and v not in s['enum']:
            raise AgentError('BAD_ARGS', f"{k} = {v!r} is not allowed.", f"Allowed: {s['enum']}.")
        if 'minimum' in s and v < s['minimum'] or 'maximum' in s and v > s['maximum']:
            raise AgentError('BAD_ARGS', f"{k} = {v} is out of range {s.get('minimum')} .. {s.get('maximum')}.")
        out[k] = v
    return out


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 3)
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if v is None or isinstance(v, (str, int, bool)):
        return v
    return str(v)


# ------------------------------------------------------------------ operator approval
def _ask_window(title, text):
    """Window with three buttons; returns 'once', 'session' or 'deny'. Falls back to the console."""
    try:
        import tkinter as tk
    except Exception:
        return _ask_console(title, text)
    ans = {'v': 'deny'}
    try:
        root = tk.Tk()
    except Exception:
        return _ask_console(title, text)
    root.title("MicroControl — agent asks")
    try:
        from .util import icon_path
        if icon_path():
            root.iconbitmap(icon_path())
    except Exception:
        pass
    root.attributes('-topmost', True)
    root.configure(bg='#1e1f24', padx=18, pady=14)
    tk.Label(root, text=title, font=('Segoe UI', 13, 'bold'), fg='#ffd257', bg='#1e1f24',
             anchor='w', justify='left').pack(fill='x')
    tk.Label(root, text=text, font=('Segoe UI', 11), fg='#e8e8e8', bg='#1e1f24', justify='left',
             wraplength=560, anchor='w').pack(fill='x', pady=(8, 14))
    row = tk.Frame(root, bg='#1e1f24')
    row.pack(fill='x')

    def set_(v):
        ans['v'] = v
        root.destroy()
    for lab, v, bg in (("Allow", 'once', '#2e7d32'), ("Allow for this session", 'session', '#1565c0'),
                       ("Deny", 'deny', '#8e2b2b')):
        tk.Button(row, text=lab, command=lambda v=v: set_(v), bg=bg, fg='white', activebackground=bg,
                  font=('Segoe UI', 11, 'bold'), relief='flat', padx=14, pady=6).pack(side='left', padx=(0, 8))
    root.protocol("WM_DELETE_WINDOW", lambda: set_('deny'))
    root.bind('<Escape>', lambda e: set_('deny'))
    root.mainloop()
    return ans['v']


def _ask_console(title, text):
    print(f"\n=== {title} ===\n{text}\n[y] allow   [s] allow for this session   [n] deny", file=sys.stderr, flush=True)
    try:
        a = input("> ").strip().lower()
    except EOFError:
        return 'deny'
    return {'y': 'once', 'yes': 'once', 's': 'session', 'д': 'once', 'да': 'once'}.get(a, 'deny')


# ------------------------------------------------------------------ agent
class Agent:
    """Tools for an AI agent. m — Microscope (or None: only scans on disk, folder root `scans`).
    approve: 'window' (default), 'console', a function (title, text) -> 'once' / 'session' / 'deny',
    or 'deny' (read-only agent). log: JSON lines file of every call (default: <logs>/agent_<date>.jsonl)."""

    def __init__(self, m=None, scans=None, approve='window', max_calls=500, max_actions_per_min=30, log=None):
        self.m = m
        self.root = scans or (m.dirs['scans'] if m is not None else os.path.abspath('scans'))
        self.approve = approve
        self.allowed = set()                           # tools allowed for the whole session
        self.max_calls, self.calls = max_calls, 0
        self.max_apm, self.acts = max_actions_per_min, []
        self.fails = {}
        self._scenes = {}
        logs = (m.dirs.get('logs') if m is not None and isinstance(getattr(m, 'dirs', None), dict) else None) \
            or os.path.join(os.path.dirname(os.path.abspath(self.root)), 'logs')
        os.makedirs(logs, exist_ok=True)
        self.log = log or os.path.join(logs, f"agent_{_dt.date.today().isoformat()}.jsonl")
        self.out = os.path.join(m.folder if m is not None else os.path.dirname(os.path.abspath(self.root)), 'agent')
        os.makedirs(self.out, exist_ok=True)
        self.images = []                               # pictures made by the last call (for MCP / Ollama)

    # ---- schemas
    def tools(self, fmt='openai'):
        out = []
        for t in _TOOLS.values():
            props = dict(t['params'])
            sch = {'type': 'object', 'properties': props, 'required': t['required'], 'additionalProperties': False}
            desc = t['desc'] + {'act': "  [Needs operator approval.]", 'work': "", 'read': ""}[t['risk']]
            if fmt == 'mcp':
                out.append(dict(name=t['name'], description=desc, inputSchema=sch))
            else:
                out.append({'type': 'function', 'function': dict(name=t['name'], description=desc, parameters=sch)})
        return out

    # ---- one call
    def call(self, name, args=None):
        """Run tool `name` with `args` (dict or JSON string). Always returns a dict:
        {"ok": true, "result": ...} or {"ok": false, "error": {"code", "message", "hint"}}."""
        t0 = time.monotonic()
        self.images = []
        try:
            if isinstance(args, str):
                try:
                    args = json.loads(args or '{}')
                except json.JSONDecodeError as e:
                    raise AgentError('BAD_ARGS', f"Arguments are not valid JSON: {e}.")
            self.calls += 1
            if self.calls > self.max_calls:
                raise AgentError('LIMIT', f"Session limit of {self.max_calls} tool calls reached.",
                                 "Stop and report to the operator.")
            t = _TOOLS.get(name)
            if t is None:
                near = difflib.get_close_matches(str(name), list(_TOOLS), n=3)
                raise AgentError('NO_TOOL', f"There is no tool {name!r}.",
                                 f"Did you mean {near}? All tools: {sorted(_TOOLS)}." if near else
                                 f"All tools: {sorted(_TOOLS)}.")
            a = _check_args(t, args)
            if t['risk'] == 'act':
                self._rate()
                self._permit(t, a)
            res = t['fn'](self, **a)
            out = {'ok': True, 'result': _jsonable(res)}
            self.fails.pop(name, None)
        except AgentError as e:
            out = {'ok': False, 'error': dict(code=e.code, message=e.message, hint=e.hint)}
        except MicroControlError as e:
            out = {'ok': False, 'error': dict(code=getattr(e, 'code', 'ERROR'), message=str(e),
                                              hint="The instrument refused. Read the message; do not repeat the "
                                                   "same call unchanged. Ask the operator if unsure.")}
        except (ValueError, FileNotFoundError, KeyError) as e:
            out = {'ok': False, 'error': dict(code='FAILED', message=f"{type(e).__name__}: {e}", hint="")}
        except Exception as e:                                     # noqa — never crash the agent loop
            out = {'ok': False, 'error': dict(code='INTERNAL', message=f"{type(e).__name__}: {e}",
                                              hint="Report this to the operator.")}
            traceback.print_exc(file=sys.stderr)
        if not out['ok']:
            n = self.fails[name] = self.fails.get(name, 0) + 1
            if n >= 3:
                out['error']['hint'] = (out['error']['hint'] + " " if out['error']['hint'] else "") + \
                    f"{name} failed {n} times in a row: stop and ask the operator."
        if self.images:
            out['images'] = [p for p in self.images]
        try:
            with open(self.log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(dict(time=_dt.datetime.now().isoformat(timespec='seconds'), tool=name,
                                        args=args if isinstance(args, (dict, type(None))) else str(args),
                                        ok=out['ok'], error=out.get('error'),
                                        seconds=round(time.monotonic() - t0, 2)), ensure_ascii=False) + "\n")
        except OSError:
            pass
        return out

    def _rate(self):
        now = time.monotonic()
        self.acts = [t for t in self.acts if now - t < 60]
        if len(self.acts) >= self.max_apm:
            raise AgentError('LIMIT', f"More than {self.max_apm} actions in a minute.", "Slow down; check the result of each action.")
        self.acts.append(now)

    def _permit(self, t, a):
        if t['name'] in self.allowed:
            return
        if self.approve == 'deny':
            raise AgentError('DENIED', "This agent is read-only: actions are not allowed.", "Tell the operator what you would do.")
        text = (t['ask'](self, a) if t['ask'] else f"{t['name']}({json.dumps(a, ensure_ascii=False)})")
        reason = a.get('reason')
        if reason:
            text += f"\n\nReason given by the agent: {reason}"
        title = f"Agent wants to: {t['name'].replace('_', ' ')}"
        f = self.approve if callable(self.approve) else {'window': _ask_window, 'console': _ask_console}[self.approve]
        v = f(title, text)
        if v == 'session':
            self.allowed.add(t['name'])
        elif v != 'once':
            raise AgentError('DENIED', f"The operator denied {t['name']}.",
                             "Do not repeat it. Ask the operator in text what to do instead.")

    # ---- helpers
    def need_m(self):
        if self.m is None:
            raise AgentError('NO_MICROSCOPE', "No microscope is connected: only scans on disk are available.")
        return self.m

    def folder(self, name):
        cands = [name, os.path.join(self.root, name)]
        for c in cands:
            if c and os.path.isdir(c):
                c = os.path.abspath(c)
                return c
        near = difflib.get_close_matches(os.path.basename(str(name)), self._scan_names(), n=3)
        raise AgentError('NO_FOLDER', f"No scan folder {name!r}.", f"Call list_scans. Close names: {near}" if near else
                         "Call list_scans to see the folders.")

    def _scan_names(self):
        if not os.path.isdir(self.root):
            return []
        if os.path.exists(os.path.join(self.root, 'sample.json')):      # образец: все съёмки всех выделений
            from . import project as P_
            return [os.path.relpath(f, self.root) for f in P_.all_scans(self.root)]
        return sorted(d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)))

    def grid(self, folder):
        from .mask import Grid
        if not os.path.exists(os.path.join(folder, 'meta.json')):
            if SM.session_of(folder)[0]:
                raise AgentError('IS_SESSION', "This is a session with several samples, not one scan.",
                                 f"Use a sample folder: {[os.path.relpath(s, self.root) for s in SM.samples(folder)]}.")
            raise AgentError('NO_SCAN', f"{folder} is not a scan folder.")
        return Grid(folder)

    def scene(self, folder):
        from .viewer import _Scene
        if not os.path.isdir(os.path.join(folder, 'join')) and not os.path.exists(os.path.join(folder, 'viewer')):
            raise AgentError('NOT_STITCHED', "This scan is not stitched yet.", "Call stitch first.")
        sc = self._scenes.get(folder)
        if sc is None:
            if len(self._scenes) > 2:
                for k in list(self._scenes)[:-1]:
                    self._scenes.pop(k).st.close()
            sc = self._scenes[folder] = _Scene(folder)
        return sc

    def px_um(self, sc):
        return float(sc.st.meta.get('pixel_size_um') or 1.0)

    def save_img(self, img, tag):
        s = IMG_SIDE / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        p = os.path.join(self.out, f"{_dt.datetime.now().strftime('%H%M%S_%f')[:-3]}_{tag}.png")
        ok, buf = cv2.imencode('.png', img)
        buf.tofile(p)
        self.images.append(p)
        return p, img.shape[1], img.shape[0]

    def square_ij(self, g, sq):
        """Square name [X, Y] -> lattice index (i, j)."""
        if not isinstance(sq, (list, tuple)) or len(sq) != 2 or not all(isinstance(v, (int, np.integer)) for v in sq):
            raise AgentError('BAD_ARGS', f"A square is [X, Y] with two integers, got {sq!r}.")
        P, t, _ = g.numbering()
        ij = np.round(P.T @ (np.array(sq) - t)).astype(int)
        return int(ij[0]), int(ij[1])


# ------------------------------------------------------------------ tools: help and state
@tool('get_manual', "Read the manual for agents: units, coordinates, workflow, safety. Read 'overview' first.",
      {'topic': {'type': 'string', 'enum': ['overview', 'stage', 'areas', 'scan', 'mosaic', 'squares', 'safety',
                                            'errors', 'all']}})
def _manual(ag, topic='overview'):
    txt = open(MANUAL, encoding='utf-8').read()
    if topic == 'all':
        return txt
    parts = {}
    cur = None
    for line in txt.split('\n'):
        if line.startswith('## '):
            cur = line[3:].split()[0].lower()
            parts[cur] = []
        if cur:
            parts[cur].append(line)
    return "\n".join(parts.get(topic, [f"No topic {topic}."]))


@tool('get_status', "Current state: connection, objective, stage position (um), limits, light, exposure, "
                    "selections (with objective and the selection they lie in) and focus points, last scan. Call it first and after anything goes wrong.")
def _status(ag):
    m = ag.m
    if m is None:
        return dict(microscope="not connected (scans on disk only)", scans_folder=ag.root,
                    scans=len(ag._scan_names()))
    p, rp = m.position, m.real_position
    lim = m.limits if hasattr(m, 'limits') else None
    fw, fh = m.frame_size
    return dict(microscope="connected", port=str(m.port), objective=f"{m.objective}x",
                position_um=dict(x=p.x, y=p.y, z=p.z), real_position_um=dict(x=rp.x, y=rp.y, z=rp.z),
                limits=str(lim) if lim is not None else None,
                field_of_view_um=[round(fw, 1), round(fh, 1)], pixel_um=m.pixel_size,
                light=m.light, exposure_ms=m.exposure, laser="not available to agents",
                max_single_move_um=dict(xy=m.max_move, z_up=m.max_move_z, z_down=m.max_move_down, jump=m.max_jump),
                areas=SM.areas_info(m), last_scan=getattr(m, 'last_scan', None), scans_folder=ag.root)


@tool('get_settings', "All instrument settings with values and allowed ranges. Which ones you may change: "
                      "see change_setting.")
def _settings(ag):
    from .core import Microscope, Setting
    m = ag.need_m()
    out = {}
    for k, v in vars(Microscope).items():
        if isinstance(v, Setting):
            try:
                out[k] = dict(value=getattr(m, k), allowed=v.allowed(m), unit=v.unit, doc=v.doc,
                              agent_may_change=k in AGENT_SETTINGS)
            except Exception:
                pass
    return out


@tool('change_setting', "Change one setting. Agents may change only: " + ", ".join(sorted(AGENT_SETTINGS)) +
      ". Safety limits, orientation and the laser are for the operator only.",
      {'name': {'type': 'string', 'enum': sorted(AGENT_SETTINGS)},
       'value': {'type': ['number', 'string', 'boolean'], 'description': "New value (see get_settings for the range)."},
       'reason': REASON}, ('name', 'value', 'reason'), risk='act',
      ask=lambda ag, a: f"Change {a['name']}: {getattr(ag.m, a['name'], '?')} -> {a['value']}" +
      ("\nThe operator must physically put this objective in place first." if a['name'] == 'objective' else ""))
def _change(ag, name, value, reason):
    m = ag.need_m()
    old = getattr(m, name)
    setattr(m, name, value)
    return dict(setting=name, old=old, new=getattr(m, name))


# ------------------------------------------------------------------ tools: camera and stage
@tool('look', "Get a picture (PNG file; also attached as an image if your client supports images). "
              "target='camera': the live camera frame now (no movement). target='mosaic': a part of a stitched scan "
              "with the red square grid, square names, selection (green) and hints (yellow); the border ruler is in "
              "mosaic pixels — use those numbers for mark_cross and label_square. target='map': the stage map with "
              "selections, focus points and the current position.",
      {'target': {'type': 'string', 'enum': ['camera', 'mosaic', 'map']}, 'folder': FOLDER,
       'x0': {'type': 'number', 'description': "mosaic: left edge, px (default: whole mosaic)"},
       'y0': {'type': 'number', 'description': "mosaic: top edge, px"},
       'x1': {'type': 'number', 'description': "mosaic: right edge, px"},
       'y1': {'type': 'number', 'description': "mosaic: bottom edge, px"}}, ('target',))
def _look(ag, target, folder=None, x0=None, y0=None, x1=None, y1=None):
    if target == 'camera':
        m = ag.need_m()
        img = m.image().copy()
        h, w = img.shape[:2]
        cv2.drawMarker(img, (w // 2, h // 2), (0, 220, 255), cv2.MARKER_CROSS, max(20, w // 30), 2)
        L = 100 if m.pixel_size * w > 400 else 20                                    # шкала, мкм
        lp = int(L / m.pixel_size)
        cv2.rectangle(img, (20, h - 34), (20 + lp, h - 26), (255, 255, 255), -1)
        cv2.putText(img, f"{L} um", (20, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        p = m.position
        path, W, H = ag.save_img(img, 'camera')
        return dict(image=path, width=W, height=H, position_um=dict(x=p.x, y=p.y, z=p.z),
                    field_of_view_um=[round(v, 1) for v in m.frame_size], sharpness=round(float(
                        __import__('microcontrol.focus', fromlist=['x']).sharpness(img, m.focus_method)), 1),
                    note="The yellow cross is the centre of the field of view.")
    if target == 'map':
        return _map_image(ag)
    if not folder:
        raise AgentError('BAD_ARGS', "target='mosaic' needs folder.")
    return _mosaic_image(ag, ag.folder(folder), x0, y0, x1, y1)


def _mosaic_image(ag, folder, x0, y0, x1, y1):
    g = ag.grid(folder)
    sc = ag.scene(folder)
    Wm, Hm = sc.size
    x0 = 0 if x0 is None else max(0.0, float(x0))
    y0 = 0 if y0 is None else max(0.0, float(y0))
    x1 = Wm if x1 is None else min(float(Wm), float(x1))
    y1 = Hm if y1 is None else min(float(Hm), float(y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise AgentError('BAD_ARGS', f"Empty region; the mosaic is 0..{Wm:.0f} x 0..{Hm:.0f} px.")
    pad = 46
    s = min((IMG_SIDE - pad) / (x1 - x0), (IMG_SIDE - pad) / (y1 - y0))
    w, h = int((x1 - x0) * s), int((y1 - y0) * s)
    img = sc.render((x0 + x1) / 2, (y0 + y1) / 2, s, w, h)
    to = lambda P: (int(round((P[0] - x0) * s)), int(round((P[1] - y0) * s)))
    if g.ready():
        cells = list(g.cells_range((x0, y0), (x1, y1)))
        num = g.numbering()
        ov = img.copy()
        for c in cells:
            col = (90, 200, 90) if c in g.selected else ((60, 210, 230) if c in g.useful else None)
            if col:
                cv2.fillConvexPoly(ov, np.array([to(q) for q in g.corners(*c)], np.int32), col)
        img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
        for c in cells:
            q = np.array([to(q) for q in g.corners(*c)], np.int32)
            cv2.polylines(img, [q], True, (40, 40, 230), 1, cv2.LINE_AA)
        side = np.hypot(*g.cr.vectors()[0]) * s
        if side > 36:
            fs = min(0.55, side / 120)
            for c in cells:
                cx, cy = to(g.cr.point(c[0] + 0.5, c[1] + 0.5))
                if 0 <= cx < w and 0 <= cy < h:
                    t_ = "{},{}".format(*g.name(*c, num=num))
                    cv2.putText(img, t_, (cx - int(len(t_) * 5 * fs / 0.5), cy + 4), cv2.FONT_HERSHEY_SIMPLEX, fs,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, t_, (cx - int(len(t_) * 5 * fs / 0.5), cy + 4), cv2.FONT_HERSHEY_SIMPLEX, fs,
                                (255, 255, 255), 1, cv2.LINE_AA)
    for q in g.cr.marks:
        cv2.circle(img, to(q), 7, (40, 40, 255), 2, cv2.LINE_AA)
    # линейка в пикселях мозаики
    out = np.full((h + pad, w + pad, 3), 32, np.uint8)
    out[pad:, pad:] = img
    step = 10 ** math.floor(math.log10(max(1.0, (x1 - x0) / 5)))
    step *= 5 if (x1 - x0) / step > 25 else (2 if (x1 - x0) / step > 10 else 1)
    for v in np.arange(math.ceil(x0 / step) * step, x1, step):
        u = pad + int((v - x0) * s)
        cv2.line(out, (u, pad - 8), (u, pad), (200, 200, 200), 1)
        cv2.putText(out, f"{v:.0f}", (u - 14, pad - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    for v in np.arange(math.ceil(y0 / step) * step, y1, step):
        u = pad + int((v - y0) * s)
        cv2.line(out, (pad - 8, u), (pad, u), (200, 200, 200), 1)
        cv2.putText(out, f"{v:.0f}", (2, u + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
    path, W, H = ag.save_img(out, 'mosaic')
    return dict(image=path, width=W, height=H, mosaic_size_px=[round(Wm), round(Hm)], region_px=[x0, y0, x1, y1],
                image_px_per_mosaic_px=round(s, 4), ruler="numbers on the top and left borders are mosaic px",
                grid=("ready" if g.ready() else "not set: mark two neighbouring crosses with mark_cross"),
                crosses_marked=len(g.cr.marks), selected=len(g.selected), hints=len(g.useful))


def _map_image(ag):
    m = ag.need_m()
    SM.init_areas(m)
    fw, fh = m.frame_size
    pts = [(m.real_position.x, m.real_position.y)]
    for rg in m._regions + ([m._pending] if m._pending else []):
        pts += [rg['A'], rg.get('B') or rg['A']]
    P = np.array(pts, float)
    lo, hi = P.min(0) - 3 * max(fw, fh), P.max(0) + 3 * max(fw, fh)
    S = 900
    s = (S - 80) / max(hi - lo)
    img = np.full((S, S, 3), 28, np.uint8)
    off = np.array([m._off['x'], m._off['y']])
    to = lambda x, y: (int(60 + (x - lo[0]) * s), int(40 + (y - lo[1]) * s))
    step = 10 ** math.floor(math.log10(max(1.0, max(hi - lo) / 6)))
    for v in np.arange(math.ceil((lo[0] - off[0]) / step) * step, hi[0] - off[0], step):
        u = to(v + off[0], lo[1])[0]
        cv2.line(img, (u, 40), (u, S - 20), (55, 55, 55), 1)
        cv2.putText(img, f"{v:.0f}", (u - 18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    for v in np.arange(math.ceil((lo[1] - off[1]) / step) * step, hi[1] - off[1], step):
        u = to(lo[0], v + off[1])[1]
        cv2.line(img, (60, u), (S - 20, u), (55, 55, 55), 1)
        cv2.putText(img, f"{v:.0f}", (4, u + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    for k, rg in enumerate(m._regions + ([m._pending] if m._pending else [])):
        c = SM.col(m, k)
        B = rg.get('B') or rg['A']
        r = SM.rect(dict(A=rg['A'], B=B), fw, fh)
        cv2.rectangle(img, to(r[0], r[1]), to(r[2], r[3]), c, 2 if 'B' in rg else 1)
        cv2.putText(img, f"Selection {SM.num(m, k)}", (to(r[0], r[1])[0] + 4, to(r[0], r[1])[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
    for n, k, p in SM.numbered(m):
        q = to(p['x'], p['y'])
        cv2.circle(img, q, 5, SM.col(m, k if k >= 0 else len(m._regions)), -1)
        cv2.putText(img, str(n), (q[0] + 6, q[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    rp = m.real_position
    a, b = to(rp.x - fw / 2, rp.y - fh / 2), to(rp.x + fw / 2, rp.y + fh / 2)
    cv2.rectangle(img, a, b, (0, 220, 255), 2)
    cv2.putText(img, "you are here", (a[0], b[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "X, um (work coordinates) ->", (60, S - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    path, W, H = ag.save_img(img, 'map')
    return dict(image=path, width=W, height=H, note="Axes: work coordinates in um; X grows to the right, Y grows down. "
                "Yellow box = field of view now.", areas=SM.areas_info(m))


def _ask_move(ag, a):
    p = ag.m.position
    return (f"Move the stage by dX {a.get('dx_um', 0):+g}, dY {a.get('dy_um', 0):+g}, dZ {a.get('dz_um', 0):+g} um."
            f"\nFrom X {p.x:.1f}, Y {p.y:.1f}, Z {p.z:.1f} um." +
            ("\n+Z is UP, towards the objective." if a.get('dz_um', 0) > 0 else ""))


@tool('move_stage', "Move the stage by a small step (relative), um. +dz_um = stage UP towards the objective "
                    "(risk of touching it), -dz_um = down. Limits per call: X/Y and Z are in get_status "
                    "max_single_move_um. For far moves use goto_position.",
      {'dx_um': {'type': 'number'}, 'dy_um': {'type': 'number'}, 'dz_um': {'type': 'number'}, 'reason': REASON},
      ('reason',), risk='act', ask=_ask_move)
def _move(ag, reason, dx_um=0.0, dy_um=0.0, dz_um=0.0):
    m = ag.need_m()
    r = m.move(dx_um, dy_um, dz_um)
    p = m.position
    return dict(position_um=dict(x=p.x, y=p.y, z=p.z), note=r.get('Note', ''))


def _ask_goto(ag, a):
    p = ag.m.position
    tgt = ", ".join(f"{k[0].upper()} {a[k]:g}" for k in ('x_um', 'y_um', 'z_um') if a.get(k) is not None)
    up = a.get('z_um') is not None and a['z_um'] > p.z
    return (f"Drive the stage to {tgt} um (work coordinates).\nNow: X {p.x:.1f}, Y {p.y:.1f}, Z {p.z:.1f} um." +
            (f"\nZ goes UP by {a['z_um'] - p.z:.1f} um, towards the objective." if up else ""))


@tool('goto_position', "Drive the stage to absolute work coordinates, um (axes not given stay). Long X/Y moves are "
                       "split into safe legs; Z is lowered first and raised last.",
      {'x_um': {'type': 'number'}, 'y_um': {'type': 'number'}, 'z_um': {'type': 'number'}, 'reason': REASON},
      ('reason',), risk='act', ask=_ask_goto)
def _goto(ag, reason, x_um=None, y_um=None, z_um=None):
    m = ag.need_m()
    if x_um is None and y_um is None and z_um is None:
        raise AgentError('BAD_ARGS', "Give x_um, y_um or z_um.")
    p0 = m.position
    if z_um is not None and z_um < p0.z:
        m.goto(z=z_um)
    if x_um is not None or y_um is not None:
        tx = p0.x if x_um is None else x_um
        ty = p0.y if y_um is None else y_um
        n = max(1, int(np.ceil(np.hypot(tx - p0.x, ty - p0.y) / (m.max_jump * 0.95))))
        for i in range(1, n + 1):
            m.goto(x=p0.x + (tx - p0.x) * i / n, y=p0.y + (ty - p0.y) * i / n)
    while z_um is not None and z_um > m.position.z + 0.5 * m.motor_step:
        m.goto(z=min(z_um, m.position.z + m.max_move_z * 0.95))
    p = m.position
    return dict(position_um=dict(x=p.x, y=p.y, z=p.z))


@tool('autofocus', "Autofocus here: searches Z around the current height (moves Z). wide=true searches a bigger range.",
      {'wide': {'type': 'boolean'}, 'reason': REASON}, ('reason',), risk='act',
      ask=lambda ag, a: f"Autofocus at X {ag.m.position.x:.0f}, Y {ag.m.position.y:.0f} um "
                        f"(Z moves up to {ag.m.focus_range * (2 if a.get('wide') else 1):g} um each way).")
def _focus(ag, reason, wide=False):
    m = ag.need_m()
    r = m.focus(wide=wide)
    return dict(z_um=m.position.z, sharpness=r.get('Sharpness'), note=r.get('Note', 'ok'))


@tool('take_photo', "Save the current camera frame as a file in photos/ (no movement).",
      {'name': {'type': 'string', 'description': "File name without folders, e.g. 'flake_12.png' (optional)."}},
      risk='work')
def _photo(ag, name=None):
    m = ag.need_m()
    if name is not None:
        name = os.path.basename(name)
        if not name.lower().endswith(('.png', '.jpg', '.tif')):
            name += '.png'
    m.photo(name)
    return dict(file=m.last_photo)


# ------------------------------------------------------------------ tools: selections
@tool('area_corner', "Selections, exactly like keys 1 and 2 in manual mode (a selection of a higher objective may lie inside one of a lower objective): corner=1 starts a NEW area at the "
                     "current stage position, corner=2 at the opposite corner fixes it (snapped to the frame grid). "
                     "Z at that moment becomes a focus point, so focus first. Areas must not overlap.",
      {'corner': {'type': 'integer', 'enum': [1, 2]}}, ('corner',), risk='work')
def _corner(ag, corner):
    m = ag.need_m()
    img = m.image() if m.cam else None
    return dict(message=SM.corner(m, corner, img), areas=SM.areas_info(m))


@tool('add_focus_point', "Remember Z here as a focus point of the area under the stage (or of the new area). "
                         "Focus first (autofocus). Points are numbered through all areas: 1..4, 5..8, ...", risk='work')
def _point(ag):
    m = ag.need_m()
    return dict(message=SM.add_point(m), areas=SM.areas_info(m))


@tool('goto_area', "Drive to the frame of selection N nearest to the stage, at its focus height (safe order: down, X/Y, up).",
      {'selection': {'type': 'integer', 'minimum': 1}, 'reason': REASON}, ('selection', 'reason'), risk='act',
      ask=lambda ag, a: f"Drive to the nearest frame of Selection {a['selection']}.")
def _goto_area(ag, selection, reason):
    sample = selection
    m = ag.need_m()
    k = SM.index_of(m, sample)
    if k is None:
        raise AgentError('BAD_ARGS', f"No selection {sample}.", "Selection numbers are in get_status -> areas.")
    msg = SM.goto_area(m, k)
    p = m.position
    return dict(message=msg, position_um=dict(x=p.x, y=p.y, z=p.z))


@tool('delete_area', "Delete selection N (the numbers of the others stay). N of the new area (corner 2 not "
                     "set) cancels it.",
      {'selection': {'type': 'integer', 'minimum': 1}, 'reason': REASON}, ('selection', 'reason'), risk='act',
      ask=lambda ag, a: f"Delete Selection {a['selection']} (made by the operator or the agent).")
def _del_area(ag, selection, reason):
    sample = selection
    m = ag.need_m()
    k = SM.index_of(m, sample)
    if k is None:
        raise AgentError('BAD_ARGS', f"No selection {sample}.", "Selection numbers are in get_status -> areas.")
    return dict(message=SM.delete_area(m, k), areas=SM.areas_info(m))


def _ask_scan(ag, a):
    info = SM.areas_info(ag.m)
    fixed = [x for x in info if x['state'] == 'fixed' and x['objective'] == ag.m.objective]
    n = sum(x['frames'] for x in fixed)
    return (f"Scan {len(fixed)} selection(s) of {ag.m.objective}x, {n} frames in total. "
            "The stage will move by itself for a while.")


@tool('scan', "Shoot all fixed selections of the CURRENT objective (each with Z from its own focus points) into one "
              "new folder; selections of other objectives (layers) are left for when that objective is in place. "
              "dry_run=true only counts frames and time, no approval needed.",
      {'dry_run': {'type': 'boolean'}, 'name': {'type': 'string', 'description': "Short name for the folder."},
       'reason': REASON}, ('reason',), risk='work')
def _scan(ag, reason, dry_run=False, name=''):
    m = ag.need_m()
    SM.init_areas(m)
    info = SM.areas_info(m)
    fixed = [x for x in info if x['state'] == 'fixed' and x['objective'] == m.objective]
    if not fixed:
        raise AgentError('NO_AREA', f"No fixed selection of {m.objective}x.",
                         "Make one with area_corner 1 and 2 (focus at each corner), or switch the objective.")
    n = sum(x['frames'] for x in fixed)
    from . import timing as TM
    fm_ = 'points' if getattr(m, '_use_fp', True) else 'off'
    est = sum(TM.estimate(m, x['frames'], fm_ if x['focus_points'] else 'off')[0] for x in fixed)
    if dry_run:
        return dict(areas=len(fixed), frames=n, about_minutes=round(est / 60, 1),
                    estimate_source=TM.estimate(m, n, fm_)[1],
                    pending_area_ignored=any(x['state'] != 'fixed' for x in info))
    t = _TOOLS['scan']
    ag._rate()
    ag._permit(dict(t, name='scan', ask=_ask_scan), dict(reason=reason))
    folder = m.scan(name=name)
    return dict(folder=os.path.relpath(folder, ag.root), frames=n,
                next="Call stitch with this folder (a session: each sample folder is stitched).")


# ------------------------------------------------------------------ tools: scans on disk
@tool('list_scans', "Scans of the open project (every scan of every selection, a new scan of a selection is a new "
                    "version): folder, frames, stitched or not, grid, chosen squares, hints.")
def _list(ag):
    out = []
    for d in ag._scan_names():
        f = os.path.join(ag.root, d)
        if d.startswith('_'):
            continue
        if os.path.exists(os.path.join(f, 'meta.json')):
            out.append(dict(_brief(ag, f), kind='scan'))
            continue
        ses, subs = SM.session_of(f)
        if ses:
            out.append(dict(folder=d, kind='session', samples=[_brief(ag, s) for s in subs]))
        elif os.path.exists(os.path.join(f, 'meta.json')):
            out.append(dict(_brief(ag, f), kind='scan'))
    return out


def _brief(ag, f):
    from .mask import Grid
    g = Grid(f)
    meta = json.load(open(os.path.join(f, 'meta.json'), encoding='utf-8'))
    return dict(folder=os.path.relpath(f, ag.root), frames=meta.get('frames') or meta.get('n_frames'),
                objective=meta.get('objective'), stitched=os.path.isdir(os.path.join(f, 'join')),
                grid=g.ready(), crosses=len(g.cr.marks), labels=len(g.labels), selected=len(g.selected),
                hints=len(g.useful))


@tool('scan_info', "Everything about one scan: frames, size, stitched, grid (tilt, pitch), labels and numbering, "
                   "selection, hints, exported files, and the suggested next step.", {'folder': FOLDER}, ('folder',))
def _info(ag, folder):
    f = ag.folder(folder)
    g = ag.grid(f)
    b = _brief(ag, f)
    out = dict(b)
    if b['stitched']:
        sc = ag.scene(f)
        out['mosaic_size_px'] = [round(v) for v in sc.size]
        out['mosaic_pixel_um'] = ag.px_um(sc)
    if g.ready():
        v1, _ = g.cr.vectors()
        out['grid'] = dict(tilt_deg=round(g.cr.angle(), 3), pitch_px=round(float(np.hypot(*v1)), 2),
                           fitted_to_all_crosses=bool(g.refined), numbering=g.numbering()[2])
    ex = os.path.join(f, 'export')
    out['exported'] = sorted(os.listdir(ex)) if os.path.isdir(ex) else []
    nxt = ("stitch" if not b['stitched'] else
           "look target=mosaic, then mark_cross on two neighbouring crosses in one row" if not g.ready() else
           "fit_grid (uses all crosses), then label_square on two squares in different rows and columns"
           if not g.labels else "find_hints / select_squares, then export")
    out['next'] = nxt
    return out


@tool('stitch', "Stitch a scan into one mosaic (CPU only, may take minutes). A session folder stitches every sample.",
      {'folder': FOLDER}, ('folder',), risk='work')
def _stitch(ag, folder):
    from .stitch import stitch
    f = ag.folder(folder)
    ag._scenes.pop(f, None)
    r = stitch(f, verbose=False)
    return dict(result=str(r).split("\n")[:12], next="look target=mosaic folder=<sample folder>")


@tool('remove_sample', "Remove one sample folder of a session from the list (moved to <session>/_removed/, "
                       "not deleted).", {'folder': FOLDER, 'reason': REASON}, ('folder', 'reason'), risk='act',
      ask=lambda ag, a: f"Remove sample folder {a['folder']} (it is moved to _removed/, not deleted).")
def _remove(ag, folder, reason):
    f = ag.folder(folder)
    if not SM.is_sample(f):
        raise AgentError('BAD_ARGS', "Only a sample folder of a session can be removed.")
    ag._scenes.pop(f, None)
    return dict(moved_to=SM.remove_sample(f))


# ------------------------------------------------------------------ tools: grid of squares
@tool('mark_cross', "Mark a cross of the lithography grid near the point (x_px, y_px) of the mosaic (from the ruler "
                    "of look target=mosaic); it snaps to the exact centre. Two neighbouring crosses in one row are "
                    "enough for the grid; more crosses far apart make it more exact.",
      {'folder': FOLDER, 'x_px': {'type': 'number'}, 'y_px': {'type': 'number'}}, ('folder', 'x_px', 'y_px'),
      risk='work')
def _mark(ag, folder, x_px, y_px):
    from .mask import snap_cross
    f = ag.folder(folder)
    g, sc = ag.grid(f), ag.scene(f)
    px_um = ag.px_um(sc)
    Mc = np.array([x_px, y_px], float)
    if not (0 <= x_px <= sc.size[0] and 0 <= y_px <= sc.size[1]):
        raise AgentError('BAD_ARGS', f"Point outside the mosaic 0..{sc.size[0]:.0f} x 0..{sc.size[1]:.0f} px.")
    r = int(min(150, max(8, 15.0 / px_um)))
    reg = sc.render(Mc[0], Mc[1], 1.0, 4 * r + 1, 4 * r + 1)
    d, resp = snap_cross(cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY), r, arm=g.arm)
    if resp < 0.05:
        raise AgentError('NO_CROSS', f"No cross found near ({x_px:.0f}, {y_px:.0f}) px (response {resp:.2f}).",
                         "Look closer (look target=mosaic with a small region) and give the centre of a cross.")
    ij = g.cr.add(Mc + d)
    g.refined = None
    g.save(px_um)
    out = dict(cross_px=[round(float(v), 1) for v in Mc + d], snapped_by_px=[round(float(v), 1) for v in d],
               crosses=len(g.cr.marks), lattice_index=list(ij))
    if g.ready():
        out.update(tilt_deg=round(g.cr.angle(), 3), pitch_um=round(float(np.hypot(*g.cr.vectors()[0]) * px_um), 2))
    return out


@tool('undo_cross', "Remove the last cross mark.", {'folder': FOLDER}, ('folder',), risk='work')
def _undo(ag, folder):
    f = ag.folder(folder)
    g = ag.grid(f)
    if not g.cr.marks:
        raise AgentError('NOTHING', "No cross marks.")
    g.cr.undo()
    g.save(ag.px_um(ag.scene(f)))
    return dict(crosses=len(g.cr.marks))


@tool('fit_grid', "Fit the grid to ALL crosses of the mosaic (after two marks): more exact tilt and pitch.",
      {'folder': FOLDER}, ('folder',), risk='work')
def _fit(ag, folder):
    f = ag.folder(folder)
    g, sc = ag.grid(f), ag.scene(f)
    if not g.ready():
        raise AgentError('NO_GRID', "Mark two neighbouring crosses first (mark_cross).")
    px_um = ag.px_um(sc)
    n, rms = g.refine(lambda x, y, w, h: sc.render(x, y, 1.0, w, h), sc.size, px_um)
    g.save(px_um)
    return dict(crosses_used=n, error_px=None if rms is None else round(rms, 2), tilt_deg=round(g.cr.angle(), 3))


@tool('label_square', "Write the real number [X, Y] of the square that contains the mosaic point (x_px, y_px), as "
                      "printed by the lithography next to it. Two labels in different rows and columns fix the "
                      "numbers of all squares. Read the number from a picture; never guess it.",
      {'folder': FOLDER, 'x_px': {'type': 'number'}, 'y_px': {'type': 'number'}, 'square': SQUARE},
      ('folder', 'x_px', 'y_px', 'square'), risk='work')
def _label(ag, folder, x_px, y_px, square):
    f = ag.folder(folder)
    g = ag.grid(f)
    if not g.ready():
        raise AgentError('NO_GRID', "Mark the crosses first.")
    if len(square) != 2 or not all(isinstance(v, int) for v in square):
        raise AgentError('BAD_ARGS', "square is [X, Y], two integers.")
    c = g.cell_at((x_px, y_px))
    g.labels[c] = (int(square[0]), int(square[1]))
    g.save(ag.px_um(ag.scene(f)))
    return dict(lattice_index=list(c), label=square, numbering=g.numbering()[2])


@tool('list_squares', "Squares with their names [X, Y], centre in mosaic px and hint score. which: 'selected', "
                      "'hints' (yellow: something is there) or 'all' (only inside the mosaic).",
      {'folder': FOLDER, 'which': {'type': 'string', 'enum': ['selected', 'hints', 'all']},
       'limit': {'type': 'integer', 'minimum': 1, 'maximum': 2000}, 'offset': {'type': 'integer', 'minimum': 0}},
      ('folder', 'which'))
def _squares(ag, folder, which, limit=200, offset=0):
    f = ag.folder(folder)
    g = ag.grid(f)
    if not g.ready():
        raise AgentError('NO_GRID', "No grid yet: mark two crosses (mark_cross).")
    if which == 'selected':
        cells = sorted(g.selected)
    elif which == 'hints':
        cells = sorted(g.useful, key=lambda c: -g.useful[c])
    else:
        sc = ag.scene(f)
        cells = [c for c in g.cells_range((0, 0), sc.size)
                 if all(0 <= q[0] <= sc.size[0] and 0 <= q[1] <= sc.size[1] for q in g.corners(*c))]
    num = g.numbering()
    sc_ = getattr(g, 'scores', {}) or g.useful
    page = cells[offset:offset + limit]
    return dict(total=len(cells), offset=offset, squares=[
        dict(name=list(g.name(*c, num=num)), centre_px=[round(float(v)) for v in g.cr.point(c[0] + .5, c[1] + .5)],
             selected=c in g.selected, hint=round(float(sc_.get(c, 0.0)), 4)) for c in page],
        more=offset + limit < len(cells))


@tool('select_squares', "Change the selection (green squares, the ones exported). action: 'add' / 'remove' the "
                        "listed squares; 'add_hints' adds all yellow hints; 'clear' and 'set' replace the selection "
                        "(need approval).",
      {'folder': FOLDER, 'action': {'type': 'string', 'enum': ['add', 'remove', 'add_hints', 'set', 'clear']},
       'squares': {'type': 'array', 'items': SQUARE, 'description': "List of [X, Y] names."}, 'reason': REASON},
      ('folder', 'action'), risk='work')
def _select(ag, folder, action, squares=None, reason=''):
    f = ag.folder(folder)
    g = ag.grid(f)
    if not g.ready():
        raise AgentError('NO_GRID', "No grid yet.")
    if action in ('set', 'clear') and g.selected:
        t = _TOOLS['select_squares']
        ag._rate()
        ag._permit(dict(t, ask=lambda ag_, a: f"Replace the selection of {len(g.selected)} squares in "
                                                f"{os.path.basename(f)} ({action})."), dict(reason=reason))
    cells = {ag.square_ij(g, q) for q in (squares or [])}
    if action in ('add', 'remove', 'set') and not cells:
        raise AgentError('BAD_ARGS', "Give squares: [[X, Y], ...].")
    n0 = len(g.selected)
    if action == 'add':
        g.selected |= cells
    elif action == 'remove':
        g.selected -= cells
    elif action == 'add_hints':
        g.selected |= set(g.useful)
    elif action == 'set':
        g.selected = set(cells)
    else:
        g.selected = set()
    g.save(ag.px_um(ag.scene(f)))
    return dict(before=n0, now=len(g.selected))


@tool('find_hints', "Yellow hints: squares where something is (flakes, crystals, any object): anomaly search — "
                    "everything that does not look like the empty substrate. sensitivity 0..100 (more = more "
                    "squares; default 50). Dust is ignored. Check the strongest ones with look before selecting.",
      {'folder': FOLDER, 'sensitivity': {'type': 'number', 'minimum': 0, 'maximum': 100}}, ('folder',), risk='work')
def _hints(ag, folder, sensitivity=None):
    f = ag.folder(folder)
    g, sc = ag.grid(f), ag.scene(f)
    if not g.ready():
        raise AgentError('NO_GRID', "No grid yet.")
    g.find_useful(sc.render, sc.size, sensitivity=sensitivity)
    g.save(ag.px_um(sc))
    num = g.numbering()
    top = sorted(g.useful, key=lambda c: -g.useful[c])[:30]
    return dict(hints=len(g.useful), sensitivity=g.sensitivity,
                strongest=[dict(name=list(g.name(*c, num=num)), score=round(g.useful[c], 4)) for c in top])


@tool('export', "Export only the selected squares: straight PNG + DXF (um) for AutoCAD, into <scan>/export/.",
      {'folder': FOLDER}, ('folder',), risk='work')
def _export(ag, folder):
    from .mask import export
    f = ag.folder(folder)
    r = export(f)
    return dict(result=str(r).split("\n")[:10])


# ------------------------------------------------------------------ Ollama chat
SYSTEM = ("You control a Raman microscope (Optosky ATR8300) through the MicroControl tools. Rules: "
          "1) Call get_status first, read get_manual('overview') once. 2) Never invent numbers, folders, squares or "
          "positions: take them only from tool results or from the operator. 3) Units are micrometres (um); +Z is "
          "UP towards the objective — be careful. 4) One action at a time; check its result before the next. "
          "5) If a tool says DENIED, do not repeat it; ask the operator. 6) If unsure, stop and ask the operator "
          "in plain words. 7) Answer the operator briefly, in the operator's language.")


def chat(m=None, model='qwen3:8b', host='http://127.0.0.1:11434', scans=None, approve='console', vision=False,
         task=None, max_steps=40):
    """Chat with a local model (Ollama) that uses the tools. The model can run on another, stronger PC:
    host='http://192.168.1.20:11434'. vision=True attaches pictures for models that see (gemma, qwen-vl).
    Type the task; empty line or 'exit' quits."""
    import urllib.request
    ag = Agent(m, scans=scans, approve=approve)
    msgs = [{'role': 'system', 'content': SYSTEM}]
    tools = ag.tools('openai')

    def ask(msgs_):
        req = urllib.request.Request(host.rstrip('/') + '/api/chat', method='POST',
                                     data=json.dumps(dict(model=model, messages=msgs_, tools=tools, stream=False)).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode())['message']

    while True:
        text = task if task is not None else input("\nOperator> ").strip()
        task = None
        if not text or text.lower() in ('exit', 'quit', 'q'):
            return ag
        msgs.append({'role': 'user', 'content': text})
        for _ in range(max_steps):
            msg = ask(msgs)
            msgs.append(msg)
            calls = msg.get('tool_calls') or []
            if not calls:
                print(f"\nAgent: {msg.get('content', '').strip()}")
                break
            for c in calls:
                fn = c.get('function', {})
                res = ag.call(fn.get('name'), fn.get('arguments') or {})
                print(f"  tool {fn.get('name')}({json.dumps(fn.get('arguments'), ensure_ascii=False)[:120]}) -> "
                      f"{'ok' if res['ok'] else res['error']['code']}")
                tm = {'role': 'tool', 'content': json.dumps(res, ensure_ascii=False)[:12000], 'tool_name': fn.get('name')}
                if vision and res.get('images'):
                    tm['images'] = [base64.b64encode(open(p, 'rb').read()).decode() for p in res['images']]
                msgs.append(tm)
        else:
            print(f"\nStopped after {max_steps} steps: ask the agent to continue or give a new task.")


# ------------------------------------------------------------------ MCP server (stdio, JSON-RPC 2.0)
def mcp_server(m=None, scans=None, approve='window', inp=None, out=None):
    """Minimal MCP server over stdio: tools/list, tools/call (text + PNG images). For any MCP client
    (ollmcp, LM Studio, Claude Desktop, ...). The operator approves actions in a window on this PC."""
    inp = inp or sys.stdin
    out = out or getattr(sys, '_mc_stdout', None) or sys.stdout
    if out is sys.stdout:                            # всё напечатанное программой — в stderr, stdout только протоколу
        sys.stdout = sys.stderr
    ag = Agent(m, scans=scans, approve=approve)
    manual = _manual(ag, 'overview')

    def send(obj):
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out.flush()

    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            send(dict(jsonrpc='2.0', id=None, error=dict(code=-32700, message="Parse error")))
            continue
        mid, meth, prm = req.get('id'), req.get('method'), req.get('params') or {}
        if mid is None:                                  # уведомление (notifications/initialized и т. п.)
            continue
        if meth == 'initialize':
            res = dict(protocolVersion=prm.get('protocolVersion', '2025-06-18'),
                       capabilities=dict(tools=dict(listChanged=False)),
                       serverInfo=dict(name='microcontrol', version=__import__('microcontrol').__version__),
                       instructions=SYSTEM + "\n\n" + manual)
        elif meth == 'ping':
            res = {}
        elif meth == 'tools/list':
            res = dict(tools=ag.tools('mcp'))
        elif meth == 'tools/call':
            r = ag.call(prm.get('name'), prm.get('arguments') or {})
            content = [dict(type='text', text=json.dumps({k: v for k, v in r.items() if k != 'images'},
                                                         ensure_ascii=False))]
            for p in r.get('images', []):
                content.append(dict(type='image', mimeType='image/png',
                                    data=base64.b64encode(open(p, 'rb').read()).decode()))
            res = dict(content=content, isError=not r['ok'])
        else:
            send(dict(jsonrpc='2.0', id=mid, error=dict(code=-32601, message=f"Method {meth} not found")))
            continue
        send(dict(jsonrpc='2.0', id=mid, result=res))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m microcontrol.agent",
                                 description="MicroControl tools for AI agents: MCP server or Ollama chat.")
    ap.add_argument('--mcp', action='store_true', help="run the MCP server on stdio")
    ap.add_argument('--chat', metavar='MODEL', help="chat with an Ollama model, e.g. qwen3:8b")
    ap.add_argument('--host', default='http://127.0.0.1:11434', help="Ollama address")
    ap.add_argument('--port', default=None, help="microscope port: auto, COM3, sim; none = scans only")
    ap.add_argument('--objective', type=int, default=None)
    ap.add_argument('--scans', default=None, help="scans folder when no microscope is connected")
    ap.add_argument('--approve', default='window', choices=['window', 'console', 'deny'])
    a = ap.parse_args(argv)
    if a.mcp:                                        # stdout — только для MCP: печать программы уходит в stderr
        sys._mc_stdout, sys.stdout = sys.stdout, sys.stderr
    m = None
    if a.port:
        from .core import Microscope
        m = Microscope(a.port, objective=a.objective)
    try:
        if a.mcp:
            mcp_server(m, scans=a.scans, approve=a.approve)
        elif a.chat:
            chat(m, model=a.chat, host=a.host, scans=a.scans, approve='console' if a.approve == 'window' and
                 not sys.stdin.isatty() else a.approve)
        else:
            ap.print_help()
    finally:
        if m is not None:
            m.close()


if __name__ == '__main__':
    main()
