"""MicroControl — control of the Optosky ATR8300 Raman microscope from Python, like a 3D printer.
Authors: Andrey Lyapeikov, Savely Domenko.

    import microcontrol as mc
    m = mc.Microscope()        # connect ('sim' — simulator)
    m.manual()                 # look at the sample, mark the area with keys 1 and 2
    folder = m.scan()          # shoot the area
    m.close()
    mc.stitch(folder)          # exact frame positions from the overlaps
    mc.view(folder)            # crosses, square selection, numbers, hints
    mc.export(folder)          # only the selected squares: PNG + DXF

Help: mc.help()
"""
__version__ = "0.21.0"
AUTHORS = ["Ляпейков Андрей", "Доменко Савелий"]

from .errors import (MicroControlError, SafetyError, LimitError, CommError, CameraError,
                     FocusError, StoreError, GcodeError, StopError, CODES as ERRORS)
from .core import Microscope, ports, imread, imwrite
from .scan import Area
from .gcode import Program
from .storage import ScanData
from .join import join
from .stitch import stitch
from .mask import export, clean
from .optics import measure_distortion as distortion
from .viewer import view, hints, train_hints
from .agent import Agent
from .selections import list_selections as selections, remove_selection, open_selection
from .util import _app_id as _a
_a()                                     # своя кнопка на панели задач Windows (значок MicroControl)
from . import agent

_HELP = """MicroControl {v}. Authors: Lyapeikov A., Domenko S.

CONNECT
  m = mc.Microscope()              Connect (the port is found automatically)
  m = mc.Microscope('COM3')        Connect to a given port
  m = mc.Microscope('sim')         Simulator, no device needed
  m.close()                        Disconnect (light and laser go off)

WORKFLOW
  m.manual()                       Live view on the full screen; mark the area with keys 1 and 2
  folder = m.scan()                Shoot the selections of the current objective (each into selection_01, ...); 30 % overlap
                                   (35 % at 50x, 100x); Z of each area from its own focus points
  mc.stitch(folder)                Exact positions of all frames by their overlaps (like ASHLAR)
  mc.view(folder)                  Mark crosses on the mosaic, straighten it, select the squares you need
  mc.hints(folder, sensitivity=50) Yellow hint without the window: squares with something in them (0-100)
  mc.train_hints(folder, default=True)  Model of the empty substrate for the hints (scan of a clean substrate)
  ag = mc.Agent(m)                 Tools for AI agents (ag.call('get_status')); mc.agent.chat(m, model='qwen3:8b')
  python -m microcontrol.agent --mcp --port COM3   MCP server for local models; actions need the operator's Allow
  mc.distortion(folder)            Measure the lens distortion by the crosses (once per objective)
  mc.export(folder)                Only the selected squares: PNG/JPG + DXF for AutoCAD
  mc.clean(folder)                 Delete the frames and the rest of the work files (shows first)
  mc.join(folder)                  Simple whole mosaic in one picture (small scans)
  m.list()                         Scanned selections as layers (map 10x > details 50x > 100x), numbered
  m.open(N)                        Look at selection N again (mc.view)
  m.remove(N)                      Delete selection N completely (asks; confirm=False: no question)
  m.info()                         Device, and where the time goes: every action timed, longest first

MANUAL WINDOW
  Arrows / WASD    Move by Step X / Step Y  Tab     Edit the steps (or click a field)
  E / Q            Stage up / down by Step Z (Shift: x5)
  G                Steps X, Y = one frame   F       Keep Z here as a focus point (Shift+F: autofocus)
  Y                Autofocus by corners during the scan (on / off)
  M                Next objective (4x, 10x, 20x, 100x)
  1 / 2            Corners of a selection, one after another; a higher objective may mark inside
                   a lower one (layers: map 10x > details 50x > 100x), one objective does not overlap
  Carousel, , / .  Selections as layers; Go / Enter: drive there; x or C (twice): delete it
  Space            Picture                  Shift+L Light on / off
  L                Ruler: click two points -> distance in um (also in mc.view)
  0                Zero here (offset)       R       Real coordinates (remove the offset)
  B                White balance
  Wheel, + / -     Zoom (Z: whole frame)    Drag    Move the zoomed picture (any button)
  Ctrl+Z / Ctrl+Y  Undo / redo any step (selections, focus points, moves: the stage drives back)
  H                Help on the screen       Esc     Exit

VIEW WINDOW (mc.view)
  Drag             Move (right / middle button in any mode; Ctrl+left in S and B)   Wheel, + / -, 0   Zoom
  G                Two crosses in a row: the tilt; the picture turns level (far crosses: more exact; C: again)
  S / B / P        Select squares: click and box / brush / polygon (Shift: unselect)   A / X  All / none
  Right click      On a green area: remove it
  N                Number a square: click it, type X Y (two squares in another row and column fix all numbers)
  Y                Yellow hint: squares with something in them   Shift+Y / Ctrl+Y  Select the yellow ones
  U, < / >         Hint sensitivity 0-100 (type it / -5 / +5)    R  Straight    E  Export

STAGE (micrometres; +Z = stage up, towards the objective; moves go one after another)
  m.move(dx, dy, dz)               Move by (X/Y up to max_move, Z up to max_move_z)
  m.goto(x, y, z)                  Go to a point (a trip longer than max_jump is refused)
  m.position                       Where the stage is (working coordinates)
  m.real_position                  Real coordinates (from home; without home - from the connection point)
  m.offset()                       The current point becomes (0, 0, 0); real coordinates stay
  m.offset_clear()                 Remove the offset
  m.measure_speed()                Measure the motor speed with the camera, set m.motor_speed
  m.confirm_position()             After a new session, a limit switch or a lost reply: "the position is right"
  m.find_limits()                  Find the limit switches (needs allow_homing=True)
  m.limits                         Travel limits of X, Y, Z
  m.set_limits(z=(-50000, 50000))  Change the travel limits (working coordinates, um)
  m.objective = 20                 Change the objective (4, 10, 20, 100): pixel size, focus window, Z step
  m.turn_180 = True                Picture upside down (on by default on ATR8300, remembered)
  m.check_orientation()            Check with the camera which way the picture moves (done before every scan)
  m.swap_xy = True                 Camera turned by 90 deg (on by default on ATR8300, remembered)

HOME OF THIS MICROSCOPE (remembered by its serial number, between sessions)
  m.set_home()                     Once: this point becomes real (0, 0, 0); a photo of it is kept
  m.go_home()                      Go to X 0, Y 0 and compare the picture with the home photo
  m.go_home(correct=True)          ... and fix the coordinates if the picture is shifted
  m.go_home(z=True)                ... and Z to 0 too (up in small steps, never above the ceiling)

CAMERA, LIGHT, FOCUS
  m.photo()                        Picture to photos/
  m.image()                        Current frame as an array, no file
  m.sharpness                      Sharpness of the current frame
  m.focus()                        Autofocus: narrow window (±4 depths of field) around the expected Z
  m.focus(wide=True)               Wide search, if the focus is lost
  m.depth_of_field                 Depth of field of the objective, um
  m.dark_frame()                   Dark frame for a dataset (light off)
  m.empty_frame()                  Frame of an empty place for a dataset (uneven light)
  m.light = 5   /   m.light = 0    Light
  m.laser = 5   /   m.laser = 0    Laser, mW (needs Microscope(..., allow_laser=True))

SETTINGS (read and change like variables; wrong values are refused)
  m.settings                       All settings with allowed values
  m.photo_pause                    Read one
  m.photo_pause = 0.5              Change one
  Old names (dwell, settle, ...) give an error with the new name.

G-CODE (like a 3D printer; units are micrometres)
  m.gcode("G0 X500 Y200")          Run commands or a .gcode file
  prog = m.plan()                  Scan program for the area: look, edit, run with m.gcode(prog, save='name')
  G0/G1 X Y Z   G53 G0 (real coordinates)   G90/G91   G92 / G92.1 offset   G28   G4 S|P
  G30 autofocus   G29 focus plane   M420 S0/S1
  M240 picture   M355 S1 P5 / S0 light   M3 S<mW> / M5 laser   M114 position

EMERGENCY STOP (from any command line, even if the notebook hangs)
  python -m microcontrol stop      (no rights to free the port: run it in an administrator console)
  python -m microcontrol status    Who controls the microscope now
  python -m microcontrol probe     Check the link: read-only requests, all bytes printed

INFO
  m                                State of the microscope
  m.info()                         Model, serial number, device type; time spent per action
  mc.ERRORS                        Error codes
  help(m.scan)                     Help on any command
"""


def __getattr__(name):
    if name == 'TileStore':
        raise AttributeError("'TileStore' is now called 'ScanData': mc.ScanData(folder)")
    raise AttributeError(f"module 'microcontrol' has no {name!r}")


def help():
    """Short MicroControl help."""
    print(_HELP.format(v=__version__))


__all__ = ["Microscope", "Agent", "ports", "Area", "Program", "ScanData", "join", "view", "hints", "train_hints", "help", "imread", "imwrite",
           "MicroControlError", "SafetyError", "LimitError", "CommError", "CameraError", "FocusError",
           "StoreError", "GcodeError", "StopError", "ERRORS"]
