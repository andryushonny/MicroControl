# MicroControl

**Python control of the Optosky ATR8300 Raman microscope: scan a substrate, stitch one picture of it, put the lithography grid on it, choose the squares you need and export them for AutoCAD.**

**Authors:** A. Lyapeikov, S. Domenko · **Version:** 0.20.0 · Русский: [docs/README_ru.md](docs/README_ru.md) · 中文: [docs/README_zh.md](docs/README_zh.md) · PDF: [EN](docs/MicroControl_en.pdf), [RU](docs/MicroControl_ru.pdf), [中文](docs/MicroControl_zh.pdf) · Full technical reference: [EN](docs/reference_en.md), [RU](docs/reference_ru.md), [中文](docs/reference_zh.md)

![Overview: numbered lithography squares, chosen squares as drafts, the 20x and 100x areas inside, the status bar at the bottom](docs/img/view.jpg)

> Independent project, not affiliated with Optosky. Provided as is, without warranty — see the [Disclaimer](#disclaimer) and the [MIT licence](LICENSE).

---

## Quick start

Five steps, from a sample on the stage to a drawing for AutoCAD:

```python
import microcontrol as mc

s = mc.Project("Graphene", "flakes for transport").sample("Chip 1", "SiO2 285 nm")   # 0. project and sample
m = mc.Microscope(objective=10, sample=s)   # 1. connect   (mc.Microscope('sim', objective=10) — simulator)

m.manual()                   # 2. green area: keys 1 and 2 at two opposite corners, focus at each
folder = m.scan()            #    shoot the area frame by frame (or S right in the window)
mc.stitch(folder)            #    join the frames into one picture (or S in mc.view)

mc.view(folder)              # 3. key C: click two neighbouring crosses -> the red grid
                             # 4. keys M / B: choose squares (green)
mc.export(folder)            # 5. the chosen squares: pictures + a drawing in micrometres (DXF / DWG)

m.close()
```

Both windows close with **Shift+Esc** or the window's ×. A single `Esc` only cancels. **H** shows all keys.

---

## The five steps

### 1. Connect

`m = mc.Microscope()` finds the controller port and the camera by itself, switches the light on and checks that the stage answers. All coordinates are in micrometres, $+Z$ is the stage going **up**, towards the objective. Every move is checked against limits (one step, the whole travel, a ceiling for $Z$), so a typo cannot drive the sample into the objective. `mc.Microscope('sim')` is a simulator: the same commands without the instrument.

### 2. Green area — what to scan

![Manual mode](docs/img/manual.jpg)

`m.manual()` opens the live camera. Move with the arrows (one press = one frame), focus with `E` / `Q` (or `F` — autofocus). Letters do things: `L` light, `I` laser, `W` white balance, `R` ruler, `S` (twice) scan and stitch right in the window, `V` open the result in the viewer.

- **One window, four steps.** The bar on top shows where you are — **1 Overview › 2 Pick areas › 3 Detail › 4 Export** (click a step for what to do) — and the main buttons: **New area**, **Scan area** (scan + stitch), **Export** (drawing for AutoCAD: DWG + DXF of the chosen area), **Settings** (steps, exposure, focus points).
- **Map** (top right; `Tab` — big map, the camera goes to the corner): everything already shot, at its real place (10x under 50x under 100x), the outlines of the areas and a yellow frame where the camera is now. **Click the map — the stage drives there**; **drag on the map — a new area** on the objective in place; **`Ctrl`+drag — a free box** exactly as drawn (around a crystal), not snapped to the squares; wheel — zoom; right drag — move the map; double-click a shot area — open it in the viewer.
- **Selection by keys** still works: go to one corner, press `1`, to the opposite one, press `2`.
- **Focus points:** the height $Z$ at each corner is remembered; `K` adds more points. During the scan every frame gets its own $Z$ from these points, so a tilted sample stays sharp. With the checkbox "Z follows the focus points" the arrows also follow them.
- **Several selections, layers:** set them one after another (`1`, `2`, then `1`, `2` again). Selections of one objective do not overlap, but a selection of a higher objective may lie inside one of a lower objective (see *Layers* below). The list on the right shows them nested: click — choose, **Go** — drive to it, **x** — delete.
- **Ruler:** `R`, then click two points — the distance in µm (the stage may move between the clicks).
- **Undo:** `Ctrl+Z` undoes any step (a selection, a focus point, a move — the stage drives back), `Ctrl+Y` redoes it.

`m.scan()` shoots every area frame by frame with 30 % overlap (35 % at 50x, 100x) and saves the frames into a new folder (one sub-folder per selection); `m.scan(name='test')` again writes into the same folder. Selections keep their numbers and colours when one is deleted. Only the selections of the objective in place are shot. It prints the expected time; this estimate learns from the previous scans of your microscope. `mc.stitch(folder)` measures from the overlaps where each frame really is and joins them into one seamless picture.

### 3. Red grid — find the squares

![Grid, numbers and chosen squares](docs/img/view_zoom.jpg)

`mc.view(folder)` opens the stitched picture (drag — move, wheel — zoom).

- Press `C` and click **two neighbouring crosses in one row**. From them the program knows the tilt and the size of the squares, turns the picture level and draws the red grid over everything. The mark lands exactly where you click; `Shift`+click snaps it to the cross centre, a mark can be dragged. The grid is fitted to all crosses at once; `F` then puts every frame onto the crosses (see *Accuracy for lithography*).
- **Numbers:** press `N`, click a square and type the number written next to it (`12 7`). Do this for two squares in different rows and columns — then every square has its real number.
- **The grid is always the grid of what is selected in the menu on the right.** Its own crosses if they are marked in its scan; otherwise the crosses of the larger area around it (50x inside 10x); otherwise the sample lattice. The red grid on the map, the squares of a new area drawn inside it and the grid of its next scan all follow this. `Shift+C` clears the crosses (then just click new ones); cleared crosses are no longer used anywhere.

### 4. Choose squares — the mask

- `M` — select: click a square (again — off) or drag a box, every square it touches is chosen (`Shift` — unselect). Right click — remove one square, `Shift` + right click — the whole green area. Chosen squares are green (`B` does the same, out of habit).

### 5. Export

`E` in the viewer or `mc.export(folder)` writes `export/` inside the scan folder: a straight PNG of the chosen squares (transparent background), a DXF for AutoCAD in micrometres and a list of the square numbers.

The final result is the drawing `export/drawing.dwg`: every chosen square is its own picture (`export/squares/*.png`, cut exactly along the grid) placed at its real position in micrometres, with the square outlines and numbers on separate layers and an empty `BONDS` layer for drawing bonds. The DWG is written by the free ODA File Converter if it is installed; otherwise there is `drawing.dxf` (AutoCAD: Save As → DWG). Keep the `export/` folder together: pictures in DWG are always separate files. The progress is shown right in the window.

`mc.projects()` and `mc.all_samples()` print a table: id, name, samples, selections, scans, created, updated, description (the Open dialogs show the same). In the viewer, Open → **New window** shows another project in its own window (`mc.view_window(folder)`), the current one stays open.

Squares selected in the viewer become a **Draft** — a sketch that is never scanned (a helper: it never goes into the export; its *draft* chip or **Scan area** turns it into a Selection of the objective in place, with no way back). Over any selection, keys `2`–`5` give it its own grid 2–5 times denser; `M` then selects cells of that grid strictly inside it, and they become drafts one level deeper (address `13,7 > 2,1`). On the map, drag over such a selection: the new area for the objective in place is made of whole cells of its grid. **Export** asks which areas go into the drawing and writes one AutoCAD drawing of the sample: origin at cross 1,1, pictures in place, every cell as a closed square, crosses with their numbers. In the viewer what you chose stays clear and the rest is tinted; a smaller area is always cut to the edge of the larger one it lies in (a pyramid: 10x → 50x → 100x), and the 50x / 100x pictures lie on top of the overview.

| | |
|---|---|
| ![Big map (Tab): 10x overview with drafts, 50x and 100x inside, the red grid of the selected area, the camera in the corner](docs/img/map_zoom.jpg) | ![The 50x scan: its grid comes from the sample lattice — tilt and pitch match the overview, no crosses marked](docs/img/view_50x.jpg) |
| *Layers on the map: 10x → Draft (grid 1/4) → 50x → 100x, the camera in the corner* | *50x scan: the grid comes from the crosses marked before — its own, else the 10x around it (no crosses to mark), 1/4 and 1/8 cells* |
| ![Export of the sample: the tilted scan is drawn square to the lattice, crosses with numbers from 1,1](docs/img/export.jpg) | ![Export, close-up: 10x picture, Draft 2 squares, 50x cells 1/4 and 100x cells 1/8 with their addresses](docs/img/export_zoom.jpg) |
| *Drawing for AutoCAD: the 2.5° tilted chip comes out square to the lattice, crosses from 1,1* | *Close-up: every cell is a closed square on its own layer, address 16,10 > 2,3 > 1,1* |

## Accuracy for lithography

![Profiles (P): two lines across the arms of a cross, R, G, B along them, the edges in yellow on the map, in the preview and on the plots](docs/img/profiles.jpg)

The drawing is used to write a circuit over the flakes, so every picture in it is placed **by the crosses around it**, not by the stage. What matters, and what the program does about it:

| What | How |
|---|---|
| Stitching in two steps | `S` — frames by their overlaps (the preview). Then `C` (2–3 crosses) and `F` — every frame is put onto the crosses: the frames, the square lattice and the lens distortion are fitted together, so seam errors do not add up. The status line shows how many crosses lie within 0.5 µm of the grid. Python: `mc.stitch(folder)`, then `mc.fit_crosses(folder)` |
| Picture in the drawing | every pixel is taken through the **4 crosses of its own square** (bilinear), crosses measured on that very scan; what is left of stitching, distortion and tilt does not reach the drawing |
| Cross centre | template + **edges of the arms**: two straight lines through the middles of the arms, their crossing is the centre (≈0.1 px, also when the chip is turned; digits next to the cross are ignored) |
| Lens distortion | `D` in the microscope window (`m.calibrate_distortion()`): 3×3 pictures with 50 % overlap around any details; matched pieces of the overlaps give $k_1, k_2, p_1, p_2$ without crosses and without trusting the stage. Remembered for the objective, used for every scan |
| Frames | full camera resolution, JPG 95 by default; `m.frame_format = 'png'` — lossless. Crosses and seams are measured in the green channel (sharpest, no colour fringes) |
| Before a scan | button **Prepare** (or `A`): a step-by-step guide over the live picture — each step says what to set or where to drive, then **Done** / Skip / Cancel: lamp and room → even light (Köhler) → **dark frame** → empty place: exposure (brightest ≈80 %, no overexposure), white balance, **flat field** (5 places, median: vignetting and dust on the optics) → a detail: autofocus, settling pause → lens distortion (once per objective). Each step says how often to repeat it. Saved for the microscope and objective; next time it asks: **Overwrite** or **Keep** (with the date). The stitch uses this flat field and dark frame. Python: `m.prepare()`, `m.flat_field()`, `m.dark_frame()` |
| Scale | `m.mask_pitch = 100` — the design pitch of the crosses on the mask: the drawing takes every size from it. `F` also measures the real pixel size by the crosses (and X/Y stretch, skew) and saves it for the objective: the ruler and the next scans use it |
| Autofocus | contrast (normalised variance) to find the peak, Vollath F4 for the exact pass, a Gaussian fitted to the sharpness curve; step and window are set for each objective in Settings (`AF step`, `AF window`, 0 = from the depth of field). Note: one Z motor step is 5.56 µm — more than the depth of field at 50x / 100x |
| Backlash | `B` — one-side approach (remembered). The stitch does not trust the stage anyway; this keeps the frames where they are expected and the camera overlay exact |

For the circuit itself, outline the flakes at 50x / 100x: at 10x the optics blur edges by about 1 µm.

---

## Projects and samples

Work is grouped like this: **Project** (a piece of work, with a description) → **Sample** (one substrate on the stage: its own selections and coordinates, a description) → **Selection** (an area on one objective; layers: map 10x > details 50x > 100x) → **Scan** (every scan of a selection is a new version) → grid, chosen squares, drawing. Selections are saved in the sample and come back when it is opened again; the chosen squares on the maps, `m.list()` and the viewer show only the open sample. Selections are layers: an outline made of rectangles (no inner squares), a level of the pyramid (objective), nesting; Ctrl+click cards, then Merge / Minus / Overlap / Group / Ungroup. Squares chosen with the brush in the viewer become selections by themselves (one per figure, on the next objective); a selection is scanned only by the frames that touch its outline; the eye on a card hides it on the maps.

```python
p = mc.Project("Graphene", "flakes for transport")   # open or create; p.uid — its short id, e.g. "k3f9x"
p = mc.Project("k3f9x")                              # open an existing project by its id
s = p.sample("Chip 1", "SiO2 285 nm")                # a sample of the project: open or create
m.sample = s                                        # work with it (its selections come back)
p.samples, s.selections, s.selections[0].scans, s.selections[0].latest.view()
mc.projects(), mc.all_samples()
```

Both windows have a menu on top: **Open** (a sample of any project), **New** (choose Project or Sample from a list), **Info** (descriptions), **Scans** (every version), **View**, **Help**. Windows open as normal windows; `m.manual(fullscreen=True)` — the whole screen.

## Layers: a map and the details

Once the crosses are marked, the map in manual mode shows the lithography grid as thin semi-transparent red lines — new areas are made of exactly these squares. The selection chosen on the right adds its own finer grid (2–5) inside it.

1. Mark a large selection with a small objective (10x) and scan it: this is the map.
2. Look at the map (`mc.view`) and find the interesting places.
3. Put a larger objective in place (`M` in manual mode, or `m.objective = 50`), mark small selections inside the map and scan again — only the 50x selections are shot. The same again with 100x inside the 50x ones.

The selections nest like Russian dolls: in the list on the right of manual mode every selection lies inside the frame of the one it was marked in, with its objective (`10x`, `50x`, `100x`) next to the name; the frames and the time are counted for that objective.

```python
m.list()          # all scanned selections as layers, numbered
m.open(3)         # look at number 3 again (mc.view)
m.remove(3)       # delete number 3 completely (no question)
m.info()          # where the time goes: every action is timed, longest first
```

## Ways to control

The same microscope can be driven in six ways; they can be mixed in one session.

| Way | For what | Example |
|---|---|---|
| **Python / Jupyter** | everything, scripts, repeatable work | `m.move(dx=200)`, `m.goto(x=0, y=0)`, `m.focus()`, `m.photo()`, `m.light = 3`, `m.scan()` |
| **Manual window** | looking, finding the sample, setting areas | `m.manual()`: arrows, keys, mouse (click a field to type a step, wheel — zoom, click the list on the right) |
| **Viewer window** | grid, numbers, choosing squares, export | `mc.view(folder)`: mouse and keys |
| **G-code** | programs of moves and pictures, like a 3D printer | `m.gcode("G0 X500 Y200\nM240")`; `m.plan()` writes the scan program, you can edit it |
| **Command line** | checks and emergency stop from another window | `python -m microcontrol probe` · `status` · `stop` |
| **AI agent** | a local language model works through tools, the operator allows each action | `mc.agent.chat(m, model='qwen3:8b')`, MCP server |

Settings are ordinary properties with checked ranges: `m.settings` lists all of them, `m.photo_pause = 0.5` changes one, a value out of range is refused with the allowed range. `mc.help()` prints a short guide.

**Emergency stop:** `python -m microcontrol stop` in any console (or the file `~/.microcontrol/STOP`): a running program sends "light 0" and "laser 0", closes the port and stops. If the program crashes or is killed without closing the instrument, a separate guard process switches the laser, motors, light and cooling off by itself.

---

## How commands reach the instrument

```
Python: your command
   │  checks: step limits, Z ceiling, allowed commands
   ▼
frame:  AA 55 | length | command code | data | checksum
   │  USB virtual COM port, 115 200 bit/s
   ▼
controller ──► motors, light, laser

camera ──► MindVision SDK over USB ──► frames (numpy arrays)
```

- **Link:** a virtual serial port, 115 200 bit/s, 8 data bits, no parity, 1 stop bit — the same settings as the original program.
- **Frame:** every command and every reply is a short packet `AA 55 | length (2 bytes) | command code | data | checksum`; the checksum is the sum of the bytes modulo 256. Example — "serial number?": `AA 55 00 04 04 08`.
- **Integers only:** a move is sent as a direction byte and a number of motor steps (one step ≈ 5.56 µm), so micrometres are converted to whole steps and the rounding is reported. Light and laser are whole numbers too.
- **Allow-list:** only the commands the original program uses can be sent; writing calibration or changing the instrument type is blocked.
- **Timing:** the controller answers within a few milliseconds, before the motion ends, so the program waits distance ÷ speed plus a settling pause; a "limit switch" answer blocks that direction.

The whole protocol, step by step: [reference, section 5](docs/reference_en.md#5-controller-protocol).

---

## How each part works

| Part | The idea in short |
|---|---|
| **Connection** | The controller is talked to over the serial port with the same commands as the original program uses (the protocol was obtained by reverse engineering). Only one program can hold the port; an emergency stop file stops everything at once. |
| **Safety** | Every move is checked before it is sent: the step size, the stage range and the ceiling for $Z$. A move that looks wrong is refused with an error code and a hint. The laser is locked unless you allow it when connecting. |
| **Manual mode** | Live camera with a mini map in whole frames. Areas and focus points are kept in the microscope's real coordinates, so they stay correct when you move or change the zero. |
| **Scan** | A program of moves and pictures (G-code, can be read and edited): frame centres on a grid with overlap, $Z$ of each frame from the focus points, a pause for the stage to settle, then the picture. Optionally (`m.refocus = True`) blurry frames are focused again and re-taken. |
| **Expected time** | A small learning model: a physical estimate at first, then the real times of your scans (see [Machine learning inside](#machine-learning-inside)). |
| **Stitching** | Neighbouring frames overlap; comparing the overlaps gives their exact shifts. All shifts are solved together, so stage errors do not add up. The uneven light of the lens (darker, differently tinted edges) is measured from the frames themselves for each colour and divided out; the brightness of every frame is then matched to its neighbours over the overlaps, so the seams do not show. Built for hundreds and thousands of frames and inspired by the ideas of BigStitcher and MIST: shifts to a fraction of a pixel regardless of the stage error, a stage model with backlash that rejects false matches (repeating crosses on a blurred frame), a sparse solver (thousands of frames in a fraction of a second); in the viewer every seam can be switched off and the frames are placed again at once. |
| **Grid** | Two crosses give the direction and the length of one square side; the grid is square, so that is all it needs. Fitting to all crosses makes it exact. Two labelled squares give the numbering of the whole grid. |
| **Stop** | Every long job has a Stop button (or `Esc`) on its progress bar; the bar goes by the stages of the job, and the status bar at the bottom always says what is going on. |
| **Export** | Only the chosen squares, straightened, at full resolution, plus a DXF with the squares in micrometres. |
| **Selections and layers** | Each selection gets its own colour, objective, focus points and sub-folder; a selection of a higher objective inside one of a lower objective is its detail (a layer). |
| **AI agents** | A local language model can use the whole program through tools; everything that moves the stage needs the operator's "Allow". See below. |
| **Simulator** | A virtual substrate with crosses, flakes and tilt, so everything can be tried without the instrument. |

---

## Machine learning inside

### Expected time of an experiment

The program predicts how long a scan will take and **learns from every finished scan** of your microscope.

1. **Prior (physics).** Before there is any data: time per frame = move (frame step ÷ motor speed) + settling pause + pause before the picture + exposure + writing the file; plus a fixed start-up time.
2. **Data.** After each finished scan the pair (number of frames $N$, real time $T$) is stored in the memory file of this microscope, separately for each combination of settings that changes the speed (objective, focus mode, pauses, exposure, checks, frame format). The last 20 scans are kept.
3. **Model.** A straight line is fitted by least squares (linear regression):

```math
T = T_0 + t\,N,
```

   $T_0$ — preparation, $t$ — seconds per frame (both kept non-negative; with only one value of $N$ the line goes through zero).
4. **Prior + data.** The prediction is a weighted mix of the physics and the fitted line; the weight of the data grows with the number of scans $n$:

```math
\hat T = (1-w)\,T_\text{physics} + w\,T_\text{fit},\qquad w = \frac{n}{n+2}
```

   (1 scan — 33 %, 4 — 67 %, 10 — 83 %), like a Bayesian estimate that trusts experience more and more.
5. **During the scan** the "Left" time starts from this prediction and moves to the real pace as frames come.

---

## Keys

Arrows move; letters do things. As much as possible is done right in the window.

**Manual mode** (`m.manual()`)

| Key | Action |
|---|---|
| Arrows | move by one frame (Step X, Y) |
| `E` / `Q` (`PgUp` / `PgDn`) | stage up / down by Step Z (`Shift` — ×5) |
| `F` · `K` | autofocus · keep $Z$ here as a focus point |
| Ctrl+click cards · Shift+drag over a selection | mark for Merge / Minus / Overlap / Group / Ungroup · move a selection |
| Ctrl+drag on the map · `Esc` during a job | a draft box on the objective in place (a crystal): with a margin for backlash, edges on motor steps; Scan area shoots it; the overview chosen + Scan area — all such boxes in it · stop the job (or Stop on its bar) |
| `1`, `2` | corners of a selection |
| `Y` | "Z follows the focus points" on / off |
| `,` `.` · `Enter` | previous / next selection · drive to it |
| `C` | cancel the new selection / delete the chosen one (Ctrl+Z brings it back) |
| `S` | scan the selections of this objective and stitch them, in the window (`Esc` — stop) |
| `V` | open the last scan in the viewer; closing it brings you back |
| `L` · `I` · `W` | light on / off · laser on / off · white balance |
| `A` · `D` | Prepare step by step (as the button) · distortion of this objective |
| `B` · `\` | one-side approach against backlash · turn the map 90° anticlockwise (both remembered) |
| `R` | ruler: two clicks — the distance |
| `Space` | take a picture |
| `0` · `Shift+0` | zero here · real coordinates |
| `Ctrl+Z` · `Ctrl+Y` | undo · redo (moves too: the stage drives back) |
| `Tab` or click a field | change steps, exposure, objective |
| `G` · `M` | steps = one frame · next objective |
| `J` · `K` | edit the stitch by hand: seams are dots coloured by their error (green < 0.5 px, yellow < 1.5 px, red; grey — off), a click on a dot switches the seam off or on and the frames are placed again at once from the others; click a frame — its row (Shift+click — its column), arrows move it by 1 px (Shift — 10 px), Backspace — back to the automatic stitch, `J` — save (the automatic one stays in `join/positions_auto.csv`, the seams switched by hand — in `join/seams_edit.json`) · straight edges: the steps at the borders are cut off (again — the whole mosaic) |
| `P` | profiles window (as Plot Profile in ImageJ, Profile in Gwyddion): two lines, cyan and pink, with R, G, B along them averaged over a band. On a cross they lie across its arms and cross each other. Drag the squares at their ends on the map — any tilt and length (`Shift` — 15° steps), the `x` in the middle — move a line, the diamond — both. Click — pin / unpin. In the window: a zoomed preview with the lines and edges, width in µm, contrast, angles; drag the yellow edges; wheel on a plot — Y scale (the same for both, click `Y` — auto), elsewhere — band width; right click — auto edges. The edges are drawn as ticks on the map, in the preview and on the map of the manual mode |
| `T` · `O` · `[` `]` | turn the camera picture 180° · check directions with the camera · arrows (and the mini map with them) the other way (all remembered) |
| `H` · `Shift+Esc` | all keys · exit |

**Viewer** (`mc.view(folder)`)

| Key | Action |
|---|---|
| arrows, drag · wheel · `0` | move · zoom · whole picture |
| `S` | stitch right here (frames by their overlaps, the preview); the view reopens stitched |
| `C` · `F` | mark crosses (`Shift+C` — start again) · frames onto the crosses (exact: frames, grid and distortion together) |
| `N` | number a square |
| `M` | select (click, box); `Shift` — unselect; right click — remove one, `Shift` + right click — the whole area; `A` / `X` — all / none |
| `2`…`5` · `1` | over a selection: its own grid 2–5 times denser (`M` then selects its cells) · no finer grid |
| `T` | picture level / as shot |
| `P` | profiles window: R, G, B across the arms of the cross under the mouse, arm edges (green), width in µm, contrast, overexposure. Click a cross — pin it; drag the yellow edge lines in the window — they move on the map too; wheel over the window — profile range; right click — edges found automatically. Cross search prefers the largest «+», so dust and dots do not pull it |
| `R` | ruler: two clicks — the distance |
| `E` | export: square pictures + the drawing (DWG / DXF), progress in the window |
| `,` `.` | previous / next selection |
| `Ctrl+Z` · `Ctrl+Y` | undo · redo any change |
| `Esc` · `H` · `Shift+Esc` | leave the mode · all keys · exit |

---

## Where things are saved

```
projects/<project>/project.json                       name, description
projects/<project>/samples/<sample>/sample.json       its selections (layers, focus points), description
    selections/S01_10x/scans/<date>/                  every scan of a selection is a new version:
        frames/  join/  grid.json  export/            frames, stitched picture, crosses and squares, drawing
    photos/                                           single pictures (Space)
logs/                                                 log of every session
```

---

## AI agents

A local model (for example through Ollama) can drive the microscope: look through the camera and at mosaics, move, scan, stitch, mark crosses, number and choose squares, export.

```python
mc.agent.chat(m, model='qwen3:8b')                 # chat with a local model
python -m microcontrol.agent --mcp --port COM3     # or a server for any MCP client
```

Every move, scan or deletion opens a window: **Allow**, **Allow for this session** or **Deny**. The laser is never available to the agent, the safety limits stay in force, and every call is logged. Details: [reference, section 25](docs/reference_en.md#25-ai-agents).

---

## Installation

Python 3.10–3.14 (3.13 recommended), Windows.

```
cd /d D:\MicroControl
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt jupyter
jupyter notebook
```

The camera needs the MindVision SDK from the instrument's driver kit. Close the original Optosky program before connecting: only one program can use the port and the camera. Examples: `example.ipynb` and the `demo/` folder.

**First start:** check the link with `python -m microcontrol probe`, then in `m.manual()` check the directions — `Q` must move the stage **down**, the arrow → must move the picture to the left. If not, see the [reference, section 20](docs/reference_en.md#20-first-start-on-the-instrument).

**Self-test on the simulator** (no instrument needed): `python test_sim.py --quick` — about 10 s; `python test_sim.py` — everything, the parts run in parallel (about 2 min on 2 cores).

## If something goes wrong

| Problem | What to do |
|---|---|
| The port is busy (`E102`) | close the Optosky program; `python -m microcontrol status`, then `python -m microcontrol stop` |
| No answer from the controller (`E301`) | reconnect the controller USB, run `python -m microcontrol probe` |
| The camera does not open (`E505`) | close other camera programs, reconnect the camera USB; or connect with `camera=False` |
| A cell hangs | `python -m microcontrol stop` in another window, or restart the kernel |
| Smeared frames | `m.move_pause = 0.5`, `m.photo_pause = 0.6` |

All error codes, the protocol, the formulas and every parameter: [full technical reference](docs/reference_en.md).

---

## Disclaimer

**No affiliation.** MicroControl is an independent project. It is not made, endorsed, supported or certified by Optosky Photonics or by MindVision. Optosky, ATR8300, MindVision and other names are trademarks of their owners and are used here only to say which equipment the software works with. The controller protocol was obtained by reverse engineering and only by reverse engineering: by observing the exchange between the stock software and the instrument over the serial port and the names exported by the stock library (OptoSky.dll), solely to make this program work with the instrument. No documentation, source code or other materials of Optosky were used; no Optosky source code is included.

**No warranty.** The software is provided "as is", without any warranty, express or implied, including warranties of merchantability, fitness for a particular purpose, accuracy of results and non-infringement (see [LICENSE](LICENSE)).

**Limitation of liability.** To the maximum extent permitted by law, the authors are not liable for any direct, indirect, incidental, special or consequential damage arising from the use of or inability to use the software, including damage to the microscope, objectives, stage, camera, spectrometer or other equipment, damage to or loss of samples, loss of data or measurement results, lost time or profit, and injury.

**Use at your own risk.** The software moves a motorised stage and can switch on a laser. The user alone is responsible for:
- safe operation of the instrument, including the objective working distance, stage limits and collision risk;
- laser safety: protective eyewear, safety rules and local regulations for the laser class of the instrument;
- checking the settings and the results before relying on them, and keeping copies of important data;
- compliance with the instrument manufacturer's licence, warranty and service terms — using third-party control software may affect the warranty;
- compliance with the laws and regulations that apply to the user.

The safety checks built into MicroControl reduce risk but do not replace the user's attention and judgement; they have not been certified by anyone. The software is not intended for medical, diagnostic or safety-critical use.

## Contact

Questions and bug reports: andrey.quant@yandex.ru
