# MicroControl

**Python control of the Optosky ATR8300 Raman microscope: scan a substrate, stitch one picture of it, put the lithography grid on it, choose the squares you need and export them for AutoCAD.**

**Authors:** A. Lyapeikov, S. Domenko · **Version:** 0.43.16 · Русский: [docs/README_ru.md](docs/README_ru.md) · 中文: [docs/README_zh.md](docs/README_zh.md) · PDF: [EN](docs/MicroControl_en.pdf), [RU](docs/MicroControl_ru.pdf), [中文](docs/MicroControl_zh.pdf) · Full technical reference: [EN](docs/reference_en.md), [RU](docs/reference_ru.md), [中文](docs/reference_zh.md)

![Overview: the 10x scan with the grid of the sample, numbered squares, a draft, 50x and 100x inside](docs/img/view.jpg)

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

The microscope window closes with **Shift+Esc** or its ×; the view — with **Esc** (when no mode, help or list is open) or its ×. **H** shows all keys. **Ctrl+P** shows / hides the Settings panel on the left.

---

## The five steps

### 1. Connect

`m = mc.Microscope()` finds the controller port and the camera by itself, switches the light on and checks that the stage answers. All coordinates are in micrometres, $+Z$ is the stage going **up**, towards the objective. Every move is checked against limits (one step, the whole travel, a ceiling for $Z$), so a typo cannot drive the sample into the objective. `mc.Microscope('sim')` is a simulator: the same commands without the instrument.

### 2. Green area — what to scan

![Manual mode](docs/img/manual.jpg)

`m.manual()` opens the live camera. Move with the arrows (one press = one frame), focus with `E` / `Q` (or `F` — autofocus). Letters do things: `L` light, `I` laser, `W` white balance, `R` ruler, `S` queue the scan (it is quick stitched in the background; the queue is at the bottom), `V` open the result in the viewer.

- **One window.** The bar on top has the main buttons: **New area**, **Scan area** (scan + stitch), **Export** (drawing for AutoCAD: DWG + DXF of the chosen area), **Settings** (steps, exposure, focus points).
- **Map** (top right; `Tab` — big map centred on the camera, the camera goes to the corner): everything already shot, at its real place (10x under 50x under 100x), the outlines of the areas and a yellow frame where the camera is now. Where the camera has only been while you drove around, its live frames stay on the map, a little darker, under the scans ("fog of war" lifted; kept with the sample in `<sample>/fog/`); a click on the map clears the fog along the way too. **Layout** (menu): one view (camera big, the map in the corner, `Tab` swaps) or two side by side — camera | map or map | camera (`Tab` swaps them); the selections list on the right or on the left (in the view too). Kept for next time. The map and the view show the sample turned the way the camera shows it (after `T` or a mirrored camera too), so what is right of the camera frame on the map is to the right on the camera; `\` turns the map 90° more. **Click the map — the stage drives there**; **drag on the map — a new area** on the objective in place; **`Ctrl`+drag — a free box** exactly as drawn (around a crystal), not snapped to the squares; wheel — zoom; right drag — move the map; double-click a shot area — open it in the viewer.
- **Selection by keys** still works: go to one corner, press `1`, to the opposite one, press `2`.
- **Focus points:** the height $Z$ at each corner is remembered; `K` adds more points. During the scan every frame gets its own $Z$ from these points, so a tilted sample stays sharp. With the checkbox "Z follows the focus points" the arrows also follow them.
- **Several selections, layers:** set them one after another (`1`, `2`, then `1`, `2` again, or drag on the map). All selections are on one level and may overlap, any objectives; the order of the list on the right is the order of the layers (top on top), drag a card to change it. Click — choose, **Go** — drive to it, **...** — all its scans (the star is the main one), **x** — delete.
- **Ruler:** `R`, then click two points — the distance in µm (the stage may move between the clicks).
- **Undo:** `Ctrl+Z` undoes any action in either window (selections, objective, ticks, order, focus points, light, steps, transparency, align, crosses and cells, a move — the stage drives back), up to 1000 steps; `Ctrl+Y` redoes it.

`m.scan()` shoots every area frame by frame with 30 % overlap (35 % at 50x, 100x); every scan of a selection is a new folder `<sample>/scans/<date>_<time>_S03_50x/`. Selections keep their numbers and colours when one is deleted. Only the selections of the objective in place are shot. It prints the expected time; this estimate learns from the previous scans of your microscope. In the windows the scan goes into the queue and is **quick-stitched while it is being shot**: every new frame is stitched at once to its neighbours already shot (in a separate process, on the other cores), the viewer shows the frames as they come and you can mark the crosses meanwhile; the quick stitch after the scan takes these seams ready-made (a stopped scan is quick-stitched as far as it got); `mc.stitch(folder)` does it from Python (`mode='quality'` — the exact stitch).

### 3. Red grid — find the squares

![Grid, numbers and chosen squares](docs/img/view_zoom.jpg)

`mc.view(folder)` opens the stitched picture (drag — move, wheel — zoom).

- Press `C` and click **two neighbouring crosses in one row**. From them the program knows the tilt and the size of the squares, turns the picture level and draws the red grid over everything. The mark lands exactly where you click; `Shift`+click snaps it to the cross centre, a mark can be dragged. The grid is fitted to all crosses at once; `Shift+C` then puts every frame onto the crosses (see *Accuracy for lithography*).
- **Numbers:** press `N`, click a square and type the number written next to it (`12 7`). Do this for two squares in different rows and columns — then every square has its real number.
- **One grid for the whole sample.** It is made from the crosses marked by hand on one selection (the one chosen in the list) and does not change while they do not. Every other scan takes it as it is; if its picture is off, `A` (align) moves the picture under the grid. To make the grid on another selection, first remove the old crosses (open that selection, `C`, `Del`) — `C` elsewhere says so. Square numbers appear after you number two squares (`N`).

### 4. Choose squares — the mask

- `M` — select: click a square (again — off) or drag a box, every square it touches is chosen (`Shift` — unselect). Right click — remove one square, `Shift` + right click — the whole green area. Chosen squares are green; **Enter** makes them a selection.

### 5. Export

`X` (in either window) or the button **Export** writes everything ticked on the cards (the boxes are off by default) into `<sample>/export/` (replaced each time). **Every ticked selection must be quality-stitched**, otherwise the export is refused and the message names the others.

The result is one drawing `export/drawing.dwg` (origin at cross 1,1, square to the lattice): the pictures of the selections at their real places in micrometres, every cell as a closed square, the crosses with their numbers, each selection on its own layer, and an empty `BONDS` layer for drawing bonds. **The drawing is made only when the crosses are marked** (the crosses are the beacons that put everything together for the lithography); without them there are the pictures and `mixed.png` only. Next to it: the pictures (`S03_50x.png` …) and `mixed.png` — all ticked scans laid over each other in the list order, at the pixel of the finest one. The DWG is written by the free ODA File Converter if it is installed; otherwise there is `drawing.dxf` (AutoCAD: Save As → DWG). Keep the `export/` folder together: pictures in DWG are always separate files. For one scan, `mc.export(folder)` writes its chosen squares straightened (PNG, DXF, list of numbers) into `export/` of that scan.

`mc.projects()` and `mc.all_samples()` print a table: id, name, samples, selections, scans, created, updated, description (the Open dialogs show the same). In the viewer, Open → **New window** shows another project in its own window (`mc.view_window(folder)`), the current one stays open.

Squares selected in the viewer (`M`) become a selection on **Enter** — a **draft** (no objective, not scanned) or with the objective chosen for new ones (`Shift+G`); it appears in the list at once and is chosen. All selections are on one level: any objectives may overlap, nothing is cut; the order of the list is the order of the layers (top on top), drag cards to change it; the box on a card puts it into the export. `U` in the viewer draws a selection by its corners: every side goes only across or up / down and snaps to the crosses and to the corners already put (grey guide lines), so the outline is always right-angled; a click on the first corner or Enter closes it (the last corner is added by itself). **Export** writes everything ticked into one folder `<sample>/export/`: one AutoCAD drawing (origin at cross 1,1, pictures in place, every cell as a closed square, crosses with their numbers), the pictures, and `mixed.png` — all ticked scans laid over each other in the list order, at the pixel you choose.

| | |
|---|---|
| ![Big map (Tab): the 10x overview with its grid, a draft, 50x and 100x inside, the camera in the corner](docs/img/map_zoom.jpg) | ![The 50x scan: the grid of the sample is there as it is, no crosses marked on it](docs/img/view_50x.jpg) |
| *Layers on the map: 10x → draft → 50x → 100x, the camera in the corner* | *50x scan: one grid for the sample — from the crosses marked on the 10x* |
| ![Export of the sample: the 10° tilted scan comes out square to the lattice, crosses with numbers](docs/img/export.jpg) | ![Export, close-up: the 50x and 100x pictures over the 10x one, the fine crosses](docs/img/export_zoom.jpg) |
| *Drawing for AutoCAD: the tilted chip is square to the lattice, its crosses on the drawn ones* | *Close-up: 50x and 100x quality pictures, each on its own layer* |

## Accuracy for lithography

![Profiles (P): two lines across the arms of a cross, R, G, B along them, the edges in yellow on the map, in the preview and on the plots](docs/img/profiles.jpg)

The drawing is used to write a circuit over the flakes, so every picture in it is placed **by the crosses around it**, not by the stage. What matters, and what the program does about it:

| What | How |
|---|---|
| Stitching in two steps | Quick — by itself after the scan (frames by their overlaps, an overview). **Quality** — Stitch / `Shift+J` (into the queue; blurry frames are shot again; for a weak seam the stage goes by itself and shoots a frame in the middle of the pair — a shifted bridge tying both). Then `C` (2–3 crosses) and `Shift+C` — every frame is put onto the crosses: the frames, the square lattice and the lens distortion are fitted together, so seam errors do not add up. Python: `mc.stitch(folder, mode='quality')`, then `mc.fit_crosses(folder)` |
| Picture in the drawing | every pixel is taken through the **4 crosses of its own square** (bilinear), crosses measured on that very scan; what is left of stitching, distortion and tilt does not reach the drawing |
| Cross centre | template + **edges of the arms**: two straight lines through the middles of the arms, their crossing is the centre (≈0.1 px, also when the chip is turned; digits next to the cross are ignored) |
| Lens distortion | `D` in the microscope window (`m.calibrate_distortion()`): 3×3 pictures with 50 % overlap around any details; matched pieces of the overlaps give $k_1, k_2, p_1, p_2$ without crosses and without trusting the stage. Remembered for the objective, used for every scan |
| Frames | full camera resolution, JPG 95 by default; `m.frame_format = 'png'` — lossless. Crosses and seams are measured in the green channel (sharpest, no colour fringes) |
| Before a scan | button **Prepare**: a guide over the live picture in 3 steps — each says where to drive, then **Done** / Skip / Cancel: a clean empty place: exposure (brightest ≈80 %, no overexposure), white balance, **flat field** (5 places, median: vignetting and dust on the optics) → a detail: autofocus, settling pause → lens distortion (once per objective). Saved for the microscope and objective; next time it asks: **Overwrite** or **Keep** (with the date). The stitch uses this flat field (and a dark frame, if taken with `m.dark_frame()`). Python: `m.prepare()`, `m.flat_field()` |
| Scale | `m.mask_pitch = 100` — the design pitch of the crosses on the mask: the drawing takes every size from it. `Shift+C` also measures the real pixel size by the crosses (and X/Y stretch, skew) and saves it for the objective: the ruler and the next scans use it |
| Autofocus | contrast (normalised variance) to find the peak, Vollath F4 for the exact pass, a Gaussian fitted to the sharpness curve; step and window are set for each objective in Settings (`AF step`, `AF window`, 0 = from the depth of field). Note: one Z motor step is 5.56 µm — more than the depth of field at 50x / 100x |
| Backlash | `B` — one-side approach (remembered). The stitch does not trust the stage anyway; this keeps the frames where they are expected and the camera overlay exact |

For the circuit itself, outline the flakes at 50x / 100x: at 10x the optics blur edges by about 1 µm.

---

## Projects and samples

Work is grouped like this: **Project** (a piece of work, with a description) → **Sample** (one substrate on the stage: its own selections and coordinates, a description) → **Selection** (an area on one objective; layers: map 10x > details 50x > 100x) → **Scan** (every scan of a selection is a new version) → grid, chosen squares, drawing. Selections are saved in the sample and come back when it is opened again; the chosen squares on the maps, `m.list()` and the viewer show only the open sample. Selections are layers: an outline made of rectangles (no inner squares) and an objective, all on one level, the list order is the layer order; Ctrl+click cards, then Merge / Minus / Overlap. Squares chosen with the brush in the viewer become a selection on Enter (one per figure, a draft or the objective chosen for new ones); a selection is scanned only by the frames that touch its outline; the eye on a card hides it on the maps.

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

Once the crosses are marked, the map in manual mode shows the lithography grid as thin semi-transparent red lines — new areas are made of exactly these squares; it is one grid for the whole sample.

1. Mark a large selection with a small objective (10x) and scan it: this is the map.
2. Look at the map (`mc.view`) and find the interesting places.
3. Put a larger objective in place (`O` in manual mode, or `m.objective = 50`), drag small selections on the map and scan again — only the 50x selections are shot. The same again with 100x on the 50x ones.

All selections are on one level (no nesting): the objective (`10x`, `50x`, `100x`) is next to the name, the frames and the time are counted for that objective, and the list order is the order of the layers. Then quality-stitch what is needed, align the scans (`A` in the view), mark the crosses (`C`) and export (`X`).

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
| **Instrument on another computer** | the old PC next to the microscope serves it, Jupyter runs on a fast PC | `python -m microcontrol serve` there · `mc.Microscope('net://KEY@192.168.50.2', objective=10)` here |
| **AI agent** | a local language model works through tools, the operator allows each action | `mc.agent.chat(m, model='qwen3:8b')`, MCP server |

Settings are ordinary properties with checked ranges: `m.settings` lists all of them, `m.photo_pause = 0.5` changes one, a value out of range is refused with the allowed range. `mc.help()` prints a short guide.

**Emergency stop:** `python -m microcontrol stop` in any console (or the file `~/.microcontrol/STOP`): a running program sends "light 0" and "laser 0", closes the port and stops. If the program crashes or is killed without closing the instrument, a separate guard process switches the laser, motors, light and cooling off by itself.

**Instrument on another computer.** On the PC the microscope and camera are plugged into: `python -m microcontrol serve` (it finds the controller, opens the camera and prints the address with the key). On the fast PC: `m = mc.Microscope('net://KEY@192.168.50.2', objective=10)` — the key is remembered, next time `net://192.168.50.2` is enough; everything else works as with the instrument next to you, the computing is done here. A direct Ethernet cable with fixed addresses or one router; Wi-Fi works too (frames then go as JPEG 95). Safety stays at the instrument: without a word from the program for 10 s, or when the connection breaks without `m.close()`, the server switches the laser, motors, light and cooling off; `python -m microcontrol stop net://192.168.50.2` stops it from anywhere. Over the internet — only through a VPN (Tailscale, WireGuard).

**Local network and Wi-Fi, AI agent on another PC.** The chain: microscope → old PC (`python -m microcontrol serve`: a thin server) → new PC on the same network (cable or Wi-Fi) → an AI agent there through MCP.
- The address need not be known: `net://KEY@auto` finds the server in the network by itself (a UDP broadcast on port 5150); `python -m microcontrol find` lists the servers (name, controller, camera, free / busy).
- On Windows the server opens port 5150 (TCP and UDP) in the firewall for private networks by itself; without administrator rights it prints the two `netsh` commands to run once. Make the network "Private" in Windows on both PCs.
- Wi-Fi: frames go as JPEG 95 (chosen by a measurement); a silent drop is noticed within ~15–30 s (TCP keepalive, timeouts), then the program reconnects; meanwhile the server has switched the instrument off after 10 s without word (`--dead`).
- AI agent: `python -m microcontrol mcp net://KEY@auto --objective 10` is an MCP server on stdio; put it into the MCP client of the harness (DeepSeek or any model that the client drives):
```json
{"mcpServers": {"microcontrol": {"command": "python", "args": ["-m", "microcontrol", "mcp", "net://KEY@auto", "--objective", "10"]}}}
```
  Moves, scans and deletions ask the operator on the new PC (Allow / Deny); the laser is never available to the agent.

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
| **Export** | Everything ticked into one folder: one AutoCAD drawing square to the lattice (when the crosses are marked), the pictures and `mixed.png`; only quality-stitched selections. |
| **Selections and layers** | Each selection gets its own colour, objective, focus points and scans; all are on one level, may overlap, and the list order is the order of the layers. |
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

**One letter — one meaning in both windows.** Keys of one window only say so in the other one (and do nothing).

| Key | Meaning | Microscope window (`m.manual()`) | View (`mc.view(folder)`) |
|---|---|---|---|
| arrows, wheel, `+` / `-`, `0` | move, zoom, whole | stage by one frame (Step X, Y); the map | the picture |
| `,` `.` | previous / next selection | ✓ | ✓ |
| Ctrl+click | mark several | cards (Merge / Minus / Overlap above the list) | cards and the picture |
| `Enter` | confirm | drive to the chosen selection | chosen cells -> a selection (chosen here and there); save the alignment |
| `Del` | delete | the chosen selection / the new one (Ctrl+Z brings it back) | the marked / chosen ones (not scanned); in the crosses mode — crosses again |
| `S` | scan: into the queue | the chosen / marked selections (Scan area) | the chosen / marked ones (the microscope window's queue) |
| `J` | stitch | check the live picture against the mosaic; `Shift+J` — exact (quality) stitch of the chosen (Stitch) | edit the seams by hand (a quick one: its quality stitch first); `Shift+J` — quality stitch into the microscope window's queue |
| `X` | export | everything ticked -> `<sample>/export/` (Export) | the same |
| `O` | objective | the one in place: next (put it in place) | of the chosen / marked selections; `Shift+O` — for new ones |
| `[` `]` | transparency | the picture of the chosen selection on the map | the picture of the chosen selection / of the marked ones |
| `R` | ruler | two clicks | point-point -> line-line -> point-line -> off |
| `T` | turn the picture | the camera by 180° (remembered) | level by the lattice / as shot |
| `M` · `C` · `N` | cells · crosses · numbers | — | select cells (`Ctrl+A` — all / none) · mark crosses (`Shift+C` — frames onto the crosses) · number a square |
| `P` · `G` · `A` · `Y` | profiles · straight edges · align · the others | — | profile lines (4 clicks) · crop the steps · align this scan with the others (Space — by itself) · other selections' pictures on / off |
| `U` | outline | — | a selection by its corners: sides only across / up-down, snapping to the crosses and the corners; Enter or the first corner closes it |
| `E` / `Q` (`PgUp` / `PgDn`) · `F` · `K` · `1` / `2` | stage up / down · autofocus · focus point · corners | ✓ (`Shift` — ×5) | — |
| `L` · `I` · `W` · `Space` · `V` | light · laser · white balance · picture · open the view (its own window: this one keeps working) | ✓ | — |
| `Tab` or click a field | steps, exposure, objective | ✓ | — |
| `Ctrl+Z` · `Ctrl+Y` | undo · redo | ✓ (moves too) | ✓ |
| `H` · `Esc` | all keys · leave the mode / unmark | ✓ | ✓; `Esc` with nothing open — **close the view** |
| `` ` `` (or `F12`) | timing: ms per frame for each part of the window (also in the log every 5 s) — what is slow on this PC | ✓ | ✓ |
| `Shift+Esc` | exit | the microscope window (only so) | ✓ |
| `Ctrl+P` | Settings panel (left) | show / hide | — |

**Settings panel** of the microscope window (button Settings, the arrow on the left or `Ctrl+P`): Z follows the focus points, One-side approach (backlash; on by default), **Prepare**, **Distortion**, **Zero here**, **Steps = frame**, **Check directions**, **Arrows X / Y** the other way, **Bad frame below sharpness** (a limit with -, +, 70% of now, Off: a less sharp frame is re-taken right there, a little further along the path, then marked bad), **Map 90 deg**, **Clear fog** (the fog is the live frames laid simply by the stage coordinates, without any matching; forget the live frames of the fog of war on the map; the setting `m.fog_refresh` (off by default: standing still, the live frame overwrites the fog anyway) makes drives over already seen places shoot again, so the fog is overwritten with new frames). On top: Prepare, New area, Scan area, Stitch, Export, Settings; the menu: Open, New, Info, Scans (history of the chosen selection; the same as `...` on its card), Queue, View, Help. Ctrl+drag on the map — a box on the objective in place (a crystal); `Esc` during a job — stop it.

---

## Where things are saved

```
projects/<project>/project.json                       name, description
projects/<project>/samples/<sample>/sample.json       its selections (the list order = layers), description
    scans/<date>_<time>_S03_50x/                      all scans, one level; every scan is a new version:
        frames/  join/  grid.json                     frames, stitched picture, crosses and squares
    export/                                           the export (replaced each time): drawing, pictures, mixed.png
    thumbs/<uid>/                                     corner pictures of the selections
    photos/                                           single pictures (Space)
logs/                                                 log of every session
```

---

## AI agents

A local model (for example through Ollama) can drive the microscope: look through the camera and at mosaics, move, scan, stitch, mark crosses, number and choose squares, export.

```python
mc.agent.chat(m, model='qwen3:8b')                 # chat with a local model
python -m microcontrol mcp COM3                    # or a server for any MCP client
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

**The camera needs the MindVision camera SDK (Windows x64): download it from the camera maker, mindvision.com.cn (Downloads → SDK), run the installer as administrator and restart the PC.** It is needed only on the PC the camera is plugged into (a PC that works through `net://` does not need it). Without it the program stops with `E502`. Close the original Optosky program before connecting: only one program can use the port and the camera. Examples: `example.ipynb` and the `demo/` folder.

**First start:** check the link with `python -m microcontrol probe`, then in `m.manual()` check the directions — `Q` must move the stage **down**, the arrow → must move the picture to the left. If not, see the [reference, section 20](docs/reference_en.md#20-first-start-on-the-instrument).

**Self-test on the simulator** (no instrument needed): `python test_sim.py --quick` — about 10 s; `python test_sim.py` — everything, the parts run in parallel (about 15 min on 2 cores).

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
