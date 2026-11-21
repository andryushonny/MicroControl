# MicroControl

**Python control of the Optosky ATR8300 Raman microscope: scan a substrate, stitch one picture of it, put the lithography grid on it, choose the squares you need and export them for AutoCAD.**

**Authors:** A. Lyapeikov, S. Domenko · **Version:** 0.20.0 · Русский: [docs/README_ru.md](docs/README_ru.md) · 中文: [docs/README_zh.md](docs/README_zh.md) · PDF: [EN](docs/MicroControl_en.pdf), [RU](docs/MicroControl_ru.pdf), [中文](docs/MicroControl_zh.pdf) · Full technical reference: [EN](docs/reference_en.md), [RU](docs/reference_ru.md), [中文](docs/reference_zh.md)

![The scan viewer: red grid of squares, chosen squares (green), hints (yellow), samples on the right](docs/img/view.jpg)

> Independent project, not affiliated with Optosky. Provided as is, without warranty — see the [Disclaimer](#disclaimer) and the [MIT licence](LICENSE).

---

## Quick start

Five steps, from a sample on the stage to a drawing for AutoCAD:

```python
import microcontrol as mc

m = mc.Microscope()          # 1. connect to the microscope   (mc.Microscope('sim') — simulator)

m.manual()                   # 2. green area: keys 1 and 2 at two opposite corners, focus at each
folder = m.scan()            #    shoot the area frame by frame
mc.stitch(folder)            #    join the frames into one picture (mosaic)

mc.view(folder)              # 3. key G: click two neighbouring crosses -> the red grid
                             # 4. keys S / B: choose squares (green); Y shows hints, Shift+Y takes them
mc.export(folder)            # 5. the chosen squares: straight PNG + DXF in micrometres

m.close()
```

Both windows close with **Shift+Esc** or the window's ×. A single `Esc` only cancels. **H** shows all keys.

---

## The five steps

### 1. Connect

`m = mc.Microscope()` finds the controller port and the camera by itself, switches the light on and checks that the stage answers. All coordinates are in micrometres, $+Z$ is the stage going **up**, towards the objective. Every move is checked against limits (one step, the whole travel, a ceiling for $Z$), so a typo cannot drive the sample into the objective. `mc.Microscope('sim')` is a simulator: the same commands without the instrument.

### 2. Green area — what to scan

![Manual mode](docs/img/manual.jpg)

`m.manual()` opens the live camera. Move with the arrows (one press = one frame), focus with `E` / `Q` (or `Shift+F` — autofocus).

- **Mini map** (top right): one cell = one camera frame; the yellow cell is where you are. It moves with the arrows: arrow up — the cell goes up.
- **Selection:** go to one corner of the sample, focus, press `1`; go to the opposite corner, focus, press `2`. The selection appears in its colour; corners snap to whole frames. Each selection remembers the objective it was marked with.
- **Focus points:** the height $Z$ at each corner is remembered; `F` adds more points. During the scan every frame gets its own $Z$ from these points, so a tilted sample stays sharp. With the checkbox "Z follows the focus points" the arrows also follow them.
- **Several selections, layers:** set them one after another (`1`, `2`, then `1`, `2` again). Selections of one objective do not overlap, but a selection of a higher objective may lie inside one of a lower objective (see *Layers* below). The list on the right shows them nested: click — choose, **Go** — drive to it, **x** — delete.
- **Ruler:** `L`, then click two points — the distance in µm (the stage may move between the clicks).
- **Undo:** `Ctrl+Z` undoes any step (a selection, a focus point, a move — the stage drives back), `Ctrl+Y` redoes it.

`m.scan()` shoots every area frame by frame with 30 % overlap (35 % at 50x, 100x) and saves the frames into a new folder (one sub-folder per selection); `m.scan(name='test')` again writes into the same folder. Selections keep their numbers and colours when one is deleted. Only the selections of the objective in place are shot. It prints the expected time; this estimate learns from the previous scans of your microscope. `mc.stitch(folder)` measures from the overlaps where each frame really is and joins them into one seamless picture.

### 3. Red grid — find the squares

![Grid, numbers, chosen squares and hints](docs/img/view_zoom.jpg)

`mc.view(folder)` opens the stitched picture (drag — move, wheel — zoom).

- Press `G` and click **two neighbouring crosses in one row**. From them the program knows the tilt and the size of the squares, turns the picture level and draws the red grid over everything. The mark lands exactly where you click; `Shift`+click snaps it to the cross centre, a mark can be dragged. `F` refines the grid using all crosses.
- **Numbers:** press `N`, click a square and type the number written next to it (`12 7`). Do this for two squares in different rows and columns — then every square has its real number.

### 4. Choose squares — the mask

- `S` — click squares or drag a box; `Shift` — unselect. `B` — brush: drag rectangles, every square in a rectangle is chosen (`Shift` — unchosen). Chosen squares are green.
- **Hints:** `Y` finds and shows yellow squares where something is (flakes, crystals); they are not computed by themselves — it takes a few seconds. `Shift+Y` takes all of them and switches to the brush to correct the choice; `<` / `>` change the sensitivity.

### 5. Export

`E` in the viewer or `mc.export(folder)` writes `export/` inside the scan folder: a straight PNG of the chosen squares (transparent background), a DXF for AutoCAD in micrometres and a list of the square numbers.

---

## Layers: a map and the details

1. Mark a large selection with a small objective (10x) and scan it: this is the map.
2. Look at the map (`mc.view`) and find the interesting places.
3. Put a larger objective in place (`M` in manual mode, or `m.objective = 50`), mark small selections inside the map and scan again — only the 50x selections are shot. The same again with 100x inside the 50x ones.

The selections nest like Russian dolls: in the list on the right of manual mode every selection lies inside the frame of the one it was marked in, with its objective (`10x`, `50x`, `100x`) next to the name; the frames and the time are counted for that objective.

```python
m.list()          # all scanned selections as layers, numbered
m.open(3)         # look at number 3 again (mc.view)
m.remove(3)       # delete number 3 completely (asks first)
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

**Emergency stop:** `python -m microcontrol stop` in any console (or the file `~/.microcontrol/STOP`): a running program sends "light 0" and "laser 0", closes the port and stops.

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
| **Stitching** | Neighbouring frames overlap; comparing the overlaps gives their exact shifts. All shifts are solved together, so stage errors do not add up. The uneven light of the lens (darker, differently tinted edges) is measured from the frames themselves for each colour and divided out; the brightness of every frame is then matched to its neighbours over the overlaps, so the seams do not show. |
| **Grid** | Two crosses give the direction and the length of one square side; the grid is square, so that is all it needs. Fitting to all crosses makes it exact. Two labelled squares give the numbering of the whole grid. |
| **Hints** | Unsupervised anomaly detection: the program learns what the empty substrate looks like and marks the squares with something unlike it (see [Machine learning inside](#machine-learning-inside)). |
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

### Hints: finding flakes and crystals on the substrate

The yellow hint is **unsupervised anomaly detection** (the idea of PaDiM): nobody has to label flakes; the program learns what *normal* substrate is and marks what is not.

1. **Patches.** The whole stitched picture is reduced (≤ 6 Mpx) and cut into small 4 × 4 px patches.
2. **Features** of every patch (8 numbers): how its colour differs from the local background in the Lab colour space (3; uneven illumination is removed), how saturated that difference is (1), and texture — edges, gradient, local contrast, difference of Gaussians (4).
3. **Model of normal substrate — Gaussian mixture.** The features of all patches are fitted with a mixture of $K = 3$ Gaussians by the EM algorithm (the substrate has a few kinds of normal patches: clean, with a marking dot, with a cross edge). Fitting is robust: the farthest 12 % of the patches — where flakes are — are dropped and the mixture is fitted again.
4. **Anomaly score.** For every patch — the Mahalanobis distance to the nearest Gaussian, in units of the spread of normal patches (median and MAD):

```math
z = \frac{\min_k \sqrt{(\mathbf f-\boldsymbol\mu_k)^{\mathsf T}\Sigma_k^{-1}(\mathbf f-\boldsymbol\mu_k)} - \mathrm{med}}{1.48\,\mathrm{MAD}},
```

   a patch with $z > 8$ is "something".
5. **Cleaning.** Patches near the grid nodes (crosses and their numbers) and at the edges are ignored; connected spots smaller than 2 patches and small dark colourless spots (dust) are dropped.
6. **Squares.** A square is yellow if "something" covers more than a share $5\,\% \cdot 10^{-2S/100}$ of it, $S$ — the sensitivity 0…100 (keys `<` `>`): changing it is instant.

Options: train the normal model once on a clean substrate — `mc.train_hints(folder, default=True)`; or use neural-network features instead of the hand-made ones (a ResNet18 file in ONNX format, run by OpenCV — the original PaDiM), see the [reference](docs/reference_en.md). The whole search takes a few seconds for thousands of squares and needs no GPU.

---

## Keys

**Manual mode** (`m.manual()`)

| Key | Action |
|---|---|
| Arrows / `W A S D` | move by one frame (Step X, Y) |
| `E` / `Q` | stage up / down by Step Z (`Shift` — ×5) |
| `Shift+F` | autofocus |
| `1`, `2` | corners of a selection |
| `F` | add a focus point here |
| `Y` | "Z follows the focus points" on / off |
| `,` `.` · `Enter` | previous / next area · drive to it |
| `C` | cancel the new area / delete the chosen one (press twice) |
| `Space` · `Shift+L` | take a picture · light on / off |
| `L` | ruler: two clicks — the distance |
| `Ctrl+Z` · `Ctrl+Y` | undo · redo (moves too: the stage drives back) |
| `Tab` or click a field | change steps, exposure, objective |
| `P` | hide / show the left panel |
| `T` · `O` · `[` `]` | turn the camera picture 180° · check directions with the camera · arrows (and the mini map with them) the other way (all remembered) |
| `H` · `Shift+Esc` | all keys · exit |

**Viewer** (`mc.view(folder)`)

| Key | Action |
|---|---|
| drag · wheel · `0` | move · zoom · whole picture |
| `G` · `F` | mark crosses · fit the grid to all crosses |
| `N` | number a square |
| `S` · `B` | select (click, box) · brush (rectangles); `Shift` — unselect |
| `Y` · `Shift+Y` · `<` `>` | hints · take the hints · sensitivity |
| `R` | picture level / as shot |
| `E` | export |
| `,` `.` | previous / next selection |
| `L` | ruler: two clicks — the distance |
| `Ctrl+Z` · `Ctrl+Y` | undo · redo any change |
| `Esc` · `H` · `Shift+Esc` | leave the mode · all keys · exit |

---

## Where things are saved

```
scans/<date>_<objective>_<name>/      one scan (several selections: selection_01/, selection_02/ inside)
    frames/        the frames
    join/          the stitched picture
    grid.json      crosses, grid, numbers, chosen squares, hints
    export/        PNG + DXF of the chosen squares
photos/            single pictures (Space)
logs/              log of every session
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
