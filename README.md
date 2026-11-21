# MicroControl

**Software control of the Optosky ATR8300 Raman microscope (MP / Mapping version): stage, illumination, laser, camera, autofocus, area scanning, stitching and preparation of images for training datasets.**

**Authors:** A. Lyapeikov, S. Domenko. **Version:** 0.18.0. Русская версия: [README_ru.md](README_ru.md).

---

## Contents

1. Purpose and components
2. Notation and units
3. Hardware and its parameters
4. Coordinate systems
5. Controller protocol
6. Motion model and timing
7. Safety: limits and their reasoning
8. Camera and image formation
9. Autofocus
10. Area scanning
11. Data storage
12. Stitching and marking: overlap stitching, crosses, square selection, export, clean-up
13. Preparing images for a training dataset
14. G-code programs
15. Manual mode and viewer
16. Microscope home and position memory
17. Emergency stop and exclusive access
18. Simulator
19. Installation
20. First start on the instrument
21. Troubleshooting
22. Reference: parameters, commands, errors, log
23. Demo programs
24. What is not verified on the instrument

---

## 1. Purpose and components

MicroControl is a Python library for controlling the microscope from a Jupyter notebook on the computer attached to the instrument (Windows 10). It replaces the stock Optosky program for optical microscopy: moving the stage in three axes, illumination, laser, imaging, autofocus and automatic scanning of a sample area followed by stitching of the frames.

The library reproduces the exchange of the stock program with the controller: the same commands, the same order, the same port settings (section 5). The protocol was recovered from the executable code of the stock program and from recordings of the exchange with the instrument.

Modules:

| Module | Purpose |
|---|---|
| `core` | class `Microscope`: connection, motion, parameters, imaging |
| `protocol` | frame format, list of commands |
| `focus` | sharpness measures and the autofocus algorithm |
| `scan`, `gcode` | scan area, G-code scan program and its execution |
| `storage` | frame storage of a scan (HDF5 or JPG) |
| `stitch`, `optics` | overlap stitching (like ASHLAR), flat field, lens distortion, multiband blending |
| `crosses`, `mask` | cross lattice, square selection, numbering, hints, export |
| `join` | simple mosaic of the frames |
| `manual`, `viewer` | manual control window, scan viewer |
| `device` | instrument memory: home and last position |
| `control` | exclusive access, emergency stop |
| `sim` | instrument simulator |
| `probe` | controller link check |

---

## 2. Notation and units

Lengths are in micrometres (µm), time in seconds (s), laser power in milliwatts (mW).

| Symbol | Meaning | Default |
|---|---|---|
| $\delta$ | stage travel per motor step | $5.556$ µm |
| $N$ | number of motor steps (integer) | — |
| $M$ | objective magnification | 4, 10, 20, 50, 100 |
| $a$ | camera sensor pixel size | $2.2$ µm |
| $p = a/M$ | camera pixel on the sample | $0.22$ µm at $M = 10$ |
| $N_x \times N_y$ | frame size in pixels | $2592 \times 1944$ |
| $W = N_x p,\ H = N_y p$ | field of view (frame size on the sample) | $570 \times 428$ µm at $M = 10$ |
| $v$ | stage speed | $3000$ µm/s (assumed value) |
| $\tau$ | pause after a move (`move_pause`) | $0$ s |
| $\mathrm{NA}$ | objective numerical aperture | 0.1 / 0.25 / 0.4 / 0.75 / 0.9 |
| $\lambda$ | wavelength | $0.532$ µm (laser) |

---

## 3. Hardware and its parameters

| Unit | Parameter | Value | Source |
|---|---|---|---|
| Stage X, Y | travel | $5 \times 5$ cm | ATR8300MP datasheet |
| Stage Z | travel | 20 mm | datasheet |
| Stage | scan speed | 20 mm/s | datasheet; matches the command rate of the stock program (2 mm in 94 ms) |
| Motors | travel per step $\delta$ | $1/0.18 = 5.556$ µm on all axes | exchange recordings: 200, 300, 500, 2000 µm in the Optosky UI → 36, 54, 90, 360 steps |
| Motors | limit switches | on all three axes | `usercfg.ini`: `DeviceMotorType=XYZLimit` |
| Controller | interface | USB, STM32 virtual COM port, VID 0483, PID 5740 | — |
| Camera | model, sensor | MindVision MV-SUA502C, $2592 \times 1944$, pixel $2.2$ µm | — |
| Illumination | levels | integers 0…5; the stock program uses 5 | — |
| Laser | wavelength, control | 532 nm, power in whole mW | — |
| Spectrometer | CCD cooling | Peltier element with a fan; set point −5 °C in operation, 20 °C at shutdown | `usercfg.ini`: `CoolTemperature=-5` |
| — | sample–objective gap before autofocus | ≈ 1 mm | label on the stage |

The fan noise heard after the stock program starts is the cooling of the spectrometer CCD (command `12`, section 5), not a stage-ready signal.

---

## 4. Coordinate systems

The controller does not report the stage position: it only accepts relative moves in steps. The position is computed by the program as the sum of the executed moves.

There are three coordinate systems.

1. **Motor steps** $(N_x, N_y, N_z)$ — the internal position, integers.
2. **Real coordinates** $(X, Y, Z)$, µm, `m.real_position`:

```math
X = \delta \cdot N_x,\quad Y = \delta \cdot N_y,\quad Z = \delta \cdot N_z,
```

counted from the microscope **home** (section 16); if no home is set — from the connection point.
3. **Working coordinates** $(x, y, z)$, µm, `m.position` — real coordinates minus the zero offset $(x_0, y_0, z_0)$ set by `m.offset()` (G-code `G92`):

```math
x = X - x_0,\quad y = Y - y_0,\quad z = Z - z_0.
```

The zero offset is for convenience; travel limits and home do not depend on it.

**Directions.** $+Z$ — stage up, towards the objective. In the stock program code the direction is a byte: X `00` — "forward", `FF` — "backward"; Y `00` — "left", `FF` — "right"; Z `00` — down, `FF` — up. The correspondence of X, Y to the image on the screen is checked at the first start (section 20) and changed if needed with `reverse_x`, `reverse_y`, `reverse_z` (motor) and `mirror_x`, `mirror_y`, `swap_xy` (camera).

**Camera rotation.** On the ATR8300 the camera is turned by 90°: the screen horizontal is the stage $Y$ axis, the vertical is $X$. This is handled by `m.swap_xy` (`True` by default on the instrument, `False` in the simulator). With `swap_xy` the step-to-pixel matrix is anti-diagonal,

```math
K = \begin{pmatrix} 0 & k_y \\ k_x & 0 \end{pmatrix},\qquad k_{x,y} = \mp\frac{\delta}{p},
```

where $p$ is the pixel size on the sample and the signs are set by `mirror_x`, `mirror_y`; the frame size in stage axes is $(h, w)$ instead of $(w, h)$. The ATR8300 camera image is upside down (mirrored in both axes); `m.turn_180` (`True` by default on the instrument) turns it by 180° when shooting, and $K$ changes sign, so the arrow directions on the screen stay the same. The orientation parameters (`turn_180`, `swap_xy`, `mirror_*`, `reverse_*`) are remembered for each microscope by its serial number (section 16).

**Micrometres to steps.** A requested move $\Delta$ (µm) becomes an integer number of steps

```math
N = \mathrm{round}\!\left(\frac{\Delta}{\delta}\right),\qquad |N\delta - \Delta| \le \frac{\delta}{2} \approx 2.8\ \text{µm}.
```

If $|N\delta - \Delta| > 0.05$ µm, the reply reports the actual move. A move $|\Delta| < \delta/2$ (giving $N = 0$) is refused (error `E214`), and so is a zero move (`E213`).

---

## 5. Controller protocol

### 5.1. Physical layer

Virtual COM port: 115 200 bit/s, 8 data bits, no parity, 1 stop bit. DTR and RTS are off, as in the stock program (the .NET `SerialPort` class does not raise them by default; pyserial does, so MicroControl turns them off explicitly).

### 5.2. Frame

Every command and every reply is a frame

```math
\mathtt{AA}\ \mathtt{55}\ \ L_{hi}\ L_{lo}\ \ C\ \ D_1 \dots D_n\ \ S,
```

where $C$ is the command code, $D_1\dots D_n$ the data, $L = n + 4$ the length (2 bytes, high byte first), $S$ the checksum:

```math
S = \left(L_{hi} + L_{lo} + C + \sum_{i=1}^{n} D_i\right) \mathrm{mod} 256.
```

The full frame length is $L + 2$ bytes. Example: serial number request `AA 55 00 04 04 08`, $S = 00 + 04 + 04 = 08$.

The reply repeats the command code and carries the data. The controller replies 1–4 ms after receiving a command, **before the motion ends**: a reply to a motion command means "command accepted", not "move finished".

### 5.3. All numbers are integers

Motor steps are 2 unsigned bytes (0…65 535) plus a separate direction byte; illumination level and laser power are 2 bytes; the cooling temperature is a sign byte (`00` or `FF`) and the magnitude in °C. Fractional values are not transmitted. MicroControl converts micrometres to steps as in section 4, and accepts only integers for light and laser: `m.laser = 2.5` is error `E208`, without rounding.

### 5.4. Commands used

| Code | Command | Data | Reply |
|---|---|---|---|
| `04` | serial number | — | text (ASCII) |
| `03` | model | — | text |
| `66` | maximum laser power | — | 2 bytes (`FF 00` — not set) |
| `31` | calibration block | index `09` | calibration data |
| `26` | instrument type | — | 1 byte: 0 — AutoFocus, 1 — Basic, 2 — Mapping |
| `2D` | illumination | level, 2 bytes | — |
| `20` | laser | power, mW, 2 bytes | — |
| `12` | CCD cooling set point | sign, magnitude °C | — |
| `24` | motors on/off | `01` / `00` | — |
| `2E`, `2F`, `2B` | move X, Y, Z | direction, $N$ (2 bytes) | status, 1 byte |

Status code in the reply to a move (the `MotorXResult`, `MotorYResult`, `MotorZResult` enums of the stock program):

| Byte | Meaning | MicroControl action |
|---|---|---|
| `00` | done | — |
| `01`, `02` | limit switch reached | the direction is blocked, the axis position is marked unconfirmed, error `E204` |
| `03` | failure | error `E306`, position unconfirmed |
| `04` | motor still moving (command came too early) | error `E306`, position unconfirmed, no retry: if the command was queued, a retry would double the move |

No other command codes are ever sent to the controller (allow-list; error `E305`). In particular, writing the calibration (`30`), the instrument type (`25`) and the maximum laser power (`65`) is blocked. The cooling set point may only be −5 °C or 20 °C — the stock program values.

### 5.5. Connect and disconnect sequences

| Step | Stock Optosky program | MicroControl | Same |
|---|---|---|---|
| Port | 115200 8N1, DTR = 0, RTS = 0, reply timeout 5 s | same | yes |
| Serial number | `AA 55 00 04 04 08` | same; up to 3 retries if there is no reply | yes |
| Model | `AA 55 00 04 03 07` | same | yes |
| Max laser power | `AA 55 00 04 66 6A` | same | yes |
| Calibration | `AA 55 00 05 31 09 3F` | same | yes |
| Instrument type | `AA 55 00 04 26 2A` | same | yes |
| Light, level 5 | `AA 55 00 06 2D 00 05 38` | same | yes |
| Cooling −5 °C | `AA 55 00 06 12 FF 05 1C` | same (`cooling=False` — do not turn on) | yes |
| Motors on (before the first move on any axis) | `AA 55 00 05 24 01 2A` | same | yes |
| Move X "backward", 36 steps | `AA 55 00 07 2E FF 00 24 58` (recorded) | `AA 55 00 07 2E FF 00 24 58` | byte for byte |
| Pause after a move | 50 ms; does not wait for the motion to end | $t = N\delta/v + \tau$ (section 6) | no, on purpose |
| Disconnect | laser 0 → motors off → light 0 → cooling 20 °C → close the port | same | yes |

Confirmed by a recording: the controller answers the serial number (`ATR30007`) and model (`25312082531`) requests within 1–4 ms.

### 5.6. Reading replies and resynchronisation

The STM32 virtual port driver (2010 version) ignores the read timeout: with no reply, an ordinary read blocks forever. MicroControl therefore does not rely on the driver: it polls the number of received bytes and stops waiting after its own deadline (1 s for working commands, 5 s for connection commands), raising `E301`.

**Port loss.** If the port handle becomes invalid (the USB device reconnected or Windows put it into power saving; Windows error `ClearCommError … The device does not recognize the command`), MicroControl closes the port, opens it again (up to three attempts, 1 s apart), restores the light level and cooling and repeats the command — only if a repeat is safe (read requests, light, laser, motors on). A move is never repeated: it is unknown whether it was executed; the axis position is marked unconfirmed and `E307` is raised. In manual mode, when there has been no exchange for more than 2 s, an instrument-type request is sent (read only, not logged) so that the port is not put to sleep. It is recommended to turn off USB power saving: "Power Options → Change plan settings → Advanced settings → USB settings → USB selective suspend setting → Disabled", and in Device Manager clear "Allow the computer to turn off this device to save power" for the USB root hubs.

A frame with a wrong length leaves the controller waiting for the missing bytes; it then reads the following frames shifted and does not reply until it meets the `AA 55` header in the right position. This is exactly how, before 0.11.1, a calibration request without an index "hung" the controller. On connection the first request is repeated up to three times — enough to resynchronise.

---

## 6. Motion model and timing

Since the controller confirms a command before the motion ends and never reports its end, the end of a move is determined by time. A move of $N$ steps is split into parts of at most $N_{\max} = 180$ steps (1 mm); after each part the program waits

```math
t = \frac{|N|\,\delta}{v} + \tau,
```

where $v$ is the assumed motor speed (`m.motor_speed`) and $\tau$ the settling pause (`m.move_pause`). At least 0.06 s passes between any two commands.

The axes move strictly one after another, never together, in the order: Z down → X → Y → Z up. Thus the rise towards the objective is done last, when the stage is already at the right point of the plane.

**Choice of $v$.** The motor speed is set by the controller; there is no speed command in the protocol and the program does not regulate it. The parameter $v$ is the program's **knowledge** of the speed, used only to compute the wait. An error is safe only one way: with $v$ lower than the true speed, the program waits longer than needed; with $v$ higher, the next command arrives during the motion, the controller answers `04` and does not execute it; the program repeats the command every 0.03 s until the motion ends (at most 5 s), so no steps are lost. A picture, however, may be taken before the stage has stopped, so the default $v$ is 3000 µm/s, 6–7 times below the datasheet value.

**Measuring $v$:** `m.measure_speed(distance=1000)`. The stage moves by `distance` in X; the camera records frames until the image stops changing (difference of neighbouring frames below the $3\sigma$ noise threshold for 1 s). $v = 0.8\,v_{\text{meas}}$ is taken. The result is rejected (`E212`, $v$ unchanged) if the motion is seen in fewer than three frames, if the last frame does not differ from the first, or if $v_{\text{meas}} > 20$ mm/s: in all these cases the speed would be overestimated. Measure in focus, on an area with details, with the light on.

**Scan time.** For a frame step $s$ (section 10) the time per frame is

```math
t_{\text{frame}} \approx \frac{s}{v} + \tau + \tau_{\text{photo}} + t_{\text{exp}} + t_{\text{write}},
```

where $\tau_{\text{photo}}$ is `photo_pause` and $t_{\text{exp}}$ the exposure. With $v = 3000$ µm/s and $s = 570$ µm (10x) the move takes about 0.2 s; after measuring the speed — several times less.

---

## 7. Safety: limits and their reasoning

### 7.1. Physics

Only bringing the sample closer to the objective (the $+Z$ move) is dangerous. Moving in the XY plane does not reduce the gap.

The objective working distance (WD) is the gap between the objective and the sample in focus. Reference values (Olympus):

| Objective | Standard series (MPLN) | Long working distance (LMPLFLN) | Depth of field $\approx \lambda/\mathrm{NA}^2$ |
|---|---|---|---|
| 5x | 20.0 mm | 22.5 mm | ≈ 50 µm |
| 10x, NA 0.25 | 10.6 mm | 21.0 mm | ≈ 8 µm |
| 20x, NA 0.4 | 1.3 mm | 12.0 mm | ≈ 3 µm |
| 100x | 0.21 mm (NA 0.9) | 3.4 mm (NA 0.8) | ≈ 1 µm |

The instrument's 100x objective is marked "L PL" — a long working distance series. Assumed WD values (conservative, `m.working_distance`): 4x — 17 mm, 10x — 10 mm, 20x — 3 mm, 100x — 2 mm. The actual value is printed on the objective barrel (WD) and is set like this: `m.working_distance = 3400`.

**Limit rule.** A rise per command $\Delta z_{\uparrow} \le \min(\mathrm{WD}/3,\ 1000\ \text{µm})$, rounded down to 50 µm: even two wrong commands in a row cannot bring the sample to the objective. The Z travel is $\pm 50\,000$ µm from the connection point by default, i.e. in practice limited by the instrument limit switches (full $Z$ travel is 20 mm); the instrument does not go beyond them by itself. A narrower travel is set with `m.set_limits` (7.2).

### 7.2. Limits

| Limit | Value | Error |
|---|---|---|
| X, Y move per `move` command (`max_move`) | ≤ 2000 µm | `E201` |
| Z rise per command (`max_move_z`) | 4x, 10x, 20x — 1000; 100x — 650 µm (with WD 3.4 mm — 1000) | `E201` |
| Z lowering per command (`max_move_down`) | ≤ 2000 µm | `E201` |
| Trip by `goto` or G-code (`max_jump`) | ≤ 3000 µm | `E210` |
| X, Y travel from the connection point (`xy_limit`) | ±25 000 µm | `E202` |
| Z ceiling | $Z_{\text{conn}} + 50\,000$ µm (changed by `m.set_limits`) | `E203` |
| Z floor | $Z_{\text{conn}} - 50\,000$ µm (changed by `m.set_limits`) | `E202` |

**Changing the travel limits.** `m.limits` shows the current limits; `m.set_limits(x=(from, to), y=(from, to), z=(from, to))` sets them in working coordinates (like `m.position`), µm, for any set of axes:

```python
m.set_limits(z=(-2000, 3000))                    # Z: from −2 mm to +3 mm
m.set_limits(x=(-20000, 20000), y=(-20000, 20000))
```

Requirements: the lower bound below the upper one, the current position inside, no farther than 100 mm each way; otherwise `E208` and the limits do not change. The new limits hold until the end of the session.

There is no minimum step: small steps are safe and necessary (field of view at 20x is about 285 µm, at 100x about 57 µm; depth of field is a few µm).

If a short working distance objective is installed (e.g. 100x with WD 0.21 mm), its WD must be set: `m.working_distance = 210` gives a 50 µm rise per command; it is also wise to set the ceiling explicitly: `m.set_limits(z=(-50000, 100))`.

The ceiling is counted from the position at connection: it is assumed that the stage is in a safe position when connecting. The objective must be given at connection (`E207`), since the Z limits depend on it.

### 7.3. Checks before motion

All limits are checked **before** the first command is sent. A G-code program is checked as a whole by simulating all its commands; on any error the stage does not move and the error names the line.

A wrong parameter value or a typo in a name raises an error at once (`E105`, `E206`, `E208`); for a typo the closest existing name is suggested.

### 7.3a. Limit switches

The controller reply to a move tells **which** limit switch is closed (from OptoSky.dll: X — `ReachForward`/`ReachBackward`, Y — `ReachLeft`/`ReachRight`, Z — `ReachTop` = 1 / `ReachBottom` = 2), not the direction of motion. If the switch lies in the direction of motion, it is a stop: the direction is blocked (`E204`, then `E205`) and the axis becomes uncertain. If the switch is on the opposite side (the stage stood on it at the start and is moving away), the motion continues and the warning `MOV-012` is logged. Autofocus that hits a limit switch shrinks its window and carries on (`AF-003`). `m.confirm_position()` also clears the direction block.

### 7.4. Unconfirmed position

After a limit switch, a lost reply, a `03`/`04` code, and when the position is restored from a previous session, the axis position is considered unconfirmed. In this state `goto`, G-code and scanning are blocked (`E211`); `move` and manual mode work. The block is cleared by `m.confirm_position()` after checking that the stage is where expected.

### 7.5. Locked functions

The laser can be turned on only when connected with `allow_laser=True` and not above `laser_max`. Limit switch search (`m.find_limits()`, `G28`) — only with `allow_homing=True`: the limit switch signal has not been verified on the instrument yet.

---

## 8. Camera and image formation

The camera works through the MindVision SDK (`MVCAMSDK_X64.dll`). On connection, auto exposure and software sharpening (it distorts the autofocus sharpness measure) are turned off, and white balance is fixed (manual), so that the colour of the scan frames does not drift. After every move the first 2 frames are dropped so that the picture has no motion blur.

The pixel size on the sample is $p = a/M$ (`pixel_size`), the field of view $W \times H = N_x p \times N_y p$ (`m.frame_size`).

| Command | Action |
|---|---|
| `m.image()` | current frame (numpy array, BGR channel order) without saving |
| `m.photo(name)` | picture into `photos/<date>/` |
| `m.exposure = 20` | exposure, ms |
| `m.white_balance()` | one-time white balance on the current frame (aim at an empty area) |
| `m.dark_frame()` | dark frame: the light is turned off, 16 frames are averaged |
| `m.empty_frame()` | flat field frame: an empty area, 16 frames are averaged |
| `m.sharpness` | sharpness of the current frame |

Reference frames are saved in `references/` as 16-bit PNG; the pixel value is the mean multiplied by 256 (keeps the precision of averaging).

---

## 9. Autofocus

### 9.0. Autofocus by depth of field (default)

Depth of field of the objective (wave part plus geometric part, $e = 2.2$ µm — sensor pixel, $\lambda = 0.55$ µm):

```math
d = \frac{\lambda n}{\mathrm{NA}^2} + \frac{n\,e}{M\cdot\mathrm{NA}}.
```

| Objective | NA | $d$, µm | search step $d/2$, µm |
|---|---|---|---|
| 4x | 0.10 | 60.5 | 30 |
| 10x | 0.25 | 9.7 | 4.8 → motor step 5.6 |
| 20x | 0.40 | 3.7 | 1.9 → 5.6 |
| 50x | 0.75 | 1.0 | 0.5 → 5.6 |
| 100x | 0.90 | 0.7 | 0.35 → 5.6 |

NA are typical values (`OBJECTIVES` in `config.py`, adjust to your objectives).

`m.focus()`:
1. **Window centre** — the expected $Z$ from the focus points of `m.manual()` (4 nearest, weights $1/r^2$); without them — the current $Z$.
2. **Window** $\pm 4d$ (at least 3 motor steps), **step** $d/2$ (at least the motor step $\delta$). At 10x — 15 frames instead of 35–40.
3. **Upward sweep**; the stage also approaches the best $Z$ from below: the backlash is taken up the same way and the found $Z$ repeats.
4. **Sharpness** — mean of 2 frames, the frame is smoothed ($\sigma = 1$ px: noise is not counted as detail) and divided by the squared mean brightness (independent of exposure).
5. The **peak** is refined by a parabola through $\ln S$ (the sharpness profile is close to Gaussian), then rounded to the motor step.
6. Peak at the window edge — the window moves there (up to 2 times). Flat curve (peak/minimum < 1.15) — focus is far away, a wide search is done (9.2).

Wide search directly: `m.focus(wide=True)`.

**Stage limit.** If the $Z$ motor step is larger than the depth of field ($\delta = 5.56$ µm at 20x, 50x, 100x), the stage may stop up to $\delta/2 = 2.8$ µm from the sharpest point, i.e. 0.8–4 depths of field; `m.focus()` prints a "Limit" note. The $Z$ step $\delta$ is assumed equal to the X, Y step; if the $Z$ drive has a finer step, set it separately.

Simulator check (10x, $Z$ backlash 3 steps = 17 µm, focus between steps): from 5 start positions between −60 and +70 µm the stage stops at the step closest to focus (error 0.4–0.6 step, the minimum possible).

### 9.1. Sharpness measures

Sharpness is computed on the central half of the frame (each side), converted to grey $I(x, y)$ and reduced $k$ times:

| `focus_method` | Measure $F$ |
|---|---|
| `laplacian` (default) | $\mathrm{Var}\!\left(\nabla^2 I\right)$ — variance of the Laplacian |
| `tenengrad` | $\langle G_x^2 + G_y^2 \rangle$, $G_{x,y}$ — Sobel gradients |
| `brenner` | $\langle (I(x+2, y) - I(x, y))^2 \rangle$ |
| `variance` | $\mathrm{Var}(I) / \langle I \rangle$ — normalised variance |

### 9.2. Wide search

The depth of field may be smaller than the search step (10x: about 8 µm with a 20 µm step), and fine details are sharp only in a narrow $Z$ interval. So the search has two passes.

1. **Coarse.** From $\min(Z_0 + R, Z_{\text{ceiling}})$ down to $\max(Z_0 - R, Z_{\text{floor}})$ with step $s_f$, where $R$ is `focus_range` and $s_f$ is `focus_step`; the measure is computed on a frame reduced $k = 8$ times. Large details are visible even out of focus, so the maximum is wide and is not missed. The pass stops when at least 4 points are done, the current $Z$ is $2 s_f$ below the best one and $F < 0.6\,F_{\max}$.
2. **Fine.** The interval $\pm s_f$ around the maximum with a step of one motor step $\delta$, measure on a frame reduced $k = 2$ times.

The stage approaches the best position from above: it first rises $s_f$ higher, then goes down. This removes the mechanism backlash.

**Peak contrast**

```math
C = \frac{F_{\max}}{\mathrm{median}(F)}
```

over the coarse pass points. With $C < 1.3$ the maximum is considered uncertain (few details in the frame): the reply has a warning and the log a WARNING record.

Default window and step:

| Objective | `focus_range`, µm | `focus_step`, µm |
|---|---|---|
| 4x | ±2000 | 100 |
| 10x | ±1000 | 50 |
| 20x | ±500 | 25 |
| 100x | ±100 | 10 |

### 9.3. Z resolution limit

The Z motor step $\delta = 5.56$ µm is larger than the depth of field of 20x (≈ 3 µm) and 100x (≈ 1 µm) objectives. If the Z travel per step on the instrument really equals $\delta$, the position of best sharpness may coincide with no reachable stage position on these objectives. This needs checking on the instrument (section 24).

---

## 10. Area scanning

### 10.1. Area and frame grid

The area is given by two corners $A = (x_A, y_A)$ and $B = (x_B, y_B)$ — the centres of the corner frames, usually with keys `1` and `2` in manual mode (section 15); or explicitly: `m.area = mc.Area(x0, y0, x1, y1)`.

The grid step for overlap $q$ (`overlap`, $0 \le q \le 0.5$; by default 20 %, 25 % at 50x and 100x — `SCAN_OVERLAP`) is rounded down to a whole number of motor steps, so there are no gaps between frames:

```math
s_x = \delta \left\lfloor \frac{W(1 - q)}{\delta} \right\rfloor,\qquad s_y = \delta \left\lfloor \frac{H(1 - q)}{\delta} \right\rfloor.
```

Number of frames

```math
n_x = \left\lceil \frac{|x_B - x_A|}{s_x} - 0.05 \right\rceil + 1,\qquad n_y = \left\lceil \frac{|y_B - y_A|}{s_y} - 0.05 \right\rceil + 1;
```

the 0.05 tolerance avoids an extra frame due to rounding. The path is a "snake" (`path='snake'`, even rows reversed) or row by row (`path='rows'`).

### 10.2. Focus during a scan

| `focus` | Method | When |
|---|---|---|
| `'plane'` | autofocus at the points of a sparse grid (every `focus_every` frames, plus the area corners); a plane $z = \alpha x + \beta y + \gamma$ is fitted through the points by least squares; the $Z$ of each frame is taken from it | flat samples (wafers, glass): the sample tilt is taken into account |
| `'each'` | autofocus in every frame | uneven samples |
| `'auto'` (default) | if focus points were set in `m.manual()` — as `'points'`, otherwise as `'off'` | — |
| `'points'` | $Z$ of each frame from the focus points of `m.manual()` (corners `1`, `2` and the `F` key): 4 nearest points with weights $1/d^2$, $z(\mathbf{r}) = \sum w_i z_i / \sum w_i$; no autofocus | tilted substrate: $Z$ changes smoothly, in small steps |
| `'off'` | $Z$ does not change: focus is set once in `m.manual()` | whenever the sample is flat: a $Z$ change shifts the picture in X and Y (backlash, axis skew) and spoils the stitching |

Points with contrast $C < 1.3$ are not used for the plane (the number skipped is reported). With fewer than three good points the plane is horizontal; with none the scan does not start (`E603`). If the points deviate from the plane by more than `focus_step`, `focus='each'` is recommended.

**Re-shooting blurred frames** (`refocus=True`). Frame $k$ is blurred if

```math
F_k < 0.5\,\mathrm{median}(F_{k-20}, \dots, F_{k-1})
```

(with at least 5 previous frames). Such a frame is shot again after autofocus; the flags `refocused` and `blurry` are written to `index.csv`.

### 10.3. Execution

`m.plan()` builds the G-code scan program (it can be read and edited); `m.scan()` runs it. Before planning, the camera checks the axis signs (`m.check_orientation()`: a shift of 20 % of the frame in X and Y and back; with a wrong sign the frames would be stitched in reverse order) and fixes `swap_xy`, `mirror_x`, `mirror_y` if needed; turn off with `m.scan_check = False`. By default frames are saved as JPG in `frames/` (`frame_format = 'jpg'`, quality `jpg_quality` = 90) reduced to `frame_scale` = 0.5: a 0.44 µm pixel at 10x is still 3 times finer than the optical resolution ($\approx 0.61\lambda/\mathrm{NA} \approx 1.3$ µm), so no detail is lost while the volume is about 30 times smaller; lossless — `m.frame_format = 'h5'`, `m.frame_scale = 1`. For each frame: move to the coordinates (in real coordinates, `G53`, so changing the working zero after planning does not shift the shooting), $Z$ from the focus model, pause `photo_pause`, stillness check (two frames must match within 0.25 px), picture. The exposure is fixed for the whole scan. `m.scan()` returns the scan folder path.

---

## 11. Data storage

```
scans/<date>_<time>_<objective>_<name>/
  meta.json          objective, pixel size, field of view, motor step, step→pixel matrix,
                     area, camera and exposure, light, focus settings, pauses, speed, reference frames,
                     zero offset, instrument info, state, number of frames
  index.csv          one line per frame: id, batch, slot, row, col, x_um, y_um, z_um, x_p, y_p, z_p,
                     sharpness, refocused, blurry, t_s
  program.gcode      the executed program
  frames/00001_r001_c001.jpg   frames as JPG (default, frame_format = 'jpg')
  batches/b0000.h5   or batches of batch_size frames (HDF5, frame_format = 'h5')
  join/              stitching results
  viewer/            viewer caches: frame thumbnails and reduced copies (levels/)
  grid.json          crosses, square selection, square numbers, hints
  export/            exported squares (PNG/JPG, DXF, JSON)
```

Each `bNNNN.h5` file has two datasets: `tiles` — lossless frames (gzip), split into $4 \times 4$ chunks (any part of a frame is read without unpacking the whole frame); `thumbs` — copies reduced 16 times for the viewer. A failure during writing damages at most one batch.

Access to the frames:

```python
st = mc.ScanData(folder)
ids = st.query(x=(0, 2000), y=(0, 1000))      # frames touching a rectangle (µm)
img = st.frame(ids[0], crop=0.5)              # central half of the frame
for i, img in st.frames(ids): ...             # iterate, reading batch by batch
st.export()                                    # PNG files and TileConfiguration.txt for Fiji
```

---

## 12. Stitching and marking (main workflow)

**Workflow.**
1. `m.manual()`: the green area — corners `1`, `2`.
2. Focus at the corners: at each corner bring the image into focus (`E`/`Q`) and press `1`/`2` (area corners) and `F` (other corners). The "Autofocus by corners" checkbox (`Y`, on by default): $Z$ of each frame from the 4 nearest points with weights $1/d^2$.
3. `folder = m.scan()` — frames with 20 % overlap, 25 % at 50x and 100x (`scan_overlap` in `config.py`). Before each picture two consecutive frames are compared by phase correlation: if the picture still moves by more than 1 px (the stage has not settled), the picture is postponed (`m.still_check`). Exposure and white balance are manual for the scan (they do not drift from frame to frame).
4. `mc.stitch(folder)` — exact frame positions from the overlaps (12.1). Once per objective — `mc.distortion(folder)` (12.5).
5–7. `mc.view(folder)` — crosses (`G`), level view (`R`), square selection (`S`, `B`, `P`), square numbers (`N`), hints (`Y`) (12.2).
8. `mc.export(folder)` — only the selected squares: PNG (transparent background) or JPG, DXF for AutoCAD (12.3).
9. `mc.clean(folder)` — show, `mc.clean(folder, confirm=True)` — delete the frames and other work files.

The whole mosaic is never assembled in memory: the viewer and the export take only the needed parts from the frames by the found positions.

### 12.1. Overlap stitching (`mc.stitch`)

The method follows ASHLAR (Muhlich et al., *Bioinformatics* 38, 4613, 2022; MIT licence), without Java.

1. **Neighbours** — pairs of frames whose rectangles predicted by the stage overlap by at least 30 % of the largest overlap (diagonal corners are dropped).
2. **Pair shift** — phase correlation over the common part of the frames. The strips are filtered by a Laplacian of Gaussian ($\sigma = 1$ px for a pixel < 0.5 µm: details stay, the smooth background goes), multiplied by a Hann window and zero-padded twice (no wrap-around of the correlation). The seam error $e = 1 - \mathrm{NCC}$ after alignment. A shift farther than `max_shift_um` = 40 µm from the prediction is dropped.
3. **Permutation test.** The same geometry, but a distant frame instead of the neighbour: errors of surely false pairs. The threshold is their 1st percentile.
4. **Stage → pixels model** from the accepted seams: $\mathbf{d}_{ij} = M\,\Delta\mathbf{p}_{ij} + B\,\Delta\mathbf{s}_{ij}$. $M$ ($2\times2$) accounts for the camera rotation relative to the stage axes, an inexact motor step and skew; $\mathbf{s}_k = (\mathrm{sign}\Delta x, \mathrm{sign}\Delta y)$ is the side from which the stage approached frame $k$, $B$ the backlash (offset when the direction changes, reported in µm). A frame without reliable seams is placed by this model.
5. **Positions** — weighted least squares

```math
\min \sum_{(i,j)} w_{ij}\,\bigl|\mathbf{r}_j - \mathbf{r}_i - \mathbf{d}_{ij}\bigr|^2 + \varepsilon\sum_i \bigl|\mathbf{r}_i - M\mathbf{p}_i\bigr|^2,\qquad \varepsilon = 0.01,
```

with outlier suppression: $w_{ij} = w^0_{ij}/\bigl(1 + (\rho_{ij}/c)^2\bigr)$, $\rho$ the seam residual, $c$ decreasing from $\infty$ to 2 px (first a rough solution, then inconsistent seams lose weight). $w^0 = 1 - e$. A frame without reliable seams is held by the stage model.

Simulator check (substrate turned by 1.2°, motor step off by +1.5 / +1.2 %, drift ±1 step per move, 42 frames): frame position error by the stage — up to 63 px; after stitching — 3.3 px (RMS), seams 0.15 px (median). Real ASHLAR (`mc.stitch(folder, engine='ashlar')`, needs `pip install ashlar` and Java) on the same frames — 9.8 px.

**Optical corrections** (applied when frames are read for stitching, viewing and export):
- *illumination*: field $F(x, y)$ — median of up to 40 frames, smoothed, $\langle F\rangle = 1$; frame $I' = I/F$ (`join/flatfield.npy`, computed at every stitch; `flat=False` — without it);
- *distortion* (12.5), if it was measured for this microscope and objective.

Result: `join/positions_all.csv`, `join/stitch.json`, `join/preview_all.jpg` (reduced mosaic), `viewer/levels/` (frames reduced 2, 4, 8 times — for the viewer on large scans).

### 12.2. Marking in the viewer (`mc.view`)

| Key / mouse | Action |
|---|---|
| drag with the right or middle button — in any mode; with the left one — in the look mode, `G`, `P`; `Ctrl`+left — in `S` and `B` | move |
| wheel, `+` / `-`, `0` — whole scan | zoom |
| `G` | crosses: click a cross, then its neighbour in the same row — the line through them gives the tilt, and the whole picture turns at once so that the rows are level; far crosses (anywhere) refine the tilt and the pitch; the cross centre is found by itself; right button — remove a mark, `C` — start again |
| `R` | level view on / off: the picture is turned by $-\theta$, the squares stand level |
| `S` | select squares: click — a square; box (drag) — all squares it **touches** (no need to reach the centres); with `Shift` — unselect |
| right click on green | remove this whole green area (connected squares); in the look mode, `S`, `B`, `P` |
| `B` | brush: drag over the squares — all squares under the cursor are selected; with `Shift` — rub out |
| `N` | square numbers: click a square, type two numbers `X Y`, `Enter` (with no digits — confirm the shown number); right button — remove a label |
| `Y` | yellow hint on / off; `Shift`+`Y` or `Ctrl`+`Y` — select all yellow squares (make them green) |
| `U`; `<` / `>` | hint sensitivity 0…100: type a number; −5 / +5 |
| `P` | polygon: clicks — vertices, `Enter` — select inside, `Del` — unselect, right button / `Backspace` — remove a vertex |
| `A` / `X` | select all / none |
| `E` | export the selection |
| `Esc` | leave the mode, then the viewer |

The lattice is always **square**: $\mathbf{v}_2$ is $\mathbf{v}_1$ turned by 90° (no skew, no "projection"); there are four unknowns — origin $\mathbf{O}$, pitch and tilt — found by least squares from the marks. Two neighbouring crosses are enough; far crosses reduce the tilt error in proportion to the distance. The `F` key optionally fits the same square lattice to all crosses of the mosaic. Simulator check (tilt 3.5°): 2 neighbouring crosses and 2 far ones — pitch 99.998 µm, tilt 3.494°. Marks and selection are stored in `grid.json` (in mosaic pixels; after a new stitch the marks must be made again).

**Cross centre** is found by a "+" template with a dark rim (it fits thin dark crosses as well as raised light ones with an outline); each candidate is checked for the cross symmetry under 90° and 180° rotation — the lithography numbers next to a cross are not symmetric this way and do not pull the centre. The cross size (arm, px) is chosen automatically when the lattice is fitted to all crosses: the one with which the found centres lie best on the lattice, stored in `grid.json` (`arm_px`). Check on sample images with numbers at the crosses: lattice over 13 crosses — 0.52 px RMS, over 11 crosses — 1.1 px. On a whole 4x frame the crosses deviate from a straight lattice by up to ~7 px towards the edges — this is lens distortion (`mc.distortion`).

**Square numbers.** Lattice indices $(i, j)$ are arbitrary (counted from the first mark). The lithography numbers are obtained as

```math
\begin{pmatrix}X\\Y\end{pmatrix} = P\begin{pmatrix}i\\j\end{pmatrix} + \mathbf{t},
```

$P$ is a signed permutation of the axes (8 variants: which lattice axis is $X$ and which way it grows). $P$ and $\mathbf{t}$ are chosen so that all labels hold exactly. One label gives only $\mathbf{t}$ (the directions are the defaults: $X$ to the right, $Y$ down on the screen); a label on a second square **in another row and column** fixes the directions of both axes — and the numbers of all squares. The number is written in the top left corner of each square (labelled ones in yellow); inconsistent labels give a warning. The numbers go to `selection.json` (`names`), to the DXF (layer LABELS) and to the file names of the squares (`square_X+012_Y+007.png`). No digit recognition is needed: two labels are enough.

**Yellow hint** — squares with something in them (flakes, crystals). A simple method for a weak PC: the whole mosaic as one reduced image (a square ≈ 96 px, smaller for a large grid — ≤ 6 Mpx in total), each square is straightened, the background is the median colour of the square (uneven illumination does not matter), "something" are pixels that differ from the background in Lab by more than $\max(12,\ 4 \cdot 1.48\,\mathrm{MAD})$ (the threshold rises on a noisy background). Not counted: the square corners (crosses and numbers), dark colourless spots (dust), spots smaller than $0.1\,\%$ of the square. A square is yellow if the share of "something" is larger than

```math
f(S) = 5\,\% \cdot 10^{-2S/100},
```

$S$ — sensitivity 0…100 (50 — 0.5 %; 0 — 5 %; 100 — 0.05 %). Computed automatically when the lattice is set (≈ 5 s per 5000 squares), changing the sensitivity is instant. Without the window: `mc.hints(folder, sensitivity=50)`. On sample images: all 5 squares with flakes found, no empty ones; on a sample without flakes — 20 of 480 (large dust particles, coloured debris), removable by the sensitivity.

**Performance** (30 × 30 frame grid, 900 frames of 1296 × 972, 2 cores): stitching 67 s, memory ≤ 0.6 GB; opening the viewer 0.1 s; a window frame at any zoom 7–63 ms, while dragging 2–25 ms. Small zoom is drawn from the overview mosaic `join/preview_all.jpg`, medium from the reduced frame copies `viewer/levels/` (written at stitching), large from the frames; the copy cache is limited to 400 MB.

### 12.3. Exporting the selection (`mc.export`)

Each selected square (with a 16 px margin) is assembled directly from the frames: pixel $(u, v)$ of square $(i, j)$ is taken from the mosaic point $\mathbf{O} + i\mathbf{v}_1 + j\mathbf{v}_2 + \tfrac{u}{a}\mathbf{v}_1 + \tfrac{v}{b}\mathbf{v}_2$; overlapping frames are blended **by frequency** (Laplacian pyramid, 5 levels): each pixel is given to the frame with the largest weight (the weight falls towards the frame edge), the masks are smoothed per level — fine details without doubling, smooth brightness without a step at the seam. `blend='feather'` — simple weighted blending. The squares are placed in a grid without rotation, joined exactly at the crosses.

- `export/selection.png` — only the selection, the rest is transparent (`fmt='jpg'` — white background);
- `export/selection.dxf` — DXF R12, units µm: layer `SQUARES` — square sides, `OUTLINE` — outer contour of the selection, `LABELS` — square numbers $(X, Y)$;
- `export/selection.json` — list of squares and their numbers, lattice pitch (µm), rotation, output pixel size;
- `cells=True` — also each square as a separate file (`export/squares/`).

Memory is limited by `max_mb` (2000 MB by default): for a large selection the image is reduced.

### 12.5. Lens distortion (`mc.distortion`)

Model (radial, one coefficient): $\mathbf{x}_d = \mathbf{c} + (\mathbf{x}_u - \mathbf{c})(1 + k_1\rho^2)$, $\rho = r/R$, $R$ — half the frame diagonal; the offset in the corner is $k_1R$.

Measurement — by the cross lattice (a ruler with an exact pitch): after `mc.stitch` and marking the crosses in `mc.view`, call `mc.distortion(folder)`. In every frame with two or more crosses the cross centres are found near the lattice nodes; the deviations $\mathbf{e}$ minus their mean over the frame (frame shift) are fitted by least squares with $a(\mathbf{u}-\mathbf{c}) + b\,\mathrm{rot}_{90}(\mathbf{u}-\mathbf{c}) + k_1(\mathbf{u}-\mathbf{c})\rho^2$ (lattice scale and rotation separately from distortion). If the corner offset is ≥ 0.5 px and the correction noticeably reduces the deviations, $k_1$ is saved for this microscope and objective (`~/.microcontrol/devices/distortion_<serial>_<M>x.json`) and applied to all following stitches; after measuring — run `mc.stitch(folder)` again.

Simulator check: the set $k_1 = -0.020$ (8 px in the corner) was measured as $-0.0196$; after the correction seams 0.21 → 0.12 px, lattice pitch 99.1 → 99.9 µm. Without distortion 1.3 px in the corner was measured — considered insignificant, not applied.

### 12.4. Simple mosaic (`mc.join`)

`mc.join(folder)` builds the whole mosaic in five stages (for small scans and training datasets).

1. **Initial positions** of the frames — from the stage coordinates and the step-to-pixel matrix.
2. **Refinement.** For each pair of neighbouring frames the mutual shift $\mathbf{d}_{ij}$ is computed by phase correlation over the overlap strip: the maximum of the inverse Fourier transform of the normalised cross spectrum

```math
R = \frac{\hat I_i \hat I_j^{*}}{|\hat I_i \hat I_j^{*}|}.
```

The shift is accepted if the peak height is at least 0.08 and the correction does not exceed a quarter of the strip; otherwise the stage shift is used.
3. **Global fit** of the positions $\mathbf{r}_i$ by least squares over all pairs at once (errors do not accumulate along a row):

```math
\min_{\{\mathbf{r}\}} \sum_{(i,j)} w_{ij}\,\bigl|(\mathbf{r}_j - \mathbf{r}_i) - \mathbf{d}_{ij}\bigr|^2 + \varepsilon \sum_i \bigl|\mathbf{r}_i - \mathbf{r}_i^{(0)}\bigr|^2,\quad \varepsilon = 0.01,
```

$w_{ij}$ — correlation peak height, $\mathbf{r}_i^{(0)}$ — position by the stage.
4. **Illumination flattening** (with at least 12 frames): field $F(x, y)$ — median of up to 40 frames, smoothed by a Gaussian with $\sigma = \min(W, H)/12$ and normalised to the mean; the frame is divided by $F$.
5. **Blending** with weights falling towards the frame edges:

```math
w(x, y) = \min(x + 1,\, N_x - x)\cdot\min(y + 1,\, N_y - y).
```

Memory is limited (`max_mb = 3000`): if needed, the mosaic is built at a reduced scale; the original frames are not changed. A part of the scan: `mc.join(folder, x=(x0, x1), y=(y0, y1))` or `center=0.3`. Result: `join/mosaic_*.png`, `preview_*.jpg`, `positions_*.csv`.

---

## 13. Preparing images for a training dataset

| Requirement | Reason | Implementation |
|---|---|---|
| Constant exposure | brightness differences must not become a feature | the exposure is fixed for the scan and written to `meta.json` |
| Lossless storage | lossy compression adds artefacts | HDF5 + gzip (`frame_format = 'h5'`, `frame_scale = 1`); camera software sharpening is off |
| Physical scale | object size in µm is a feature | `pixel_size_um`, frame coordinates in µm |
| Sharpness | blurred frames hurt training | focus model, re-shooting, `sharpness` and `blurry` for every frame |
| Illumination correction | vignetting, non-uniformity | reference frames $D$ and $E$ are saved; the frames themselves are not changed |
| Reproducibility | repeating the acquisition | `program.gcode`, all parameters in `meta.json`, link to the log |

**Field correction** when preparing the data:

```math
I' = \frac{I - D}{E - D}\,\langle E - D \rangle,
```

where $I$ is the frame, $D$ the dark frame (`m.dark_frame()`), $E$ the frame of an empty area (`m.empty_frame()`), $\langle\cdot\rangle$ the mean over the frame.

Recommendations:
1. Before a series: warm up the illumination for 2–5 min, `m.white_balance()`, `m.empty_frame()` on a clean substrate, `m.dark_frame()`. Repeat when the objective or light level changes.
2. Do not change exposure, light and white balance within a series.
3. Overlap: for overlap stitching use the default 20 %; for labelling objects on frame borders — 0.1–0.2 is enough.
4. Split into training and validation sets by areas or samples, not by random frames: neighbouring frames are correlated, and a random split overestimates the quality.
5. Exclude frames with `blurry = 1`.

---

## 14. G-code programs

Control in the spirit of CNC machines and 3D printers. Units — µm. A comment follows `;`. `m.gcode(program, save='name')` — run a string or a `.gcode` file; with `save`, `M240` pictures go to a new scan folder.

| Command | Parameters | Action |
|---|---|---|
| `G0`, `G1` | X Y Z | move |
| `G53 G0` | X Y Z | move in real coordinates |
| `G4` | P (ms) or S (s) | pause |
| `G28` | X Y Z | limit switch search (needs `allow_homing=True`) |
| `G29` | — | build the focus plane for the area |
| `G30` | — | autofocus |
| `G90` / `G91` | — | absolute / relative coordinates |
| `G92` | X Y Z | zero offset: the current point gets these working coordinates |
| `G92.1` | — | remove the zero offset |
| `M3` / `M5` | S (mW) | laser on / off |
| `M114` | — | print the position |
| `M240` | R C | picture (row R, column C) |
| `M355` | S1 P(level) / S0 | light |
| `M420` | S0 / S1 | focus plane off / on |

The `F` (feed) parameter is not supported (`E902`): the speed is set by the controller (section 6). Before execution the program is checked as a whole (section 7.3).

---

## 15. Manual mode and viewer

`m.manual()` opens the camera image on the full screen; `m.manual(fullscreen=False)` — in a window. The `manual` and `view` windows have the same size: `WINDOW_SIZE` = 1280 × 900 by default (config), then the last size the window was stretched to.

The camera is read in a separate thread: the window is redrawn at once (pan, zoom, text) without waiting for the next frame; during a move, focusing and a picture the thread pauses. All text and fields are always visible, the mini map is always shown, slightly transparent.

| Key | Action |
|---|---|
| arrows, WASD | move the field of view by Step X / Step Y (one frame by default) |
| `Tab` or click on a field | edit Step X, Y, Z (µm), exposure (ms, fractions with `.` or `,`) and objective (4, 10, 20, 50, 100); digits, `Backspace`, `Enter` — accept, `Esc` — cancel |
| `G` | Steps X, Y = one frame |
| `M` | next objective; Steps X, Y — one new frame |
| `E` / `Q` (PgUp / PgDn) | stage up / down by Step Z (100 µm by default) |
| `Shift`+`E` / `Shift`+`Q` | the same, by five Steps Z (500 µm by default) |
| `F` | keep $Z$ here as a focus point (3, 4, …) |
| `Y` (or the checkbox) | autofocus by corners during the scan: $Z$ of a frame from the focus points |
| `Shift`+`F` | autofocus |
| wheel, `+` / `-`; `Z` | zoom (to the cursor / to the centre); whole frame |
| drag the picture (any button; the left one — not on the fields) | move the zoomed picture |
| `1` / `2` | area corner A / B; $Z$ at that moment is kept as focus point A / B |
| `C` | delete the area and focus points |
| `0` | working zero here (`m.offset()`); the area corners stay at their place on the sample |
| `R` | remove the zero offset |
| `B` | white balance on the current frame |
| `L` | light on / off |
| Space | picture |
| `H` | help |
| `Esc` | exit; the area is written to `m.area` |

**Step fields** (top left, under the coordinates): Step X, Step Y, Step Z, µm. Only multiples of 5 µm are allowed; the number is rounded to the nearest multiple of 5 and capped by the per-command limit (`max_move` for X, Y; for Z the smaller of `max_move_z`, `max_move_down`). The steps are kept between entries into manual mode within a session.

**Directions.** The left/right and up/down arrows move the field of view on the screen regardless of the camera rotation: the program picks the stage axis itself (with `swap_xy` the horizontal is the $Y$ axis, the vertical is $X$) and the sign. If a direction on the screen is reversed, fix it with `m.mirror_x = True` or `m.mirror_y = True` (remembered for the microscope).

One key press — one move: key presses accumulated during a move (auto-repeat while the key is held) are dropped.

**Mini map** (top right) — a window of $11 \times 9$ frames, a "room": while the stage is inside, the window stands still and the field-of-view frame moves across it; when the stage reaches the edge (the last cell), the window turns a page and the stage appears at the opposite edge of the new room. If the area fits in the window, the window stands on it. It can be flipped with `[` (left-right) and `]` (up-down), remembered for the microscope (`m.map_flip_x`, `m.map_flip_y`; both `True` by default on the ATR8300). Flipping changes only the map picture, not the motion or stitching. Blue dots are the focus points.

- **Grey grid** without borders fills the whole window: each cell is a frame (step — field of view rounded down to the motor step). The grid is tied to corner A, before it is marked — to the working zero. Only visible cells are drawn.
- **Row and column numbers** — always, at every visible cell: from the first frame of the area (preview or fixed), without an area — from the working zero; the stage row and column numbers are yellow and repeated under the map and in the status line.
- **White frame** — the field of view, **yellow fill** — exactly what is inside it (not the nearest cell).
- **Area preview**: after the `1` key the green area stretches from A to the stage cell; at the bottom — "Preview $n_x \times n_y = N$".
- **Fixed area**: the `2` key puts corner B exactly in the centre of the current grid cell; at the bottom — "Area $n_x \times n_y = N$", along the edges — the column and row numbers of the area.
- If the area is larger than the window: at the bottom — the number of area frames hidden on each side ("hidden <9 >10 ^6 v5"), under the map — an overview of the whole area with the visible window.
- Corner A marked without B is kept until the next entry into manual mode.

The mini map grid is the scan frames with `SCAN_OVERLAP` overlap (20 %; 25 % for 50x and 100x).

The stage coordinates are shown large in the top left corner; the status line at the bottom also has the frame number in the area ("Frame: row $r$/$n_y$, column $c$/$n_x$").

**Scan viewer:** `mc.view(folder)` — move: drag with the mouse, arrows or WASD (faster with Shift), double click — put the spot in the centre; zoom — wheel, `+` `-`, PgUp / PgDn; `0` — the whole scan; `Esc` — exit. While the picture is dragged it is drawn from reduced copies (smoothly), afterwards in full quality. Marking keys — section 12.2.

---

## 16. Microscope home and position memory

**Home** is the permanent origin of the real coordinates of a given instrument. It is set once with `m.set_home()` at a chosen point (e.g. at a noticeable mark). Saved: the position, objective, light, exposure and the image of that place. A repeated command is refused (`E215`); to set it again — `m.set_home(replace=True)`.

The memory is kept **for each instrument separately** by model and serial number: `~/.microcontrol/devices/<model>_<serial>.json` and `..._home.png`. The position is written after each part of a move; before a move a "moving" flag is written.

On the next connection the position is restored **as unconfirmed** (section 7.4): between sessions the stage could be moved by hand or by the stock program. If the "moving" flag is still set (the session was interrupted during a move), a warning is shown.

**Check:** `m.go_home()` moves the stage to $X = Y = 0$ and compares the image with the saved one. The image shift $\mathbf{s}$ (pixels) is found by phase correlation and converted to the stage offset $\mathbf{e} = K^{-1}\mathbf{s}$ (steps), where $K$ is the step-to-pixel matrix. `m.go_home(correct=True)` corrects the coordinates by this shift. Conditions: the same sample and the same objective. By default $Z$ does not change: the focus height of another sample may differ; `go_home(z=True)` also moves $Z$, raising the stage in parts of at most `max_move_z`.

---

## 17. Emergency stop and exclusive access

**Exclusive access.** Only one program controls the instrument at a time: on connection a file lock `~/.microcontrol/control_<port>.lock` is taken (process id, time, folder). A second connection is refused (`E106`) with the owner named. Check: `python -m microcontrol status`.

**Emergency stop** — from any command line, including when the notebook does not respond:
```
python -m microcontrol stop
```

1. The file `~/.microcontrol/STOP` is created. A running program checks it every 0.05 s while waiting; when found, it sends "laser 0" and "light 0", closes the port, releases the lock and ends with error `E110`.
2. If the lock is not released within 3 s (the program hangs), the process is terminated.
3. The port is opened again, "laser 0" and "light 0" are sent.
4. If the port is held by another process (Windows: `Access is denied`), all processes that can hold it are terminated: MicroControl processes (found by an open log `logs/MC_*.log` or the lock file; this includes "hung" Jupyter kernels with a connected microscope) and the stock Optosky software (`OptoskyRamanMicroscope.exe`, `RamanMicroscope.exe`). Then step 3 is repeated. If there are not enough rights to terminate a process (it runs as administrator or another user), run the same command in an administrator console.

The list of such processes without stopping them: `python -m microcontrol holders`. Exact detection needs the `psutil` package (in `requirements.txt`); without it the processes are found by command line (Python with `microcontrol`, `ipykernel`, `jupyter`).

Limitation: the controller has no "stop" command. The part of a move already sent (at most 1 mm) will be executed. The last resort is the instrument power switch.

---

## 18. Simulator

`mc.Microscope('sim', objective=10)` — a virtual instrument with the same commands, limits, log and errors. For learning and for testing programs without the instrument.

**Sample.** A substrate with roughness, particles and flakes of different thickness; on it — gold crosses with a 100 µm period (span 20 µm, line width 2.5 µm), every fifth twice as large. The sample is defined in micrometres, so the image scale matches the objective. The sample plane is tilted: $\partial Z/\partial X = 0.02$, $\partial Z/\partial Y = -0.015$. The focus at the start point is 100 µm above the initial stage position. Test options: `m._world.turn_deg` (lattice rotation), `pitch`, `backlash` (per axis), `dist_k1` (lens distortion), `vignette`, `flakes` (bright flakes for the hint).

**Optics.** Defocus is modelled by a Gaussian blur with

```math
\sigma = \frac{\mathrm{NA}\,|\Delta z|}{2p}\ \text{pixels},
```

matching the blur circle radius $\mathrm{NA}\,|\Delta z|$; sensor noise is normal, $\sigma_n = 2$ levels, new in every frame.

**Laser** 532 nm — a spot in the centre of the frame with

```math
s = \frac{0.61\,\lambda}{2\,\mathrm{NA}} + \frac{\mathrm{NA}\,|\Delta z|}{2},
```

brightness proportional to $P/s^2$. With a power of at least 20 mW and $s < 3$ µm a mark (burn) is left on the sample.

**Motors:** speed about 5 mm/s, a move takes time; a command arriving during a move gets reply `04` and is not executed. Limit switches at ±25 mm. Before the "motors on" command (`24 01`) no moves are executed. A calibration request without an index gets no reply, as on the instrument.

The simulator remembers the "stage" position between sessions in the same folder, like the real instrument.

---

## 19. Installation

**Python 3.13** is required (3.10–3.14 work; there are no ready `h5py` builds for 3.15 yet).

```
cd /d D:\MicroControl
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt jupyter
jupyter notebook
```

Then on every start: `cd /d D:\MicroControl`, `.venv\Scripts\activate`, `jupyter notebook`.

Dependencies: `pyserial`, `numpy`, `opencv-python`, `h5py`, `psutil`. The MindVision camera SDK (MindVision Camera Platform Setup 2.1.10.161) is installed from the instrument driver kit.

---

## 20. First start on the instrument

1. Close the Optosky program (its process must not remain in the Task Manager).
2. Check the link: `python -m microcontrol probe`. Replies to all 5 connection requests must come.
3. Connect without the camera: `m = mc.Microscope(objective=10, camera=False)`; expect the Mapping type and the cooling fan noise. Then `m.move(dx=200)`.
4. Connect with the camera, `m.manual()`. Key `Q`: the stage must go **down** (the image blurs slightly). If it goes up — `Esc` at once, then `m.reverse_z = True`.
5. Arrow `→`: the image must shift **left** by a frame width; likewise `↓` — up. If not — `mirror_x`, `mirror_y` or `reverse_x`, `reverse_y`.
6. `Shift`+`F` — autofocus; `Esc`; `m.measure_speed()` in focus on an area with details.
7. Check the emergency stop: while `m.manual()` runs, execute `python -m microcontrol stop` in another window.
8. Set the home at a noticeable mark: `m.set_home()`. Disconnect, connect, `m.confirm_position()`, `m.go_home()` — expect "matches the home photo".
9. Mark the corners of a small area (`1`, `2`), `folder = m.scan()`, `mc.stitch(folder)`, `mc.view(folder)`.
10. For smeared frames: `m.move_pause = 0.5`, `m.photo_pause = 0.6`.

---

## 21. Troubleshooting

On connection each stage is printed: `Port … opening`, `Controller …`, `Camera opening …`. The line where the process stopped points to the cause; so does the last line of the log in `logs/`.

| Symptom | Cause | Action |
|---|---|---|
| Stops at `Port … opening`, then `E108` | the port driver does not respond (often after a previous failure) | reconnect the controller USB, restart the kernel |
| `E102`, the Optosky program says "Access to the port … is denied" | the port is held by another program | `python -m microcontrol status`; `python -m microcontrol stop` (in an administrator console if needed) |
| `E301` on connection | the controller does not respond or is out of sync | close Optosky, reconnect USB, `python -m microcontrol probe` |
| Stops at `Camera opening`, then `E505` | the camera is held by another program or USB hung | close Optosky and the MindVision program, reconnect the camera USB; work without the camera: `camera=False` |
| `E307`; Windows error `ClearCommError … does not recognize the command` | the port was lost: USB reconnected or was put to sleep by Windows | the link is restored automatically; if it repeats — turn off USB power saving (section 5.6) |
| `E306` "still moving" | the move did not end within 5 s | lower `m.motor_speed`, check the stage, `m.confirm_position()` |
| `E205` after a rise | the previous move in this direction ended on a limit switch | go the other way or `m.confirm_position()` (clears the block) |
| `E211` | the position is not confirmed | check the position, `m.confirm_position()` |
| A Jupyter cell cannot be interrupted | hang inside the driver | `python -m microcontrol stop` or restart the kernel |
| `cv2` has no `getMouseWheelDelta` | an old OpenCV build | nothing to do: MicroControl reads the wheel by itself |

COM port interceptor programs (e.g. com0com) may hold or rename ports: the controller (STMicroelectronics Virtual COM Port) must be visible in Device Manager under its own number.

---

## 22. Reference

### 22.1. Connection

```python
m = mc.Microscope(port='auto', objective=10, camera=True, allow_laser=False,
                  allow_homing=False, cooling=True, folder='.', sdk_folder=None)
```

| Argument | Meaning |
|---|---|
| `port` | `'auto'` (search by VID/PID), `'COM3'`, `'sim'` |
| `objective` | magnification, required: 4, 10, 20, 50, 100 |
| `camera` | `False` — without the camera |
| `allow_laser`, `allow_homing` | allow the laser, the limit switch search |
| `cooling` | turn on the CCD cooling, like the stock program |
| `folder` | working folder (logs, pictures, scans) |

### 22.2. Parameters

Read `m.photo_pause`, change `m.photo_pause = 0.5`, table of all parameters `m.settings`.

| Parameter | Default | Allowed | Meaning |
|---|---|---|---|
| `objective` | — | 4, 10, 20, 50, 100 | objective |
| `light` | 5 | integers 0…5 | illumination |
| `laser` | 0 | integers 0…`laser_max`, mW | laser |
| `laser_max` | 10 | 0…100 mW | maximum allowed power |
| `exposure` | 20 | 0.05…1000 ms | exposure |
| `pixel_size` | $2.2/M$ | 0.001…10 µm | pixel on the sample |
| `mirror_x`, `mirror_y` | False (on the instrument `mirror_y` = True) | — | image mirrored in X / Y; remembered for the microscope |
| `swap_xy` | True (instrument), False (simulator) | — | camera turned by 90°; remembered for the microscope |
| `turn_180` | True (instrument), False (simulator) | — | the frame is turned by 180° (the ATR8300 camera gives an upside-down image); remembered |
| `reverse_x`, `reverse_y`, `reverse_z` | False | — | the motor turns the other way |
| `motor_step` | 5.556 | 0.5…50 µm | $\delta$ |
| `max_move` | 2000 | 1…2000 µm | X, Y per `move` command |
| `max_move_z` | from WD | 1…1000 µm | rise per command |
| `working_distance` | by objective | 100…40 000 µm | working distance; sets `max_move_z` |
| `max_move_down` | 2000 | 1…20 000 µm | lowering per command |
| `xy_limit` | 25 000 | 100…50 000 µm | X, Y travel |
| `z_max` | +50 000 | up to +100 000 µm | ceiling (same as `set_limits(z=…)`) |
| `z_min` | −50 000 | down to −100 000 µm | floor |
| `motor_speed` | 3000 | 50…20 000 µm/s | $v$ |
| `max_jump` | 3000 | 100…25 000 µm | `goto` / G-code trip |
| `move_pause` | 0 | 0…10 s | $\tau$ |
| `photo_pause` | 0 | 0…10 s | pause before a picture |
| `light_pause` | 0 | 0…120 s | pause after the light is turned on |
| `focus_method` | laplacian | see 9.1 | sharpness measure |
| `focus_range` | by objective | 1…2000 µm | $R$ |
| `focus_step` | by objective | 1…500 µm | $s_f$ |
| `focus_every` | 5 | integers 1…100 | focus plane point period, frames |
| `refocus` | False | — | re-shoot blurred frames with autofocus (changes $Z$) |
| `frame_format` | jpg | jpg, h5 | frames as JPG files or HDF5 batches |
| `frame_scale` | 0.5 | 0.1…1 | size of the saved frame |
| `jpg_quality` | 90 | 50…100 | JPG quality |
| `scan_check` | True | — | camera check of the axis signs before a scan |
| `still_check` | True | — | wait until the picture stops moving before each scan picture |
| `map_flip_x`, `map_flip_y` | True on the instrument | — | mini map flipped (picture only) |
| `arrows_flip_x`, `arrows_flip_y` | True / False on the instrument | — | manual mode arrows reversed (no effect on pictures and stitching) |
| `batch_size` | 100 | integers 1…1000 | frames per HDF5 file (the file can be opened when it is full) |
| `save_jpg` | True | — | each frame is written to `<scan>/jpg/` at once (to watch during the scan) |
| `jpg_scale` | 0.5 | 0.05…1 | size of these JPG relative to the frame |
| `log_bytes` | True | — | write the exchange bytes to the log |

### 22.3. Commands

| Command | Action | G-code |
|---|---|---|
| `m.move(dx, dy, dz)` | relative move, µm | `G91 G0` |
| `m.goto(x, y, z)` | move to a point (working coordinates) | `G90 G0` |
| `m.position`, `m.real_position` | working / real coordinates | `M114` |
| `m.offset(x, y, z)`, `m.offset_clear()` | zero offset / remove | `G92`, `G92.1` |
| `m.confirm_position()` | confirm the position | — |
| `m.limits`, `m.set_limits(x, y, z)` | travel limits / set them ("from, to" pairs, µm) | — |
| `m.set_home()`, `m.go_home()` | microscope home | `G53 G0 X0 Y0` |
| `m.find_limits()` | limit switch search | `G28` |
| `m.measure_speed()` | measure the speed | — |
| `m.focus()` | autofocus | `G30` |
| `m.check_orientation()` | check the axis signs with the camera | — |
| `m.image()`, `m.photo()` | frame / picture | `M240` |
| `m.white_balance()`, `m.dark_frame()`, `m.empty_frame()` | white balance, reference frames | — |
| `m.manual()` | manual mode | — |
| `m.plan()`, `m.scan()` | scan program / execution | `G29`, `M240` |
| `m.gcode(program)` | run G-code | — |
| `m.info()`, `m` | instrument info, state | — |
| `m.close()` | disconnect | — |
| `mc.stitch()`, `mc.view()`, `mc.hints()`, `mc.export()`, `mc.distortion()`, `mc.clean()` | stitching, marking, hints, export, lens distortion, clean-up | — |
| `mc.join()`, `mc.ScanData()` | simple mosaic, access to the frames | — |
| `mc.help()` | short help | — |

Command line: `python -m microcontrol stop | status | holders | probe`.

### 22.4. Error codes

Error texts are in English, with a code. List: `mc.ERRORS`.

| Code | Meaning |
|---|---|
| E101–E108 | connection: port not found, busy, not an ATR8300, closed, unknown parameter, instrument held by another program, wrong reply, the port did not open |
| E110 | emergency stop |
| E201 | step too large |
| E202 | target outside the allowed travel |
| E203 | above the Z ceiling (risk of hitting the objective) |
| E204, E205 | limit switch; direction blocked |
| E206 | invalid number |
| E207 | unknown objective |
| E208 | invalid parameter value |
| E209 | limit switch search blocked or failed |
| E210 | suspicious move |
| E211 | position not confirmed |
| E212 | no motion registered while measuring the speed |
| E213, E214 | nothing to do; move smaller than a motor step |
| E215, E216 | home already set; home not set |
| E301–E304 | link: no reply, garbled reply, checksum, reply to another command |
| E305 | command not allowed |
| E306 | move not executed (failure or "still moving") |
| E307 | link to the controller lost (port reopened; the move is not repeated) |
| E401, E402 | laser locked; power above the limit |
| E501–E505 | camera: not found, no SDK, not connected, does not respond |
| E601–E603 | autofocus: no room to search, unknown measure, no clear maximum at any point |
| E701–E703 | scanning: area not set, low disk space, invalid parameters |
| E801 | scan folder not found |
| E901, E902 | G-code: syntax, unsupported command |

### 22.5. Log

`logs/MC_YYYYMMDD_HHMMSS.log`, one line per event in time order:
```
2026-11-20 12:00:05.011  DEBUG  HW-TX    MOTOR_X    AA 55 00 07 2E 00 00 66 9B
2026-11-20 12:00:05.013  DEBUG  HW-RX    MOTOR_X    AA 55 00 05 2E 00 33
2026-11-20 12:00:05.310  INFO   MOV-001  Moved by dX +567.0, dY +0.0, dZ +0.0 um.
2026-11-20 12:00:09.900  ERROR  E201     Step too large: dX 2000 um, the limit is 1000 um.
```

| Code | Event |
|---|---|
| SYS-001/002 | session start / end |
| CON-001…006 | connect, disconnect, instrument info, position restored, cooling, link restored |
| LCK-001 | instrument lock taken / released |
| MOV-001…013 | move, go to, zero offset, limit switches, objective, motor step, speed, confirmation, home set, go home, motors on, limit code while leaving, resend while moving |
| LGT-001, LSR-001 | light, laser |
| CAM-001…003 | picture, camera setting, reference frame |
| AF-001…004 | autofocus, focus plane, focus at a limit, focus window moved |
| MAN-001…007, ORI-001 | manual mode: start, end, area corner, step change, cross, focus point, cross label; orientation check |
| GC-001…003, SCN-001…003 | G-code; scan frame, re-shoot, stage still moving |
| STI-001/002, MSK-001/002 | stitching, stitch engine; export, clean-up |
| JOI-001, STO-001, SET-001, STP-001 | mosaic, batch written, parameter change, emergency stop |
| HW-TX / HW-RX | bytes to / from the instrument (`m.log_bytes = False` — do not write) |

### 22.6. Files

```
logs/                   session logs
photos/<date>/          pictures
references/             reference frames (16 bit)
scans/                  scans (section 11)
demo/                   demo programs (section 23)

~/.microcontrol/devices/   home, last position, orientation and lens distortion of each instrument
~/.microcontrol/control_<port>.lock   who controls the instrument
~/.microcontrol/STOP       emergency stop signal
```

---

## 23. Demo programs

Folder `demo/`. All programs run on the simulator by default (`PORT = 'sim'`); for the instrument — `PORT = 'auto'`. Run from the `demo` folder: `python demo_1_simulator.py`.

| File | Contents |
|---|---|
| `demo_1_simulator.py` | full cycle without the instrument: autofocus → area → scan → stitching → viewer |
| `demo_2_first_connect.py` | first connection to the instrument without the camera and without moving |
| `demo_3_manual_scan.py` | main scenario: area in manual mode → program → scan → stitching |
| `demo_4_gcode.py` | a G-code program; a suspicious program is refused before any motion |
| `demo_5_dataset.py` | reference frames, reading a scan, field correction |
| `demo_6_home.py` | microscope home: setting it, a new session, image check |

Minimal example (simulator):

```python
import microcontrol as mc

m = mc.Microscope('sim', objective=10)
m.focus()                                  # autofocus
w, h = m.frame_size
m.area = mc.Area(0, 0, 4 * w, 3 * h)       # area of 5 x 4 frames
folder = m.scan(name='Sample')
m.close()
mc.stitch(folder)
mc.view(folder)
```

Main scenario on the instrument:

```python
m = mc.Microscope(objective=10)
m.manual()                 # E / Q — focus; 1, 2 — area corners; F — more focus points; Esc
folder = m.scan(name='Sample_1')
m.close()
mc.stitch(folder)
mc.view(folder)            # G — two crosses in a row; S / B — select squares; N — numbers; E — export
mc.export(folder)
mc.clean(folder, confirm=True)
```

On the instrument, do not raise the stage "blindly" with commands like `m.move(dz=…)`: focus in manual mode first.

---

## 24. What is not verified on the instrument

- The correspondence of the axis directions to the image, the pixel size and the camera mirroring (section 20, steps 4–5).
- Motor speed; 3000 µm/s is assumed by default (`m.measure_speed()`).
- Turning the motors on with `24 01` before the first move — the order is taken from the stock program code.
- Limit switch codes `01`/`02` in the controller reply; the limit switch search.
- Z travel per step (assumed equal to $\delta$, as in the Optosky UI); with $\delta = 5.56$ µm, reachability of focus at 20x and 100x (section 9.3): at 100x press `E`/`Q` several times and check whether a position with a sharp image exists.
- Laser: the command format is taken from the stock program code.
- Illumination levels above 5.

---

## Contact

Questions and bug reports: [andrey.quant@yandex.ru](mailto:andrey.quant@yandex.ru)
