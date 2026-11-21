Short MicroControl examples. All of them run on the simulator by default (PORT = 'sim');
for the instrument: PORT = 'auto'. Run from the demo folder:  python demo_1_simulator.py

demo_1_simulator.py      full cycle without the instrument: focus -> area -> scan -> stitching -> viewer
demo_2_first_connect.py  first connection to the instrument: no camera, no motion
demo_3_manual_scan.py    main scenario: area in m.manual() -> plan -> scan -> stitching -> marking -> export
demo_4_gcode.py          a G-code program; a suspicious program is refused before any motion
demo_5_dataset.py        reference frames, reading a scan, field correction for a dataset
demo_6_home.py           microscope home: set_home / new session / go_home

On the instrument Z is never raised blindly: focus in m.manual() first (keys E / Q, Shift+F).

Русский: короткие примеры; по умолчанию работают на симуляторе, для прибора PORT = 'auto'.
Подробное описание — docs/reference_ru.md, раздел 23. 中文说明：docs/reference_zh.md，第 23 节。 English: docs/reference_en.md, section 23.
