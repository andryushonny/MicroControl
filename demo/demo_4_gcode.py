"""Demo 4. Control by a G-code program (units - um).

The whole program is checked before any motion: on an error the stage does not move.
"""
import sys; sys.path.insert(0, '..')
import microcontrol as mc

PORT = 'sim'

program = """
G90                 ; absolute coordinates
M355 S1 P5          ; light, level 5
G92 X0 Y0 Z0        ; working zero here (real coordinates do not change)
G0 Z2000            ; a 2 mm rise in one command - more than allowed at 10x (1000 um)
"""
# A rise larger than max_move_z in one command is refused as suspicious (E210):
try:
    m = mc.Microscope(PORT, objective=10)
    m.gcode(program)
except mc.MicroControlError as e:
    print("Refused before any motion:", e)

program = """
G30                 ; autofocus around the current Z
G90
G0 X500 Y0          ; move, um
M240 R0 C0          ; picture (row 0, column 0)
G0 X1000
M240 R0 C1
M114                ; print the position
"""
print(m.gcode(program, save='gcode_demo'))   # M240 pictures -> a new scan folder
m.close()
