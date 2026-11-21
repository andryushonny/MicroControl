"""Demo 2. First connection to the instrument without the camera and without moving the sample.

Before running: the Optosky software is closed. Link check from the command line:
    python -m microcontrol probe
"""
import sys; sys.path.insert(0, '..')
import microcontrol as mc

PORT = 'auto'                     # 'sim' - try on the simulator
if len(sys.argv) > 1: PORT = sys.argv[1]

m = mc.Microscope(PORT, objective=10, camera=False)   # the objective is always given: the Z limits depend on it
print(m.info())                   # model, serial number, type (Mapping expected)
print(m)                          # position, light, limits
m.light = 0                       # light: integer levels 0..5
m.light = 5
m.close()                         # laser 0, motors off, light 0, cooling 20 C - as in the Optosky software
