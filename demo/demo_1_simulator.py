"""Demo 1. Full cycle on the simulator (no instrument needed).

Connect -> autofocus -> area -> scan -> stitching -> viewer.
Run:  python demo_1_simulator.py      (from the MicroControl/demo folder)
"""
import sys; sys.path.insert(0, '..')
import microcontrol as mc

m = mc.Microscope('sim', objective=10, folder='demo_out')   # virtual instrument: same commands and protections
print(m.focus())                  # autofocus: Z, sharpness, peak contrast

w, h = m.frame_size               # field of view, um
m.area = mc.Area(0, 0, 4 * w, 3 * h)          # area of 5 x 4 frames (corners = centres of the corner frames)
folder = m.scan(name='Demo', notes='simulator, sample with crosses')
m.close()

print(mc.stitch(folder))          # exact frame positions from the overlaps
mc.view(folder)                   # viewer: G - two crosses in a row, S - select squares, 0 - whole scan, Esc - exit
