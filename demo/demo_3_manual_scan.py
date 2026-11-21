"""Demo 3. Main scenario: choose the area by hand, shoot, stitch, select squares, export.

In the m.manual() window: arrows - one frame, E/Q - Z up/down, Shift+F - autofocus,
1 and 2 - area corners, F - one more focus point, H - help, Esc - exit.
"""
import sys; sys.path.insert(0, '..')
import microcontrol as mc

PORT = 'sim'                      # on the instrument: 'auto'

m = mc.Microscope(PORT, objective=10)
m.manual()                        # go to one corner of the area (key 1), to the other (key 2), Esc
print(m.area)                     # the area, um
print(m.plan())                   # scan program (G-code) - can be read before shooting
folder = m.scan(name='Sample_1')  # 20 % overlap by default
m.close()
mc.stitch(folder)                 # frame positions from the overlaps
mc.view(folder)                   # G - two crosses in a row (the picture turns level), S / B - select, N - numbers, Y - hints
mc.export(folder)                 # only the selected squares: PNG + DXF
