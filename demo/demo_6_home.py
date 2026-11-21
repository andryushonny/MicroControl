"""Demo 6. Microscope home: a permanent origin of the real coordinates.

Set once; on the next connections the coordinates continue from the same home.
"""
import sys; sys.path.insert(0, '..')
import microcontrol as mc

PORT = 'sim'

m = mc.Microscope(PORT, objective=10, folder='demo_out')
m.focus()
try:
    print(m.set_home())           # (0, 0, 0) here; a photo of the place is saved
except mc.MicroControlError as e:
    print(e)                      # E215: home already set - again only with m.set_home(replace=True)
m.move(dx=500, dy=300)
m.close()

m = mc.Microscope(PORT, objective=10, folder='demo_out')   # a new session
print(m.position)                 # restored, but not confirmed
m.confirm_position()              # nobody touched the stage
print(m.go_home())                # X, Y = 0 and comparison with the home photo
m.close()
