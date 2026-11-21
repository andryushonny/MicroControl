"""Demo 5. Frames for a dataset: reference frames, reading a scan, field correction.

Correction: I' = (I - D) / (E - D) * <E - D>,  D - dark frame, E - frame of an empty area.
"""
import sys; sys.path.insert(0, '..')
import numpy as np
import microcontrol as mc

PORT = 'sim'

m = mc.Microscope(PORT, objective=10, folder='demo_out')
m.focus()                         # on the instrument: focus in m.manual() first
m.white_balance()                 # on an empty area
# reference frames go to references/ as 16-bit PNG (mean x 256)
E = mc.imread(m.empty_frame()['File']).astype(np.float32) / 256   # flat field: illumination non-uniformity
D = mc.imread(m.dark_frame()['File']).astype(np.float32) / 256    # light off: dark signal of the sensor
w, h = m.frame_size
m.area = mc.Area(0, 0, 2 * w, 2 * h)
m.frame_format, m.frame_scale = 'h5', 1     # lossless frames for a dataset
folder = m.scan(name='Dataset')
m.close()

st = mc.ScanData(folder)
good = [i for i, r in zip(range(len(st)), st.index) if not int(r['blurry'])]   # without blurred frames
gain = (E - D).mean() / np.maximum(E - D, 1.0)
for i, img in st.frames(good):
    I = (img.astype(np.float32) - D) * gain           # corrected frame
    print(i, img.shape, f"{I.mean():.1f}")
st.export()                       # PNG files + TileConfiguration.txt for Fiji
