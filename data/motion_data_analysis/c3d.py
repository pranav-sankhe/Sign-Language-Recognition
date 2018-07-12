import btk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

reader = btk.btkAcquisitionFileReader()  # build a btk reader object 
reader.SetFilename("RG0_Corpus_201707_01A_01_t01.c3d") # set a filename to the reader
acq = reader.GetOutput()                 # btk aquisition object
acq.Update()                             # Update ProcessObject associated with DataObject


print('Acquisition duration: %.2f s' %acq.GetDuration()) 
print('Point frequency: %.2f Hz' %acq.GetPointFrequency())
print('Number of frames: %d' %acq.GetPointFrameNumber())
print('Point unit: %s' %acq.GetPointUnit())
print('Analog frequency: %.2f Hz' %acq.GetAnalogFrequency())
print('Number of analog channels: %d' %acq.GetAnalogNumber()) 
print('Number of events: %d' %acq.GetEventNumber())


print('Marker labels:')
for i in range(0, acq.GetPoints().GetItemNumber()):
    print(acq.GetPoint(i).GetLabel())   
print('\n\nAnalog channels:')
for i in range(0, acq.GetAnalogs().GetItemNumber()):
    print(acq.GetAnalog(i).GetLabel() )


for i in range(0, acq.GetEvents().GetItemNumber()):
    print(acq.GetEvent(i).GetLabel() + ' at frame %d' %acq.GetEvent(i).GetFrame())   



for i in range(acq.GetMetaData().GetChildNumber()):
    print(acq.GetMetaData().GetChild(i).GetLabel() + ':')
    for j in range(acq.GetMetaData().GetChild(i).GetChildNumber()):
        print(acq.GetMetaData().GetChild(i).GetChild(j).GetLabel())
    print('\n')


l = []
data = np.empty((3, acq.GetPointFrameNumber(), 1))
for i in range(0, acq.GetPoints().GetItemNumber()):
    label = acq.GetPoint(i).GetLabel()
    data = np.dstack((data, acq.GetPoint(label).GetValues().T))
    l.append(label)

data = data.T
data = np.delete(data, 0, axis=0)  # first marker is noisy for this file
data[data==0] = np.NaN             # handle missing data (zeros)
shape = data.shape
data = data + np.random.rand(shape[0], shape[1], shape[2])*0
print(np.nanmin(data), np.nanmax(data))



Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)


dat = data[:, 130:340, :]
freq = acq.GetPointFrequency()

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.view_init(10, 150)
pts = []
for i in range(dat.shape[0]):
    pts += ax.plot([], [], [], 'o', markersize=3)

ax.set_xlim3d([np.nanmin(dat[:, :, 0]), np.nanmax(dat[:, :, 0])])
ax.set_ylim3d([np.nanmin(dat[:, :, 1])-400, np.nanmax(dat[:, :, 1])+400])
ax.set_zlim3d([np.nanmin(dat[:, :, 2]), np.nanmax(dat[:, :, 2])])
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')

# animation function
def animate(i):
    for pt, xi in zip(pts, dat):
        x, y, z = xi[:i].T
        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])   
    return pts

# Animation object
anim = animation.FuncAnimation(fig, func=animate, frames=dat.shape[1], interval=1000/freq, blit=True)
anim.save('lines.mp4', writer=writer)
plt.show()