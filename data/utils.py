import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation


NUM_MARKERS = 107         #Total Number of Markers
MOTION_PARAMETERS = 6


def num_elements(shape):
	num = 1
	for i in list(shape):
		num *= i
	return num

def range_len(rang):
	return rang[1]-rang[0]+1

def convert_to_one_hot(data, rang):

	data_shape = data.shape
	data_oned = data.reshape(num_elements(data_shape))

	enc_data_shape = ( num_elements(data_shape), range_len(rang) )
	enc_data = np.zeros(enc_data_shape)
	enc_data[np.arange(num_elements(data_shape)), data_oned] = 1

	enc_data_shape = list(data_shape)
	enc_data_shape.append(range_len(rang))
	enc_data = enc_data.reshape(enc_data_shape)

	return enc_data

def meanVariance(mdata):
	xdata = mdata[:,0]
	ydata = mdata[:,1]
	zdata = mdata[:,2]
	zrot = mdata[:,3]
	yrot = mdata[:,4]
	xrot = mdata[:,5]


	var_df = pd.DataFrame(columns=['xpos', 'ypos', 'zpos', 'zrot', 'yrot', 'xrot'])

	var_df['xpos'] = np.var(xdata, axis = 1)
	var_df['ypos'] = np.var(ydata, axis = 1)
	var_df['zpos'] = np.var(zdata, axis = 1)
	var_df['zrot'] = np.var(zrot, axis = 1)
	var_df['yrot'] = np.var(yrot, axis = 1)
	var_df['xrot'] = np.var(xrot, axis = 1)

	var_df['xpos'] = var_df['xpos'].apply(lambda x: 0 if x < 1e-5 else x)
	var_df['ypos'] = var_df['ypos'].apply(lambda x: 0 if x < 1e-5 else x)
	var_df['zpos'] = var_df['zpos'].apply(lambda x: 0 if x < 1e-5 else x)
	var_df['zrot'] = var_df['zrot'].apply(lambda x: 0 if x < 1e-5 else x)
	var_df['yrot'] = var_df['yrot'].apply(lambda x: 0 if x < 1e-5 else x)
	var_df['xrot'] = var_df['xrot'].apply(lambda x: 0 if x < 1e-5 else x)


	pd.DataFrame.to_csv(var_df, 'var.csv')

def plot3D(mdata, interactiveMode=True):
	ax = plt.axes(projection='3d')
	plt.ion()
	plt.show()
	
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	xdata = mdata[:,0]
	ydata = mdata[:,1]
	zdata = mdata[:,2]
	zrot = mdata[:,3]
	yrot = mdata[:,4]
	xrot = mdata[:,5]
	print mdata.shape[2]

	for timestamp in range(int(mdata.shape[2])):	
		ax.scatter3D(xrot[:,timestamp], yrot[:,timestamp], zrot[:,timestamp], c=zrot[:,timestamp], cmap='Greens');
		print timestamp
		plt.pause(0.001)	

		


def render_motionData(meta_filepath, filepath):

	motionData = pd.read_csv(filepath)
	cLength = motionData.shape[1]
	cList = np.arange(cLength)
	motionData = pd.read_csv(filepath, names = cList)

	motionData = motionData.values
	motionData = np.reshape(motionData, (NUM_MARKERS, MOTION_PARAMETERS, cLength))
	return motionData


