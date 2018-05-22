import argparse
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def plot3D(data, interactiveMode=True):
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	xdata = 

	ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');


