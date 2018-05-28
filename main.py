from __future__ import absolute_import
from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import sys
import numpy as np 
sys.path.append(os.path.abspath("/home/user/Documents/SignLangRecog/data"))
import utils


filepath = '/home/user/Documents/SignLangRecog/data/nturgbd_skeletons/nturgb+d_skeletons/S001C001P001R001A001.skeleton'

utils.readSkeletonFiles(filepath)