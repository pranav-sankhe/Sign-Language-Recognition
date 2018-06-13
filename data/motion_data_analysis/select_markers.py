from __future__ import absolute_import
from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import numpy as np 
import utils


BASE_PATH = "/home/user/Documents/SignLangRecog/data/01_CSV_full"

count = 0 
mat = []
markers = []
handcrafted = True
drop_posdata = True

for filename in os.listdir(BASE_PATH):
    if filename.endswith("csv"):
        print filename
        motionData_filepath = BASE_PATH +'/' + filename
        motionData = utils.render_motionData(motionData_filepath,2)
        markers = utils.Variance(motionData, filename,handcrafted, drop_posdata)        
        print "file count:: ", count 
        count = count + 1
        if count > 1000:                                           #control the number of iterations
            break

np.save('MARKER_INDICES', markers)