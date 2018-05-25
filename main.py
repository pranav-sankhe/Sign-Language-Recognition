from __future__ import absolute_import
from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import sys
import numpy as np 
sys.path.append(os.path.abspath("/home/user/Documents/SignLangRecog/data"))
import utils



BASE_PATH = "/home/user/Documents/SignLangRecog/data/01_CSV_full"

count = 0 
mat = []
for filename in os.listdir(BASE_PATH):
    if filename.endswith("csv"):
        print filename
        motionData_filepath = BASE_PATH +'/' + filename
        motionData = utils.render_motionData(motionData_filepath)
        # utils.plot3D(motionData, interactiveMode=True)
        selected_markers_var = utils.Variance(motionData, filename)
        print selected_markers_var, selected_markers_var.shape
        mat = utils.VARsum(selected_markers_var)
        
        print "file count:: ", count 
        count = count + 1
        if count > 5:                                           #control the number of iterations
            break

mat = np.transpose(mat)
for i in range(2,3):
    plt.plot(mat[i])
plt.show()

# meta_filepath = "/home/user/Documents/SignLangRecog/data/position_full.xlsx"
# motionData_filepath =  "/home/user/Documents/SignLangRecog/data/RG0_Corpus_201801_P203_02_t01.csv"

# motionData = utils.render_motionData(meta_filepath, motionData_filepath)
# # utils.plot3D(motionData, interactiveMode=True)
# utils.meanVariance(motionData)


# if __name__ == '__main__':
#   tf.app.run()