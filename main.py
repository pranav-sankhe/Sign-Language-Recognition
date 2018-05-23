from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os 
import sys
sys.path.append(os.path.abspath("/home/user/Documents/SignLangRecog/data"))
import utils

meta_filepath = "/home/user/Documents/SignLangRecog/data/position_full.xlsx"
motionData_filepath =  "/home/user/Documents/SignLangRecog/data/RG0_Corpus_201707_01A_02_t01.csv"

motionData = utils.render_motionData(meta_filepath, motionData_filepath)
# utils.plot3D(motionData, interactiveMode=True)
utils.meanVariance(motionData)


# if __name__ == '__main__':
# 	tf.app.run()