from __future__ import absolute_import
from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import sys
import numpy as np 
sys.path.append(os.path.abspath("/home/user/Documents/SignLangRecog/data"))
import utils


BASE_PATH = '/home/user/Documents/SignLangRecog/data/nturgbd_skeletons/nturgb+d_skeleton'
count = 0 
l = []
for filename in os.listdir(BASE_PATH):
    if filename.endswith("skeleton"):
        count = count + 1 
        print "Running through the skeleton file: ", filename, " File Number: ", count
        bad_files = np.load('./data/badfiles.npy')
        if filename.split('.')[0] in bad_files:
            print "bad file detected. Continuing."
            continue
        
        filepath = BASE_PATH +'/' + filename
        utils.readSkeletonFiles(filepath)

        if count >= 1:
        	break