from __future__ import absolute_import
from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import sys
import numpy as np 
sys.path.append(os.path.abspath("/home/user/Documents/SignLangRecog/data"))
import utils


BASE_PATH = '/home/user/Documents/SignLangRecog/data/nturgbd_skeletons/nturgb+d_skeletons/'
count = 0 
l = []
for filename in os.listdir(BASE_PATH):
    if filename.endswith("skeleton"):
        count = count + 1 
        print "Running through the skeleton file: ", filename, " File Number: ", count
        filepath = BASE_PATH +'/' + filename
        l = utils.get_max_frame_count(filepath,l)

L = max(l)



BASE_PATH = "/home/user/Documents/SignLangRecog/data/01_CSV_full"

count = 0 
l = []

for filename in os.listdir(BASE_PATH):
    if filename.endswith("csv"):
        motionData_filepath = BASE_PATH +'/' + filename
        print "Running through the csv file: ", filename , " File Number: ", count
        l = utils.get_max_time_steps(motionData_filepath,l)
        count = count + 1 
        

print "max frame length for skeleton files = ", L 
print "max frame length for CSV files = ", max(l)