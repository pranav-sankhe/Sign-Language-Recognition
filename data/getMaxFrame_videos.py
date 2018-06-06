from __future__ import absolute_import
from __future__ import division
import os 
import sys
import numpy as np 
sys.path.append(os.path.abspath("/home/user/Documents/SignLangRecog/data"))
import utils
import cv2

l = []

filepath = '/home/data/02_CSV_REAL/01_CSV_full/memo.txt'
f = open(filepath, "r")                     # Open the file
for line in f:
    l.append(int(line[len(line) - 5:len(line)]))
print l
print len(l)
print max(l)    