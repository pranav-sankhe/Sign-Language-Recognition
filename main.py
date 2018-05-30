from __future__ import absolute_import
from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import sys
import numpy as np 
sys.path.append(os.path.abspath("/home/user/Documents/SignLangRecog/models"))
import rnn


BATCH_SIZE = 100 
MAX_FRAME_LEN = 429 
NUM_FEATURES = 45

INPUT_SIZE = (BATCH_SIZE, MAX_FRAME_LEN, NUM_FEATURES)           # input size =  (batch size, number of time steps, hidden size) 
OUTPUT_SIZE = 60                    
FULLY_CONNECTED_SIZE = 512
HIDDEN_LAYER_1 = 256 
HIDDEN_LAYER_2 = 256

NUM_EPOCHS = 462


l= rnn.get_skeleton_batch_data()
print l[0].shape, l[1].shape
