from __future__ import absolute_import
from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import sys
import numpy as np 
# sys.path.append(os.path.abspath("/home/user/Documents/SignLangRecog/models"))
# import rnn
sys.path.append(os.path.abspath("./data"))
import utils


BATCH_SIZE = 32 
MAX_FRAME_LEN = 429 
NUM_FEATURES = 45

INPUT_SIZE = (BATCH_SIZE, MAX_FRAME_LEN, NUM_FEATURES)           # input size =  (batch size, number of time steps, hidden size) 
OUTPUT_SIZE = 60                    
FULLY_CONNECTED_SIZE = 512
HIDDEN_LAYER_1 = 256 
HIDDEN_LAYER_2 = 256

NUM_EPOCHS = 1384


# l= rnn.get_skeleton_batch_data()
# print l[0].shape, l[1].shape
#def get_skeleton_batch_data():
BASE_PATH = './data/nturgbd_skeletons/nturgb+d_skeletons'

file_list = os.listdir(BASE_PATH)
file_list = np.array(file_list)
bad_files = np.load('./data/badfiles.npy')
for i in range(len(file_list)):
    file_list[i] = file_list[i].split('.')[0]
file_list = np.setdiff1d(file_list, bad_files)
t = BATCH_SIZE*NUM_EPOCHS
data = np.zeros((t, MAX_FRAME_LEN, NUM_FEATURES))
labels = []
count = 0 
for i in range(len(file_list)):
    if i == NUM_EPOCHS*BATCH_SIZE:
        break                
    print "Reading file ", file_list[i] , " filename ", count
    count = count + 1  
    filepath = BASE_PATH +'/' + file_list[i] + '.skeleton' 
    train_data = utils.readSkeletonFiles(filepath)
    motion_data = train_data[0]
    label = train_data[1]
    motion_data = np.pad(motion_data, ((0,MAX_FRAME_LEN - motion_data.shape[0]), (0,0)), mode='constant', constant_values=0)    

    data[i, :, : ] = motion_data
    labels.append(label)
    # for j in range(MAX_FRAME_LEN):
    #     for l in range(NUM_FEATURES):
    #         labels[i,j,l] = label       
    #print data
    #print labels
labels = np.array(labels)
np.save('data', data)
np.save('labels', labels)

print len(np.unique(labels))  