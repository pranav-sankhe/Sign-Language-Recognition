import utils
import numpy as np
import os 

l = []

l =  utils.bad_file_txt(l)

BASE_PATH = './nturgbd_skeletons/nturgb+d_skeletons/'
count = 0 
for filename in os.listdir(BASE_PATH): 
    if filename.endswith("skeleton"):
        count = count + 1 
        print "Running through the skeleton file: ", filename, " File Number: ", count
        filepath = BASE_PATH +'/' + filename
        l = utils.twoPpl_list(l, filepath)


l = np.array(l)
l = np.unique(l)
print count
print len(l)

np.save('badfiles.npy', l)