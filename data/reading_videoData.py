from __future__ import absolute_import
from __future__ import division
import os
import cv2
import numpy as np
from numpy import *
from PIL import Image
import pyflow


BATCH_SIZE = 32
NUM_EPOCHS = 100
IMG_HEIGHT = 256
IMG_WIDTH = 256 
IN_CHANNELS = 1
NUM_FRAMES = 858
BASE_PATH = '/home/data/All_video_data/HRIJ_0130_movie'

l_numframe = []

filepath = '/home/data/02_CSV_REAL/01_CSV_full/memo.txt'
f = open(filepath, "r")                     # Open the file
for line in f:
    l_numframe.append(int(line[len(line) - 5:len(line)]))

filenames = os.listdir(BASE_PATH)
filenames = filenames[0:BATCH_SIZE*NUM_EPOCHS]

def save_as_jpg():
    if not os.path.exists('video_frames'):
        os.makedirs('video_frames')
    if not os.path.exists('opflow_frames'):
        os.makedirs('opflow_frames')


    for i in range(len(filenames)):
        if filenames[i].split('.')[1] != 'mp4':
            continue
        if not os.path.exists('video_frames' + '/' + filenames[i].split('.')[0]):
            os.makedirs('video_frames' + '/' + filenames[i].split('.')[0])

        if not os.path.exists('opflow_frames' + '/' + filenames[i].split('.')[0]):
            os.makedirs('opflow_frames' + '/' + filenames[i].split('.')[0])
            
        cap = cv2.VideoCapture(BASE_PATH + '/' + filenames[i])
        print "Reading file", filenames[i], i 
        ret, frame1 = cap.read()
        frame1_save = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame1_save = cv2.resize(frame1_save, (IMG_HEIGHT, IMG_WIDTH))
        count = 0 

        num = '' 
        diff = 3 - len(str(count))
        if diff == 0:
            pass
        else:
            for j in range(diff):
                num = num + '0'
        
        num = num + str(count)   
        if int(count)%2 == 0:
            #print 'video_frames' + '/' + filenames[i].split('.')[0] + '/' +  num + '.jpg', i
            cv2.imwrite('video_frames' + '/' + filenames[i].split('.')[0] + '/' +  num + '.jpg', frame1_save)
            cv2.imwrite('opflow_frames' + '/' + filenames[i].split('.')[0] + '/' + num + '.jpg', np.random.rand(256,256,3))
        
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        count = count + 1


        while(cap.isOpened() == True):
            #print "At frame", count
            ret, frame2 = cap.read()
            if ret == False: 
                break
            
            frame2_save = cv2.resize(frame2, (IMG_HEIGHT, IMG_WIDTH))
            frame2_save = cv2.cvtColor(frame2_save, cv2.COLOR_BGR2GRAY)

            num = '' 
            diff = 3 - len(str(count))
            if diff == 0:
                pass
            else:
                for j in range(diff):
                    num = num + '0'
            
            num = num + str(count)   
            if int(count)%2 == 0:
                cv2.imwrite('video_frames' + '/' + filenames[i].split('.')[0] + '/'  + num + '.jpg', frame2_save)

            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            prvs = cv2.resize(prvs, (IMG_HEIGHT, IMG_WIDTH))
            next = cv2.resize(next, (IMG_HEIGHT, IMG_WIDTH))
                
            flow = cv2.calcOpticalFlowFarneback(prvs,next,0.5, 1, 3, 15, 3, 5, 0)


            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv = cv2.resize(hsv, (IMG_HEIGHT, IMG_WIDTH))

            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                       
            num = '' 
            diff = 3 - len(str(count))
            if diff == 0:
                pass
            else:
                for j in range(diff):
                    num = num + '0'
            
            num = num + str(count)
            if int(count)%2 == 0:
                #print "opflow",'opflow_frames' + '/' + filenames[i].split('.')[0] + '/' + num + '.jpg', i 
                cv2.imwrite('opflow_frames' + '/' + filenames[i].split('.')[0] + '/' + num + '.jpg', rgb)

            prvs = next
            count = count + 1
            filename = filenames[i].split('.')[0]
       
        
        cap.release()



def opflow_xy():
    if not os.path.exists('video_frames'):
        os.makedirs('video_frames')
    if not os.path.exists('opflow_xy'):
        os.makedirs('opflow_xy')


     # check is the file is a video file   
    for i in range(len(filenames)):             
        if filenames[i].split('.')[1] != 'mp4':
            continue

        if not os.path.exists('opflow_xy' + '/' + filenames[i].split('.')[0]):
            os.makedirs('opflow_xy' + '/' + filenames[i].split('.')[0])
        count = 0    
        cap = cv2.VideoCapture(BASE_PATH + '/' + filenames[i])
        print "Reading file", filenames[i], i 

        ret, frame1 = cap.read()
        if ret == False: 
            break
        dim = (256,256)
        frame1 = cv2.resize(frame1, dim, interpolation = cv2.INTER_CUBIC)

        
        count = count + 1
        while(cap.isOpened() == True):
            ret, frame = cap.read()
            if ret == False: 
                break
            dim = (256,256)
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_CUBIC)
            im1 = frame1
            im2 = frame
            im1 = im1.astype(float) / 255.
            im2 = im2.astype(float) / 255.

            alpha = 0.012
            ratio = 0.75
            minWidth = 20
            nOuterFPIterations = 10
            nInnerFPIterations = 1
            nSORIterations = 30
            colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))


            u, v, im2W = pyflow.coarse2fine_flow(
                im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                nSORIterations, colType)
            # u = u*255.0
            # v = v*255.0
            u = u*255
            u = np.ceil(u)
            u = np.int16(u)

            v = v*255
            v = np.ceil(v)
            v = np.int16(v)


            num = '' 
            diff = 3 - len(str(count))
            if diff == 0:
                pass
            else:
                for j in range(diff):
                    num = num + '0'
            
            num = num + str(count)


            cv2.imwrite('opflow_xy' + '/' + filenames[i].split('.')[0] + '/' + '_x_' + num + '.jpg', u)
            cv2.imwrite('opflow_xy' + '/' + filenames[i].split('.')[0] + '/' + '_y_' + num + '.jpg', v)
            count = count + 1
            frame1 = frame
        
        cap.release()


opflow_xy()
cv2.destroyAllWindows()


