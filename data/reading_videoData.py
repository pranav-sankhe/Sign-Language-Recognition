import os
import cv2
import numpy as np
from numpy import *

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
def get_batch(step):

    filename_batch = filenames[step: (step+1)*BATCH_SIZE]
    vid_data = np.zeros((BATCH_SIZE, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS))
    opFlow_data = np.zeros((BATCH_SIZE, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS))    
    

    for i in range(BATCH_SIZE):
        cap = cv2.VideoCapture(BASE_PATH + '/' + filename_batch[i])
        print "Reading file", filename_batch[i]
        ret, frame1 = cap.read()
        frame1_save = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame1_save = cv2.resize(frame1_save, (IMG_HEIGHT, IMG_WIDTH))
        count = 0 
        vid_data[i,count,:,:,0] = frame1_save
        opFlow_data[i,count,:,:,0] = frame1_save

        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        count = count + 1


        while(cap.isOpened() == True):
            print "At frame", count
            ret, frame2 = cap.read()
            frame2_save = cv2.resize(frame1_save, (IMG_HEIGHT, IMG_WIDTH))
            vid_data[i,count,:,:,0] = frame2_save

            if ret == False: 
                break
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            prvs = cv2.resize(prvs, (IMG_HEIGHT, IMG_WIDTH))
            next = cv2.resize(next, (IMG_HEIGHT, IMG_WIDTH))
            
                
            flow = cv2.calcOpticalFlowFarneback(prvs,next,0.5, 1, 3, 15, 3, 5, 0)
            #flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 10, 3, 5, 1.2, 0)


            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv = cv2.resize(hsv, (IMG_HEIGHT, IMG_WIDTH))

            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)

            opFlow_data[i,count,:,:,0] = rgb
            # cv2.imshow('frame2',rgb)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break
            # elif k == ord('s'):
            #     cv2.imwrite('opticalfb.png',frame2)
            #     cv2.imwrite('opticalhsv.png',rgb)
            prvs = next
            count = count + 1
        print vid_data[i,:,:,:,:].shape
        print opFlow_data[i,:,:,:,:].shape    
        cap.release()
    return vid_data, opFlow_data



def save_as_numpy():
    if not os.path.exists('npVideos'):
        os.makedirs('npVideos')
    if not os.path.exists('npopFlow'):
        os.makedirs('npopFlow')        

    for i in range(len(filenames)):

        vid_data = np.zeros((NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS))
        opFlow_data = np.zeros((NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS))    

        cap = cv2.VideoCapture(BASE_PATH + '/' + filenames[i])
        print "Reading file", filenames[i]
        ret, frame1 = cap.read()
        frame1_save = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame1_save = cv2.resize(frame1_save, (IMG_HEIGHT, IMG_WIDTH))
        count = 0 
        vid_data[count,:,:,0] = frame1_save
        opFlow_data[count,:,:,0] = frame1_save

        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        count = count + 1


        while(cap.isOpened() == True):
            print "At frame", count
            ret, frame2 = cap.read()
            frame2_save = cv2.resize(frame1_save, (IMG_HEIGHT, IMG_WIDTH))
            vid_data[count,:,:,0] = frame2_save

            if ret == False: 
                break
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            prvs = cv2.resize(prvs, (IMG_HEIGHT, IMG_WIDTH))
            next = cv2.resize(next, (IMG_HEIGHT, IMG_WIDTH))
                
            flow = cv2.calcOpticalFlowFarneback(prvs,next,0.5, 1, 3, 15, 3, 5, 0)
            #flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 10, 3, 5, 1.2, 0)


            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv = cv2.resize(hsv, (IMG_HEIGHT, IMG_WIDTH))

            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)

            opFlow_data[count,:,:,0] = rgb
            # cv2.imshow('frame2',rgb)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break
            # elif k == ord('s'):
            #     cv2.imwrite('opticalfb.png',frame2)
            #     cv2.imwrite('opticalhsv.png',rgb)
            prvs = next
            count = count + 1
            filename = filenames[i].split('.')[0]
        
        vid_data = vid_data[::2, :, :, :]
        opFlow_data = opFlow_data[::2, :, :, :]
        np.save('npVideos' + '/' + filenames[i], vid_data)
        np.save('npopFlow' + '/' + filenames[i], opFlow_data)
        
        cap.release()
    

save_as_numpy()



cv2.destroyAllWindows()

