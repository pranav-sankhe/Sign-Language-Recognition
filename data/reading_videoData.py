import cv2
import numpy as np
from numpy import *

BATCH_SIZE = 32
NUM_EPOCHS = 100
IMG_HEIGHT = 220
IMG_WIDTH = 220 
IN_CHANNELS = 1
NUM_FRAMES = 858
BASE_PATH = ''

filenames = os.listdir(BASE_PATH):
filenames = filenames[0:BATCH_SIZE*NUM_EPOCHS]
def get_batch(step):

    filename_batch = filenames[step: (step+1)*BATCH_SIZE]
    data = np.zeros((BATCH_SIZE, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS))
    cap = cv2.VideoCapture("test.mp4")

    ret, frame1 = cap.read()
    #frame1 = cv2.resize(frame1, (220, 220))
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    count = 0 

    while(cap.isOpened() == True):
        ret, frame2 = cap.read()
        if ret == False: 
            break
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        prvs = cv2.resize(prvs, (220, 220))
        next = cv2.resize(next, (220, 220))
        
            
        flow = cv2.calcOpticalFlowFarneback(prvs,next,0.5, 1, 3, 15, 3, 5, 0)
        #flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 10, 3, 5, 1.2, 0)


        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = cv2.resize(hsv, (220, 220))

        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    #    cv2.imshow('frame2',rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = next
        count = count + 1
    print count
    cap.release()


# count = 0 
# while(1):

#     ret, frame2 = cap.read()
#     if ret == False: 
#         break

#     count = count + 1

# print count     


cap.release()
cv2.destroyAllWindows()