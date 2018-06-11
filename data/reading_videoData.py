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
BASE_PATH = '/home/data/All_video_data/HRJI_0721_movie'

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
    if not os.path.exists('video_frames'):
        os.makedirs('video_frames')
    if not os.path.exists('opflow_frames'):
        os.makedirs('opflow_frames')


    for i in range(len(filenames)):
        if not os.path.exists('video_frames' + '/' + filenames[i].split('.')[0]):
            os.makedirs('video_frames' + '/' + filenames[i].split('.')[0])

        if not os.path.exists('opflow_frames' + '/' + filenames[i].split('.')[0]):
            os.makedirs('opflow_frames' + '/' + filenames[i].split('.')[0])

#        video=cv2.VideoWriter('video.avi',-1,1,(IMG_WIDTH,IMG_HEIGHT))    
            
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
        
        #vid_data[count,:,:,0] = frame1_save
        #opFlow_data[count,:,:,0] = frame1_save

        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        count = count + 1


        while(cap.isOpened() == True):
            #print "At frame", count
            ret, frame2 = cap.read()
            if ret == False: 
                print "error"
                break
            
            frame2_save = cv2.resize(frame2, (IMG_HEIGHT, IMG_WIDTH))
            frame2_save = cv2.cvtColor(frame2_save, cv2.COLOR_BGR2GRAY)
            # vid_data[count,:,:,0] = frame2_save

            num = '' 
            diff = 3 - len(str(count))
            if diff == 0:
                pass
            else:
                for j in range(diff):
                    num = num + '0'
            
            num = num + str(count)   
            if int(count)%2 == 0:
                #print "frame 2", 'video_frames' + '/' + filenames[i].split('.')[0] + '/' +  num + '.jpg', i
                cv2.imwrite('video_frames' + '/' + filenames[i].split('.')[0] + '/'  + num + '.jpg', frame2_save)

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
            
            #rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
            #video.write(rgb)
           
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

            #opFlow_data[count,:,:,0] = rgb
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
       
        #vid_data = vid_data[::2, :, :, :]
        #opFlow_data = opFlow_data[::2, :, :, :]
        #np.save('npVideos' + '/' + filenames[i], vid_data)
        #np.save('npopFlow' + '/' + filenames[i], opFlow_data)
        
        cap.release()
    

save_as_numpy()



cv2.destroyAllWindows()


# #!/usr/bin/env python

# #The MIT License (MIT)
# #Copyright (c) 2016 Massimiliano Patacchiola
# #
# #THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
# #MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
# #CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
# #SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# #In this example three motion detectors are compared:
# #frame differencing, MOG, MOG2.
# #Given a video as input "cars.avi" it returns four
# #different videos: original, differencing, MOG, MOG2

# import numpy as np
# import cv2
# from deepgaze.motion_detection import DiffMotionDetector
# from deepgaze.motion_detection import MogMotionDetector
# from deepgaze.motion_detection import Mog2MotionDetector

# #Open the video file and loading the background image
# video_capture = cv2.VideoCapture("./cars.avi")
# background_image = cv2.imread("./background.png")

# #Decalring the diff motion detector object and setting the background
# my_diff_detector = DiffMotionDetector()
# my_diff_detector.setBackground(background_image)
# #Declaring the MOG motion detector
# my_mog_detector = MogMotionDetector()
# my_mog_detector.returnMask(background_image)
# #Declaring the MOG 2 motion detector
# my_mog2_detector = Mog2MotionDetector()
# my_mog2_detector.returnGreyscaleMask(background_image)

# # Define the codec and create VideoWriter objects
# fourcc = cv2.cv.CV_FOURCC(*'XVID')
# out = cv2.VideoWriter("./cars_original.avi", fourcc, 20.0, (1920,1080))
# out_diff = cv2.VideoWriter("./cars_diff.avi", fourcc, 20.0, (1920,1080))
# out_mog = cv2.VideoWriter("./cars_mog.avi", fourcc, 20.0, (1920,1080))
# out_mog2 = cv2.VideoWriter("./cars_mog2.avi", fourcc, 20.0, (1920,1080))


# while(True):

#     # Capture frame-by-frame
#     ret, frame = video_capture.read()

#     #Get the mask from the detector objects
#     diff_mask = my_diff_detector.returnMask(frame)
#     mog_mask = my_mog_detector.returnMask(frame)
#     mog2_mask = my_mog2_detector.returnGreyscaleMask(frame)

#     #Merge the b/w frame in order to have depth=3
#     diff_mask = cv2.merge([diff_mask, diff_mask, diff_mask])
#     mog_mask = cv2.merge([mog_mask, mog_mask, mog_mask])
#     mog2_mask = cv2.merge([mog2_mask, mog2_mask, mog2_mask])

#     #Writing in the output file
#     out.write(frame)
#     out_diff.write(diff_mask)
#     out_mog.write(mog_mask)
#     out_mog2.write(mog2_mask)

#     #Showing the frame and waiting
#     # for the exit command
#     if(frame is None): break #check for empty frames
#     cv2.imshow('Original', frame) #show on window
#     cv2.imshow('Diff', diff_mask) #show on window
#     cv2.imshow('MOG', mog_mask) #show on window
#     cv2.imshow('MOG 2', mog2_mask) #show on window
#     if cv2.waitKey(1) & 0xFF == ord('q'): break #Exit when Q is pressed

# #Release the camera
# video_capture.release()
# print("Bye...")