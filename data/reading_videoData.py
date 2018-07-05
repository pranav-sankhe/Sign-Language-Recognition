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
FRAME_MEAN = 39
IN_CHANNELS = FRAME_MEAN
NUM_FRAMES = 858
BASE_PATH = '/home/data/All_video_data/HRIJ_0130_movie'

l_numframe = []

filepath = '/home/data/02_CSV_REAL/01_CSV_full/memo.txt'
# f = open(filepath, "r")                     # Open the file
# for line in f:
#     l_numframe.append(int(line[len(line) - 5:len(line)]))

filenames = os.listdir(BASE_PATH)
filenames = filenames[0:BATCH_SIZE*NUM_EPOCHS]

def save_frames_hri():
    if not os.path.exists('video_frames'):
        os.makedirs('video_frames')

    for i in range(len(filenames)):
        if filenames[i].split('.')[1] != 'mp4':
            continue
        if not os.path.exists('video_frames' + '/' + filenames[i].split('.')[0]):
            os.makedirs('video_frames' + '/' + filenames[i].split('.')[0])

            
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
        cv2.imwrite('video_frames' + '/' + filenames[i].split('.')[0] + '/' +  num + '.jpg', frame1_save)        
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
            if int(count)%8 == 0:
                cv2.imwrite('video_frames' + '/' + filenames[i].split('.')[0] + '/'  + num + '.jpg', frame2_save)

            count = count + 1
       
        return
        cap.release()



def opflow_xy():
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
            nOuterFPIterations = 7
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


folders = ["nturgbd_rgb_s001", "nturgbd_rgb_s002", "nturgbd_rgb_s003", "nturgbd_rgb_s004", "nturgbd_rgb_s005", "nturgbd_rgb_s006", "nturgbd_rgb_s007", "nturgbd_rgb_s008", "nturgbd_rgb_s009",
           "nturgbd_rgb_s010", "nturgbd_rgb_s011", "nturgbd_rgb_s012", "nturgbd_rgb_s013", "nturgbd_rgb_s014", "nturgbd_rgb_s015", "nturgbd_rgb_s016", "nturgbd_rgb_s017"]

def opflow_xy_pretrain():
    if not os.path.exists('opflow_xy'):
        os.makedirs('opflow_xy')
    BASE_PATH = "/media/user/DATA/newFolder"
    folders = os.listdir(BASE_PATH)
    folders = np.sort(folders)
    for folder in folders: 
        if os.path.exists('opflow_xy' + '/' + folder):
            os.makedirs('opflow_xy' + '/' + folder)
        videos = os.listdir(BASE_PATH + '/' + folder)
        videos = np.sort(videos) 
        
        for video in videos:
            img_files = os.listdir(BASE_PATH + '/' + folder + '/' + video)
            img_files = np.sort(img_files)
            frame1 = img_files[0]

            if not os.path.exists('opflow_xy' + '/' + folder + '/' + video):
                os.makedirs('opflow_xy' + '/' + folder + '/' + video)
            for i in range(len(img_files)-1):
                print "Reading file" , BASE_PATH + '/' + folder + '/' + video + img_files[i]
                frame2 = img_files[i + 1]
                im1 = np.asarray(Image.open(BASE_PATH + '/' + folder + '/' + video + '/' + frame1))
                im2 = np.asarray(Image.open(BASE_PATH + '/' + folder + '/' + video + '/' + frame2))
                im1 = im1.astype(float) / 255.
                im2 = im2.astype(float) / 255.

                alpha = 0.012
                ratio = 0.75
                minWidth = 20
                nOuterFPIterations = 7
                nInnerFPIterations = 1
                nSORIterations = 15
                colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))


                u, v, im2W = pyflow.coarse2fine_flow(
                    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                    nSORIterations, colType)
                u = u*255
                u = np.ceil(u)
                u = np.int16(u)

                v = v*255
                v = np.ceil(v)
                v = np.int16(v)

                cv2.imwrite('opflow_xy' + '/' + folder + '/' + video + '/' + '_x_' + str(i) + '.jpg', u)
                cv2.imwrite('opflow_xy' + '/' + folder + '/' + video + '/' + '_y_' + str(i) + '.jpg', v)
                frame1 = frame2
  

def savefile_pretrain():
    if not os.path.exists('newFolder'):
        os.makedirs('newFolder')

    for folder in folders: 
        filenames = os.listdir(folder + '/' + 'nturgb+d_rgb')
        if not os.path.exists('newFolder'+ '/' + folder):
            os.makedirs('newFolder'+ '/' + folder)        

        for file in filenames:
            file_top = file.split('.')[0]
            action_class = file_top[-7:-4]
            action_class = int(action_class)
            
            if action_class > 49:
                continue
            else:
                vidcap = cv2.VideoCapture(folder + '/' + 'nturgb+d_rgb' + '/' + file)
                success,image = vidcap.read()
                count = 0 
                if not os.path.exists('newFolder'+ '/' + folder + '/' + file.split('.')[0]):
                    os.makedirs('newFolder'+ '/' + folder + '/' + file.split('.')[0])                
                while success:
                    success, image = vidcap.read()
                    if success==False:
                        break
                    resize = cv2.resize(image, (256, 256), interpolation = cv2.INTER_CUBIC) 
                    path = 'newFolder' + '/' + folder + '/' + file.split('.')[0] +'/'+ str(count) + '.png'
                    cv2.imwrite(path, resize)     
                    count = count + 1 
                    if cv2.waitKey(10) == 27:                     
                        break

save_frames_hri()