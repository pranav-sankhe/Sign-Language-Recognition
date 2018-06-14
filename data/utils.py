import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.pyplot import figure, show
import tensorflow as tf
import os
import sys
import pdb
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math
import lmdb
from keras.utils import np_utils
from sklearn import preprocessing

BAD_JOINTS = np.array([8,12,13,14,15,16,17,18,19,20]) 

BATCH_SIZE = 100 
MAX_FRAME_LEN = 429 
NUM_FEATURES = 75

NUM_MARKERS = 107         #Total Number of markers
MOTION_PARAMETERS = 6
VARIANCE_THRESHOLD = 0    #for 1000 file iterations
THRESHOLD_CRITERION = 700 #for 1000 file iterations
PLOT_ARG = False
HANDCRAFTED = True
DROP_POSDATA = True

bad_filepath = './nturgbd_skeletons/samples_with_missing_skeletons.txt'


selectCriterion_matrix = np.zeros(NUM_MARKERS)
VARsumFlag = 0 
selec_var_sum = []


def num_elements(shape):
    num = 1
    for i in list(shape):
        num *= i
    return num

def range_len(rang):
    return rang[1]-rang[0]+1

def convert_to_one_hot(data, rang):

    data_shape = data.shape
    data_oned  = data.reshape(num_elements(data_shape))

    enc_data_shape = ( num_elements(data_shape), range_len(rang) )
    enc_data       = np.zeros(enc_data_shape)
    
    enc_data[np.arange(num_elements(data_shape)), data_oned] = 1

    enc_data_shape = list(data_shape)
    enc_data_shape.append(range_len(rang))
    enc_data = enc_data.reshape(enc_data_shape)

    return enc_data



class ZoomPan:
    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None


    def zoom_factory(self, ax, base_scale = 2.):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'down':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print event.button

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event',onPress)
        fig.canvas.mpl_connect('button_release_event',onRelease)
        fig.canvas.mpl_connect('motion_notify_event',onMotion)

        #return the function
        return onMotion

def render_motionData(filepath, prescaler):

    motionData = pd.read_csv(filepath)
    
    cLength = motionData.shape[1]
    cList   = np.arange(cLength)
    
    motionData = pd.read_csv(filepath, names = cList)

    motionData = motionData.values
    motionData = np.reshape(motionData, (NUM_MARKERS, MOTION_PARAMETERS, cLength))
    motionData = downsampleMotionData(motionData, prescaler)
    return motionData


def getSelectedBodParts(Selected_Markers):
    body_desc_path = "csv"
    df = pd.read_csv(body_desc_path)
    body_desc_array = df.values

    selected_body_desc = body_desc_array[Selected_Markers]
    Selected_Markers   = np.array(Selected_Markers)
    selected_body_desc = np.array(selected_body_desc)
    selected_body_desc = np.reshape(selected_body_desc, max(np.array(selected_body_desc.shape)))
    body_desc_array    = np.reshape(body_desc_array, max(np.array(body_desc_array.shape)))
    Selected_Markers   = np.reshape(Selected_Markers, max(np.array(Selected_Markers.shape)))


    selected_body_desc = np.pad(selected_body_desc, [0,len(body_desc_array) - len(selected_body_desc)], mode='constant', constant_values=0)
    Selected_Markers   = np.pad(Selected_Markers, [0,len(body_desc_array) - len(Selected_Markers)], mode='constant', constant_values=0)

    selected_body_descDF = pd.DataFrame({'selected body parts':selected_body_desc,'indices':Selected_Markers, 'all parts': body_desc_array })
    pd.DataFrame.to_csv(selected_body_descDF, 'selected_body_parts.csv' )
    return selected_body_desc

def store_variances(filename, var_data):
    pd.DataFrame.to_csv(var_data, '' + filename)

def drop_posData(DROP_POSDATA):
  return DROP_POSDATA

def handCrafted(HANDCRAFTED):
    return HANDCRAFTED



def Variance(mdata, filename, handcrafted, drop_posdata):
    
    xdata = mdata[:,0]
    ydata = mdata[:,1]
    zdata = mdata[:,2]
    zrot  = mdata[:,3]
    yrot  = mdata[:,4]
    xrot  = mdata[:,5]
    

    if drop_posdata == True:
        mdata = [zrot, yrot, xrot]
        mColumn_list = ['zrot', 'yrot', 'xrot']
                
    else:
        mdata = [xdata, ydata, zdata, zrot, yrot, xrot]                    #all motion data
        mColumn_list = ['xpos', 'ypos', 'zpos', 'zrot', 'yrot', 'xrot']         


    var_df = pd.DataFrame(columns=mColumn_list)                            # create a dataframe to store variances     
    var_df = var_df
    filename = os.path.splitext(filename)[0]                        
    for i in range(len(mColumn_list)):
        var_df[mColumn_list[i]] = np.var(mdata[i], axis=1)
        var_df[mColumn_list[i]] = var_df[mColumn_list[i]].apply(lambda x: 0 if x < 1e-5 else x)         # map low variances to zero
    
    if handCrafted(handcrafted) == True:
        for i in range(9):
            for j in range(3):
                var_df.iloc[i,j] = 0 
       
        
    body_desc_path = "body.csv"
    body_df = pd.read_csv(body_desc_path)
    body_df = body_df.values



    store_variances('var.csv', var_df)   #save variance data in the data folder 

    var_sum = var_df[mColumn_list[0]]               # compute the sum of variances.  Gives an idea of the motion of the marker
    for i in range(1,len(mColumn_list)):
        var_sum = var_sum + var_df[mColumn_list[i]]     

    varianceThreshold = VARIANCE_THRESHOLD                               
    highVar_makers    = np.where( var_sum > varianceThreshold)   #markers having a positive variances will be selected and the rest will be thrown out



    selectCriterion_matrix[highVar_makers] = selectCriterion_matrix[highVar_makers] + 1 # This will be iterated over all the files and a score is alloted to a marker is it has a positive variance
    numSelected = np.sum(selectCriterion_matrix >=THRESHOLD_CRITERION)                   #number of selected markers

    Selected_Markers = np.where(selectCriterion_matrix >= THRESHOLD_CRITERION)            #The indices of the selected markers after examining all the files since the threshold criterion matrix is computed for all the files            
         

    print "High variance markers", highVar_makers
    print "variance values", var_sum.values[highVar_makers]   
    print "selection criterion matrix", selectCriterion_matrix
    print "Number of seleted points ", numSelected , "  Number of markers not selected  ", NUM_MARKERS - numSelected 
    print "Selected matrix", Selected_Markers
    
    selected_markers_var = var_df.values[Selected_Markers]                            #The variances of selected markers only  
    selected_markers_var = pd.DataFrame(selected_markers_var, columns=mColumn_list)     
    store_variances('selected_var.csv', selected_markers_var)
    pd.DataFrame.to_csv(selected_markers_var, 'selected_var.csv')               #  

    if PLOT_ARG == True: 
        fig = figure()
        for i in range(1):
            

            ax = fig.add_subplot(111, xlim=(0,120), ylim=(0,3000), autoscale_on=False)

            ax.set_title('Click to zoom')
            ax.set_xlabel('markers')
            ax.set_ylabel('variance')
            ax.plot(var_df[mColumn_list[i]], label=mColumn_list[i])
            scale = 1.1
            zp = ZoomPan()
            figZoom = zp.zoom_factory(ax, base_scale = scale)
            figPan = zp.pan_factory(ax)
            ax.legend()
            #plt.savefig("normalized_var_all/" + filename)

        fig = figure()
        plt.plot(var_sum)

        plt.savefig("normalized_var_sum/" + filename)
        for i in range(MOTION_PARAMETERS):
          plt.plot(var_df[mColumn_list[i]], label=mColumn_list[i])
          plt.legend()
        plt.show()


    return Selected_Markers

def VARsum(selected_markers_var):
    mat = np.array(selected_markers_var)
    global selec_var_sum
    global VARsumFlag

    if VARsumFlag == 0 and len(mat) > 0:
        selec_var_sum = mat
        VARsumFlag = 1
    elif len(mat)>0:
        selec_var_sum = np.add(selec_var_sum, mat)
    return selec_var_sum        



def plot3D(mdata, interactiveMode=True):
    
    plt.ion()
    plt.show()
    xdata = mdata[:,0]
    ydata = mdata[:,1]
    zdata = mdata[:,2]
    zrot  = mdata[:,3]
    yrot  = mdata[:,4]
    xrot  = mdata[:,5]
    print mdata.shape[2]

    for timestamp in range(int(mdata.shape[2])):    
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter3D(xdata[:,timestamp], ydata[:,timestamp], zdata[:,timestamp], c=zdata[:,timestamp], cmap='spring_r');
        print "timestamp: ", timestamp
        plt.pause(0.001)    

def downsampleMotionData(data, prescaler):
    data = data[:, :, ::prescaler]
    return data

# def downsampleVideo(filepath):



# SpineBase = 1;
# SpineMid = 2;
# Neck = 3;
# Head = 4;
# ShoulderLeft = 5;
# ElbowLeft = 6;
# WristLeft = 7;

# HandLeft = 8;

# ShoulderRight = 9;
# ElbowRight = 10;
# WristRight = 11;

# HandRight = 12;

# HipLeft = 13;
# KneeLeft = 14;
# AnkleLeft = 15;
# FootLeft = 16;
# HipRight = 17;
# KneeRight = 18;
# AnkleRight = 19;
# FootRight = 20;

# SpineShoulder = 21;
# HandTipLeft = 22;
# ThumbLeft = 23;
# HandTipRight = 24;
# ThumbRight = 25;


def bad_file_txt(l):
    f = open(bad_filepath, "r")
    
    for line in f:
        line = line.strip('\n')
        l.append(line)
    return l 

def twoPpl_list(l,filepath):  
    filename = os.path.basename(filepath)
    filename = filename.split('.')[0]
    actionLabel = filename[len(filename) - 3:]
    actionLabel = int(actionLabel)
    if actionLabel > 49:
        l.append(filename)
        print "two people detected"
    f = open(filepath, "r")                     # Open the file
    numFrames = f.readline()                    # Read the 1st line of the file which gives the number of frames
    numFrames = int(numFrames)                  # convert string to int
        
    for i in range(numFrames):                  # for each frame loop over the contents
        skeletons = f.readline()                      # a constant 1
       
        skeletons = int(skeletons)

        if skeletons != 1: 
            print "two people detected"
            l.append(os.path.basename(filepath).split('.')[0])
            f.close()
            #print l
            return l
        meta_data = f.readline()                # a line containing all the meta data     
        numJoints = f.readline()                # Read the number of joints. Always equal to 25
        numJoints = int(numJoints)              # Convert string to int     
         
 
        for j in range(numJoints):
            motion_line = f.readline()         #read line containing data of a single joint    


    return l      
                                  
       

def get_max_frame_count(filepath, l):
    f = open(filepath, "r")
    numFrames = f.readline()
    numFrames = int(numFrames)    
    l.append(numFrames)
    return l

def get_max_time_steps(filepath, l):
    data = render_motionData(filepath, prescaler=2)
    l.append(data.shape[2])
    return l


def readSkeletonFiles(filepath):
    f = open(filepath, "r")                     # Open the file
    numFrames = f.readline()                    # Read the 1st line of the file which gives the number of frames
    numFrames = int(numFrames)                  # convert string to int
    motion_data = np.zeros((numFrames, NUM_FEATURES)) # store motion data of file in this matrix which we will return 
    print "Reading file: ", os.path.basename(filepath).split('.')[0] 
    for i in range(numFrames):                  # for each frame loop over the contents
        skeletons = f.readline()                      # a constant 1


        meta_data = f.readline()                # a line containing all the meta data 
        #print meta_data
        meta_data = meta_data.split() 

        bodyID              = float(meta_data[0])       # extrcat all meta data
        clipedEdges         = float(meta_data[1])
        handLeftConfidence  = float(meta_data[2])
        handLeftState       = float(meta_data[3])
        handRightConfidence = float(meta_data[4])
        handRightState      = float(meta_data[5])
        isResticted         = float(meta_data[6])     
        xLean               = float(meta_data[7])
        yLean               = float(meta_data[8])
        trackingState       = float(meta_data[9]) 

        
        numJoints = f.readline()                # Read the number of joints. Always equal to 25
        numJoints = int(numJoints)              # Convert string to int     
         
        count = 0
        iterator = 0 
        for j in range(numJoints):
            motion_line = f.readline()         #read line containing data of a single joint    
            motion_line = motion_line.split()
                        
            # if j + 1 in BAD_JOINTS:             # check if the joint is unwanted 
            #     #print "bad joint detected"
            #     continue     
            

            x_pos = float(motion_line[0])
            y_pos = float(motion_line[1])
            z_pos = float(motion_line[2])
            #print x_pos, y_pos, z_pos
            motion_data[i,3*iterator] = float(motion_line[0])
            motion_data[i,3*iterator+1] = float(motion_line[1])            
            motion_data[i,3*iterator+2] = float(motion_line[2])
            iterator = iterator + 1                    
    
    filename = os.path.basename(filepath)
    filename = filename.split('.')[0]
    actionLabel = filename[len(filename) - 3:]
    #print actionLabel
    actionLabel = int(actionLabel)
    #motion_data = preprocessing.normalize(motion_data, norm='l2', axis=0)
    #print motion_data

                
    return [motion_data, actionLabel]


VOCAB_SIZE = 277 + 2 

def embeddings():
    
    # Embedding
    embedding_encoder = variable_scope.get_variable(
        "embedding_encoder", [VOCAB_SIZE, 512])
    # Look up embedding:
    #   encoder_inputs: [max_time, batch_size]
    #   encoder_emb_inp: [max_time, batch_size, embedding_size]
    encoder_emb_inp = embedding_ops.embedding_lookup(
        embedding_encoder, encoder_inputs)
