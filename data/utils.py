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


NUM_MARKERS = 107         #Total Number of markers
MOTION_PARAMETERS = 6
VARIANCE_THRESHOLD = 0    #for 1000 file iterations
THRESHOLD_CRITERION = 700 #for 1000 file iterations
PLOT_ARG = False
HANDCRAFTED = True
DROP_POSDATA = True

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
    data_oned = data.reshape(num_elements(data_shape))

    enc_data_shape = ( num_elements(data_shape), range_len(rang) )
    enc_data = np.zeros(enc_data_shape)
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

def render_motionData(filepath):

    motionData = pd.read_csv(filepath)
    cLength = motionData.shape[1]
    cList = np.arange(cLength)
    motionData = pd.read_csv(filepath, names = cList)

    motionData = motionData.values
    motionData = np.reshape(motionData, (NUM_MARKERS, MOTION_PARAMETERS, cLength))
    return motionData


def getSelectedBodParts(Selected_Markers):
    body_desc_path = "body.csv"
    df = pd.read_csv(body_desc_path)
    body_desc_array = df.values

    selected_body_desc = body_desc_array[Selected_Markers]
    Selected_Markers = np.array(Selected_Markers)
    selected_body_desc = np.array(selected_body_desc)
    selected_body_desc = np.reshape(selected_body_desc, max(np.array(selected_body_desc.shape)))
    body_desc_array = np.reshape(body_desc_array, max(np.array(body_desc_array.shape)))
    Selected_Markers = np.reshape(Selected_Markers, max(np.array(Selected_Markers.shape)))


    selected_body_desc = np.pad(selected_body_desc, [0,len(body_desc_array) - len(selected_body_desc)], mode='constant', constant_values=0)
    Selected_Markers = np.pad(Selected_Markers, [0,len(body_desc_array) - len(Selected_Markers)], mode='constant', constant_values=0)

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
    zrot = mdata[:,3]
    yrot = mdata[:,4]
    xrot = mdata[:,5]
    

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
    highVar_makers = np.where( var_sum > varianceThreshold)   #markers having a positive variances will be selected and the rest will be thrown out



    selectCriterion_matrix[highVar_makers] = selectCriterion_matrix[highVar_makers] + 1 # This will be iterated over all the files and a score is alloted to a marker is it has a positive variance
    thresholdCriterion = THRESHOLD_CRITERION
    numSelected = np.sum(selectCriterion_matrix >=thresholdCriterion)                   #number of selected markers

    Selected_Markers = np.where(selectCriterion_matrix >= thresholdCriterion)            #The indices of the selected markers after examining all the files since the threshold criterion matrix is computed for all the files            
         

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
    # selec_var_sum = np.transpose(selec_var_sum)
    return selec_var_sum        



def plot3D(mdata, interactiveMode=True):
    
    plt.ion()
    plt.show()
    xdata = mdata[:,0]
    ydata = mdata[:,1]
    zdata = mdata[:,2]
    zrot = mdata[:,3]
    yrot = mdata[:,4]
    xrot = mdata[:,5]
    print mdata.shape[2]

    for timestamp in range(int(mdata.shape[2])):    
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter3D(xdata[:,timestamp], ydata[:,timestamp], zdata[:,timestamp], c=zdata[:,timestamp], cmap='spring_r');
        print "timestamp: ", timestamp
        plt.pause(0.001)    

        




classes = {
    0: 'drink water',
    1: 'eat meal/snack',
    2: 'brushing teeth',
    3: 'brushing hair',
    4: 'drop',
    5: 'pickup',
    6: 'throw',
    7: 'sitting down',
    8: 'standing up (from sitting position)',
    9: 'clapping',
    10: 'reading',
    11: 'writing',
    12: 'tear up paper',
    13: 'wear jacket',
    14: 'take off jacket',
    15: 'wear a shoe',
    16: 'take off a shoe',
    17: 'wear on glasses',
    18: 'take off glasses',
    19: 'put on a hat/cap',
    20: 'take off a hat/cap',
    21: 'cheer up',
    22: 'hand waving',
    23: 'kicking something',
    24: 'put something inside pocket / take out something from pocket',
    25: 'hopping (one foot jumping)',
    26: 'jump up',
    27: 'make a phone call/answer phone',
    28: 'playing with phone/tablet',
    29: 'typing on a keyboard',
    30: 'pointing to something with finger',
    31: 'taking a selfie',
    32: 'check time (from watch)',
    33: 'rub two hands together',
    34: 'nod head/bow',
    35: 'shake head',
    36: 'wipe face',
    37: 'salute',
    38: 'put the palms together',
    39: 'cross hands in front (say stop)',
    40: 'sneeze/cough',
    41: 'staggering',
    42: 'falling',
    43: 'touch head (headache)',
    44: 'touch chest (stomachache/heart pain)',
    45: 'touch back (backache)',
    46: 'touch neck (neckache)',
    47: 'nausea or vomiting condition',
    48: 'use a fan (with hand or paper)/feeling warm',
    49: 'punching/slapping other person',
    50: 'kicking other person',
    51: 'pushing other person',
    52: 'pat on back of other person',
    53: 'point finger at the other person',
    54: 'hugging other person',
    55: 'giving something to other person',
    56: 'touch other persons pocket',
    57: 'handshaking',
    58: 'walking towards each other',
    59: 'walking apart from each other'
    }

training_subjects = [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]
_EPS = np.finfo(float).eps * 4.0
#feat_dim = 150 
feat_dim = 350                # dimesion of features

def quaternion_matrix(q):
  """Return homogeneous rotation matrix from quaternion.
  """
  qw = q[0]
  qx = q[1]
  qy = q[2]
  qz = q[3]

  return np.array([
      [1-2*qy*qy - 2*qz*qz,       2*qx*qy - 2*qz*qw,        2*qx*qz + 2*qy*qw],
      [2*qx*qy + 2*qz*qw    ,     1-2*qx*qx - 2*qz*qz,      2*qy*qz - 2*qx*qw],
      [2*qx*qz - 2*qy*qw    ,     2*qy*qz + 2*qx*qw,        1-2*qx*qx - 2*qy*qy],
      ])

def matrix_quaternion(R):  
  trace = np.trace(R)
  if trace > 0:
    s = 0.5 / np.sqrt(trace+1.0)
    qw = 0.25 / s 
    qx = (R[2,1]-R[1,2])*s
    qy = (R[0,2]-R[2,0])*s
    qz = (R[1,0]-R[0,1])*s
  else:
    if ( R[0,0] > R[1,1] and R[0,0] > R[2,2] ):
      s = 2.0 * np.sqrt( 1.0 + R[0,0] - R[1,1] - R[2,2])
      qw = (R[2,1] - R[1,2]) / s
      qx = 0.25 * s
      qy = (R[0,1] + R[1,0]) / s
      qz = (R[0,2] + R[2,0]) / s
    elif (R[1,1] > R[2,2]):
      s = 2.0 * np.sqrt( 1.0 + R[1,1] - R[0,0] - R[2,2])
      qw = (R[0,2] - R[2,0] ) / s
      qx = (R[0,1] + R[1,0] ) / s
      qy = 0.25 * s
      qz = (R[1,2] + R[2,1] ) / s
    else:
      s = 2.0 * np.sqrt( 1.0 + R[2,2] - R[0,0] - R[1,1] )
      qw = (R[1,0] - R[0,1] ) / s
      qx = (R[0,2] + R[2,0] ) / s
      qy = (R[1,2] + R[2,1] ) / s
      qz = 0.25 * s
  
  return np.array([qw,qx,qy,qz])


def draw_skeleton(joints_list,show=0,save=0,autoscale=0,outdir="",outname=""):
  #plt.ion()
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  colors = ['red','blue','green']
  if autoscale:
    X = []
    Y = []
    Z = []
  for jj in range(0,len(joints_list)):
    joints = joints_list[jj]
    for num,joint in joints.iteritems():
      ax.scatter(joint[0],joint[1],joint[2],color=colors[jj])
      if autoscale:
        X.append(joint[0])
        Y.append(joint[1])
        Z.append(joint[2])

    connectivity = [(0,1),(1,20),(20,2),(2,3),(20,8),(8,9),(9,10),(10,11),(11,24),(24,23),
        (20,4),(4,5),(5,6),(6,7),(7,22),(22,21),(0,16),(16,17),(17,18),(18,19),(0,12),(12,13),(13,14),(14,15)]
    
    for connection in connectivity:
      t = connection[0]
      f = connection[1]

      ax.plot([joints[f][0],joints[t][0]],[joints[f][1],joints[t][1]],[joints[f][2],joints[t][2]])
    ax.plot([joints[8][0],joints[4][0]],[joints[8][1],joints[4][1]],[joints[8][2],joints[4][2]],color='black')
 

    if autoscale:
      ax.set_xlabel('X Label')
      ax.set_xlim(np.min(X),np.max(X))
      ax.set_ylabel('Y Label')
      ax.set_ylim(np.min(Y),np.max(Y))
      ax.set_zlabel('Z Label')
      ax.set_zlim(np.min(Z),np.max(Z))
    else:
      ax.set_xlabel('X Label')
      ax.set_xlim(-2,2)
      ax.set_ylabel('Y Label')
      ax.set_ylim(-4,4)
      ax.set_zlabel('Z Label')
      ax.set_zlim(-5,5)
    ax.view_init(elev=90., azim=90)
  #ax = fig.add_subplot(122)
  #ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
  if show:
    plt.show()
  
  if save:
    plt.savefig(outdir+outname+".png")


def normalize_skeleton(jinfo,anchor=None,norm_dist=0,right_to_left=None,spine_to_top=None):
  if anchor == None and norm_dist == 0:
    anchor = np.array([jinfo[1][0],jinfo[1][1],jinfo[1][2]])
    base = np.array([jinfo[0][0],jinfo[0][1],jinfo[0][2]])
    norm_dist = np.linalg.norm(anchor-base) 


  ## TRANSLATE TO SPINE ORIGIN FIRST  
  norm_joints = {}
  for jnum, unnorm_joint in jinfo.iteritems():
    normalized_pos = np.array([unnorm_joint[0]-anchor[0],
                      unnorm_joint[1]-anchor[1],
                      unnorm_joint[2]-anchor[2]]) #/ norm_dist
    norm_joints[jnum] = normalized_pos

  if right_to_left is None:
    right_to_left = np.array([norm_joints[8][0],norm_joints[8][1],norm_joints[8][2]]) - np.array([norm_joints[4][0],norm_joints[4][1],norm_joints[4][2]])
    right_to_left = right_to_left / (np.linalg.norm(right_to_left)+_EPS)
  

  ## COMPUTE ROTATION SUCH THAT RIGHT TO LEFT IS PARALLEL TO X_AXIS
  x_axis = np.array([1,0,0])
  y_axis = np.array([0,1,0])

  new_x = right_to_left
  new_y = np.cross(right_to_left,x_axis)
  new_y = new_y/(np.linalg.norm(new_y)+_EPS)
  new_z = np.cross(new_x,new_y)
  new_z = new_z/(np.linalg.norm(new_z)+_EPS)
  Rx = np.transpose(np.array([new_x,new_y,new_z]))
  #pdb.set_trace()


  rotated_and_norm = {}
  for jnum, joint in norm_joints.iteritems():
    turn_to_x = np.dot(joint,Rx)
    #turn_to_y = np.dot(Ry,turn_to_x)
    rotated_and_norm[jnum] = turn_to_x 

  new_right_to_left = np.array([rotated_and_norm[8][0],rotated_and_norm[8][1],rotated_and_norm[8][2]]) - np.array([rotated_and_norm[4][0],rotated_and_norm[4][1],rotated_and_norm[4][2]])
  

  if spine_to_top is None:
    spine_to_top = np.array([rotated_and_norm[0][0],rotated_and_norm[0][1],rotated_and_norm[0][2]]) - np.array([rotated_and_norm[1][0],rotated_and_norm[1][1],rotated_and_norm[1][2]])
    spine_to_top = spine_to_top / (np.linalg.norm(spine_to_top)+_EPS)
  

  ## COMPUTE ROTATION SUCH THAT SPINE TO TOP IS PARALLEL TO Y_AXIS
  new_y = spine_to_top
  new_x = np.cross(spine_to_top,y_axis)
  new_x = new_x/(np.linalg.norm(new_x)+_EPS)
  new_z = np.cross(new_x,new_y)
  new_z = new_z/(np.linalg.norm(new_z)+_EPS)
  Ry = np.transpose(np.array([new_x,new_y,new_z]))


  rotated_and_norm2 = {}
  for jnum, joint in rotated_and_norm.iteritems():
    #turn_to_x = np.dot(Rx,joint)
    turn_to_y = np.dot(joint,Ry)

    qR = quaternion_matrix(jinfo[jnum][3:])
    rotated_q = np.dot(np.dot(qR,Rx),Ry)
    qq = matrix_quaternion(rotated_q)

    normed_final_vec = turn_to_y / (norm_dist+_EPS)
    
    rotated_and_norm2[jnum] = np.concatenate((normed_final_vec,qq))

  return rotated_and_norm2,anchor,norm_dist,right_to_left,spine_to_top


def vids_with_missing_skeletons():
  f = open('/home/tk/dev/data/nturgbd/samples_with_missing_skeletons.txt','r')
  bad_files = []
  for line in f:
    bad_files.append(line.strip()+'.skeleton')
  f.close()
  return bad_files

def generate_data(argv):
  bad_files = vids_with_missing_skeletons()
  skeleton_dir_root = "/home/tk/dev/data/nturgbd/nturgb+d_skeletons"
  skeleton_files = os.listdir(skeleton_dir_root)
  data_out_dir = '/media/tk/EE44DA8044DA4B4B/subjects_split_rot_norm_quat/'

  #sk_info = {} # key: file_name, value: corresponding vid_info dict

  max_vid_length = -1
  X_train = []
  X_test = []
  y_train = []
  y_test = []
  n_classes = 60

  num_files = len(skeleton_files)
  count = 0
  for file_name in skeleton_files:
    if file_name in bad_files:
      continue
    action_class = int(file_name[file_name.find('A')+1:file_name.find('A')+4])
    subject_id = int(file_name[file_name.find('S')+1:file_name.find('S')+4])

    sf = open(os.path.join(skeleton_dir_root,file_name),'r')
    num_frames = int(sf.readline())
    #vid_info = dict() ## key=frame_num,  value=body info dicts
  
    feature = np.zeros((num_frames, feat_dim))
    for n in range(0,num_frames):
      body_count = int(sf.readline())
      #print body_count

      if body_count > 2:
        # JUST ITERATE THROUGH THE LINES, IGNORE 
        for b in range(0,body_count):
          body_info = sf.readline()
          joint_count = int(sf.readline())
          for j in range(0,joint_count):
            joint_info = sf.readline()
      else:
        
        binfo = dict()
        norm_dist = 0
        anchor = None
        right_to_left = None
        spine_to_top = None
        for b in range(0,body_count):
          body_info = sf.readline()
          bsp = body_info.split()
          
          body_id = bsp[0]
          cliped_edges = bsp[1]
          lefthand_confidence = bsp[2]
          lefthand_state = bsp[3]
          righthand_confidence = bsp[4]
          righthand_state = bsp[5]
          is_restricted = bsp[6]
          lean_x = bsp[7]
          lean_y = bsp[8]
          body_tracking_state = bsp[9]

          #binfo[b] = bsp
          joint_count = int(sf.readline()) ## ASSUMING THIS IS ALWAYS 25


          jinfo = dict()
          
          for j in range(0,joint_count):
            joint_info = sf.readline()
            jsp = joint_info.split()
            x = float(jsp[0])
            y = float(jsp[1])
            z = float(jsp[2])
            depth_x = float(jsp[3])
            depth_y = float(jsp[4])
            rgb_x = float(jsp[5])
            rgb_y = float(jsp[6])
            rw = float(jsp[7])
            rx = float(jsp[8])
            ry = float(jsp[9])
            rz = float(jsp[10])
            joint_tracking_state = jsp[11]

            jinfo[j] = (x,y,z,rw,rx,ry,rz)
          ## END JOINT LOOP
          
          
          norm_jinfo,anchor,norm_dist,right_to_left,spine_to_top = normalize_skeleton(jinfo,anchor=anchor,norm_dist=norm_dist,right_to_left=right_to_left,spine_to_top=spine_to_top)
          
          binfo[b] = norm_jinfo
      
        ## END BODY LOOP
        sample_ind = 0
        sample = np.zeros(feat_dim)
        
        ## CONSTRUCT THE FEATURE FOR THIS N-th FRAME
        for bind, body in binfo.iteritems():
          for jind, joint in body.iteritems():
            sample[sample_ind] =   joint[0] #x
            sample_ind += 1
            sample[sample_ind] =   joint[1] #y
            sample_ind += 1
            sample[sample_ind] =   joint[2] #z
            sample_ind += 1
        feature[n] = sample
      ## END BODY COUNT IF-ELSE
    ## END FRAME LOOP

    if body_count <= 2:
      if subject_id in training_subjects:
        X_train.append(feature)
        y_train.append(action_class-1)
      else:
        X_test.append(feature)
        y_test.append(action_class-1)
    sf.close()
    count += 1
    if count % 100 == 0:
      print count,"/",num_files
  ## END FILE LOOP


  print "Writing out data . . . "
  
  train_max_len = max([len(c) for c  in X_train])
  test_max_len = max([len(c) for c  in X_test])
  max_len = max(train_max_len,test_max_len)
  print max_len
  step_size = 2000
  

  ## WRITE OUT TRAIN
  index = 0 
  batch_size = 128
  lmdb_file_x = os.path.join(data_out_dir,'Xtrain_lmdb')
  lmdb_file_y = os.path.join(data_out_dir,'Ytrain_lmdb')

  lmdb_env_x = lmdb.open(lmdb_file_x, map_size=int(1e12))
  lmdb_env_y = lmdb.open(lmdb_file_y, map_size=int(1e12))
  lmdb_txn_x = lmdb_env_x.begin(write=True)
  lmdb_txn_y = lmdb_env_y.begin(write=True)

  item_id = -1
  for i in range(0,len(X_train)):
    item_id += 1
    keystr = '{:0>8d}'.format(item_id)

    X = np.zeros((max_len,feat_dim))
    num_rows = X_train[i].shape[0]
    X[0:num_rows] =  X_train[i]
    Y = np_utils.to_categorical(y_train[i], n_classes)

    lmdb_txn_x.put( keystr, X.tobytes() )
    lmdb_txn_y.put( keystr, Y.tobytes() )

    # write batch
    if(item_id + 1) % batch_size == 0:
      lmdb_txn_x.commit()
      lmdb_txn_x = lmdb_env_x.begin(write=True)
      lmdb_txn_y.commit()
      lmdb_txn_y = lmdb_env_y.begin(write=True)
      print (item_id + 1)
          #pdb.set_trace()

  # write last batch
  if (item_id+1) % batch_size != 0:
    lmdb_txn_x.commit()
    lmdb_txn_y.commit()
    print 'last batch'
    print (item_id + 1)

  print "WROTE TRAINING"

  lmdb_file_x = os.path.join(data_out_dir,'Xtest_lmdb')
  lmdb_file_y = os.path.join(data_out_dir,'Ytest_lmdb')

  lmdb_env_x = lmdb.open(lmdb_file_x, map_size=int(1e12))
  lmdb_env_y = lmdb.open(lmdb_file_y, map_size=int(1e12))
  lmdb_txn_x = lmdb_env_x.begin(write=True)
  lmdb_txn_y = lmdb_env_y.begin(write=True)

  item_id = -1
  for i in range(0,len(X_test)):
    item_id += 1
    keystr = '{:0>8d}'.format(item_id)

    X = np.zeros((max_len,feat_dim))
    num_rows = X_test[i].shape[0]
    X[0:num_rows] =  X_test[i]
    Y = np_utils.to_categorical(y_test[i], n_classes)

    lmdb_txn_x.put( keystr, X.tobytes() )
    lmdb_txn_y.put( keystr, Y.tobytes() )

    # write batch
    if(item_id + 1) % batch_size == 0:
      lmdb_txn_x.commit()
      lmdb_txn_x = lmdb_env_x.begin(write=True)
      lmdb_txn_y.commit()
      lmdb_txn_y = lmdb_env_y.begin(write=True)
      print (item_id + 1)
          #pdb.set_trace()

  # write last batch
  if (item_id+1) % batch_size != 0:
    lmdb_txn_x.commit()
    lmdb_txn_y.commit()
    print 'last batch'
    print (item_id + 1)

  #pdb.set_trace()
  print "WROTE TESTING"
  print "TRAINING SAMPLES: ",len(X_train), "TESTING SAMPLES:", len(X_test)
  
  

# if __name__ == "__main__":
#     generate_data(sys.argv)