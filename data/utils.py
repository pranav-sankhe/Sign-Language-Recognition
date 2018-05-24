import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.pyplot import figure, show
import tensorflow as tf
import os

NUM_MARKERS = 107         #Total Number of markers
MOTION_PARAMETERS = 6
selectCriterion_matrix = np.zeros(NUM_MARKERS)
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

def getSelectedBodParts(Selected_Markers):
    body_desc_path = "./data/body.csv"
    df = pd.read_csv(body_desc_path)
    body_desc_array = df.values 
    selected_body_desc = body_desc_array[Selected_Markers]
    selected_body_descDF = pd.DataFrame(selected_body_desc, Selected_Markers)
    pd.DataFrame.to_csv(selected_body_descDF, './data/selected_body_parts.csv' )
    return selected_body_desc


def Variance(mdata, filename):
    xdata = mdata[:,0]
    ydata = mdata[:,1]
    zdata = mdata[:,2]
    zrot = mdata[:,3]
    yrot = mdata[:,4]
    xrot = mdata[:,5]
    mdata = [xdata, ydata, zdata, zrot, yrot, xrot]
    mColumn_list = ['xpos', 'ypos', 'zpos', 'zrot', 'yrot', 'xrot']
    var_df = pd.DataFrame(columns=mColumn_list)
    var_df = var_df
    filename = os.path.splitext(filename)[0]
    for i in range(MOTION_PARAMETERS):
        var_df[mColumn_list[i]] = np.var(mdata[i], axis=1)
        var_df[mColumn_list[i]] = var_df[mColumn_list[i]].apply(lambda x: 0 if x < 1e-5 else x)
    

    pd.DataFrame.to_csv(var_df, './data/var.csv')   #save variance data in the data folder 
    

    var_sum = var_df[mColumn_list[0]]
    for i in range(1,6):
        var_sum = var_sum + var_df[mColumn_list[i]]     


    highVar_makers = np.where( var_sum > 1)    



    selectCriterion_matrix[highVar_makers] = selectCriterion_matrix[highVar_makers] + 1
    thresholdCriterion = 200
    numSelected = np.sum(selectCriterion_matrix >=thresholdCriterion)

    Selected_Markers = np.where(selectCriterion_matrix > thresholdCriterion)        
         

    # print "High variance markers", highVar_makers
    # print "variance values", var_sum.values[highVar_makers]   
    # print "selection criterion matrix", selectCriterion_matrix
    print "Number of seleted points ", numSelected , "  Number of markers not selected  ", NUM_MARKERS - numSelected 
    print "Selected matrix", Selected_Markers
    # fig = figure()
    # for i in range(1):
        

    #     ax = fig.add_subplot(111, xlim=(0,120), ylim=(0,3000), autoscale_on=False)

    #     ax.set_title('Click to zoom')
    #     ax.set_xlabel('markers')
    #     ax.set_ylabel('variance')
    #     ax.plot(var_df[mColumn_list[i]], label=mColumn_list[i])
    #     scale = 1.1
    #     zp = ZoomPan()
    #     figZoom = zp.zoom_factory(ax, base_scale = scale)
    #     figPan = zp.pan_factory(ax)
    #     ax.legend()
    #     #plt.savefig("./data/normalized_var_all/" + filename)

    # fig = figure()
    # plt.plot(var_sum)

    #plt.savefig("./data/normalized_var_sum/" + filename)
    # for i in range(MOTION_PARAMETERS):
    #   plt.plot(var_df[mColumn_list[i]], label=mColumn_list[i])
    #   plt.legend()
    #plt.show()  
    return Selected_Markers



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

        


def render_motionData(filepath):

    motionData = pd.read_csv(filepath)
    cLength = motionData.shape[1]
    cList = np.arange(cLength)
    motionData = pd.read_csv(filepath, names = cList)

    motionData = motionData.values
    motionData = np.reshape(motionData, (NUM_MARKERS, MOTION_PARAMETERS, cLength))
    return motionData


