import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional 
from keras.layers import BatchNormalization, Dropout, Flatten, Merge, Activation
from keras.models import load_model
from keras.layers import concatenate
from matplotlib import pyplot as plt
from IPython.display import clear_output

import os
import sys
sys.path.append(os.path.abspath("../data"))

import utils


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()




# create a sequence classification instance
config = tf.ConfigProto( device_count = {'GPU': 4 } ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


BATCH_SIZE = 100 
MAX_FRAME_LEN = 429 
NUM_FEATURES = 45

INPUT_SIZE = (BATCH_SIZE, MAX_FRAME_LEN, NUM_FEATURES)           # input size =  (batch size, number of time steps, hidden size) 
OUTPUT_SIZE = 48                    
FULLY_CONNECTED_SIZE = 512
HIDDEN_LAYER_1 = 256 
HIDDEN_LAYER_2 = 256

NUM_EPOCHS = 462



def get_skeleton_batch_data():
    BASE_PATH = '../data/nturgbd_skeletons/nturgb+d_skeletons'
    
    file_list = os.listdir(BASE_PATH)
    file_list = np.array(file_list)
    bad_files = np.load('../data/badfiles.npy')
    for i in range(len(file_list)):
        file_list[i] = file_list[i].split('.')[0]
    file_list = np.setdiff1d(file_list, bad_files)
    t = BATCH_SIZE*NUM_EPOCHS
    data = np.zeros((t, MAX_FRAME_LEN, NUM_FEATURES))
    labels = [] #np.zeros((t, MAX_FRAME_LEN, NUM_FEATURES))
    count = 0 
    for i in range(len(file_list)):                
        print "Reading file ", file_list[i] , " filename ", count
        count = count + 1  
        filepath = BASE_PATH +'/' + file_list[i] + '.skeleton' 
        train_data = utils.readSkeletonFiles(filepath)
        motion_data = train_data[0]
        label = train_data[1]
        motion_data = np.pad(motion_data, ((0,MAX_FRAME_LEN - motion_data.shape[0]), (0,0)), mode='constant', constant_values=0)    

        data[i, :, : ] = motion_data
        for j in range(MAX_FRAME_LEN):
            for l in range(NUM_FEATURES):
                labels.append(label)       
        #print data
        #print labels
    labels = np.array(labels)    
    np.save('data', data)
    np.save('labels', labels)

    return data, labels    

#get_skeleton_batch_data()

# define Model

# gru = GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', 
#                       recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, 
#                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
#                       dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, 
#                       unroll=False, reset_after=False)

# lstm = LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
#                       bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
#                       kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, 
#                       return_state=False, go_backwards=False, stateful=False, unroll=False)

model = Sequential()


model.add(Bidirectional(GRU(HIDDEN_LAYER_1,  return_sequences=True), batch_input_shape=(BATCH_SIZE, MAX_FRAME_LEN, NUM_FEATURES)))
#model.add(BatchNormalization(epsilon=1e-5))
model.add(Bidirectional(GRU(HIDDEN_LAYER_2, return_sequences=True)))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(BatchNormalization(epsilon=1e-5)))
model.add(TimeDistributed(Dropout(0.25)))

model.add(TimeDistributed(Dense(OUTPUT_SIZE, activation='relu')))    #Fully Connected layer

rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.9)

model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc'])


# def fork (model, n=2):
#     forks = []
#     for i in range(n):
#         f = Sequential()
#         f.add (model)
#         forks.append(f)
#     return forks

# left = Sequential()
# left.add(LSTM(output_dim=HIDDEN_LAYER_1, init='uniform', inner_init='uniform',
#                forget_bias_init='one', return_sequences=True, activation='tanh',
#                inner_activation='sigmoid', input_shape=(MAX_FRAME_LEN, NUM_FEATURES)))
# right = Sequential()
# right.add(LSTM(output_dim=HIDDEN_LAYER_1, init='uniform', inner_init='uniform',
#                forget_bias_init='one', return_sequences=True, activation='tanh',
#                inner_activation='sigmoid', input_shape=(MAX_FRAME_LEN, NUM_FEATURES), go_backwards=True))

# model = Sequential()
# model.add(Merge([left, right], mode='concat'))
# #concatenate([x1, x2], axis=-1)
# #Add second Bidirectional LSTM layer
# print "MODEL", model
# left, right = fork(model)

# left.add(LSTM(output_dim=HIDDEN_LAYER_2, init='uniform', inner_init='uniform',
#                forget_bias_init='one', return_sequences=True, activation='tanh',
#                inner_activation='sigmoid'))

# right.add(LSTM(output_dim=HIDDEN_LAYER_2, init='uniform', inner_init='uniform',
#                forget_bias_init='one', return_sequences=True, activation='tanh',
#                inner_activation='sigmoid',  go_backwards=True))

# #Rest of the stuff as it is

# model = Sequential()
# model.add(Merge([left, right], mode='concat'))

# model.add(TimeDistributed(Dense(FULLY_CONNECTED_SIZE, activation='relu')))
# model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
# model.add(Activation('softmax'))

# rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.9)

# model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
print("Train...")

X_train = np.load('data.npy')
Y_train = np.load('labels.npy')

print type(X_train)
print type(Y_train)
print X_train.shape
print Y_train.shape
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, verbose=2,callbacks=[plot_losses])


model.save('bi-GRU.h5')

    # # evaluate LSTM
# X,y = get_sequence(INPUT_SIZE)
# yhat = model.predict_classes(X, verbose=0)
# for i in range(n_timesteps):
#   print('Expected:', y[0, i], 'Predicted', yhat[0, i])    