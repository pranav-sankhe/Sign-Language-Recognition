import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional 
from keras.layers import BatchNormalization, Dropout
from keras.models import load_model

# create a sequence classification instance
config = tf.ConfigProto( device_count = {'GPU': 4 } ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


BATCH_SIZE = 100 
MAX_FRAME_LEN = 429 
NUM_FEATURES = 45

INPUT_SIZE = (BATCH_SIZE, MAX_FRAME_LEN, NUM_FEATURES)           # input size =  (batch size, number of time steps, hidden size) 
OUTPUT_SIZE = 60			 		
FULLY_CONNECTED_SIZE = 512
HIDDEN_LAYER_1 = 256 
HIDDEN_LAYER_2 = 256

NUM_EPOCHS = 462



def get_skeleton_batch_data(epoch_value):
    BASE_PATH = '/home/user/Documents/SignLangRecog/data/nturgbd_skeletons/nturgb+d_skeleton'
    
    

    for filename in os.listdir(BASE_PATH):
        if filename.endswith("skeleton"):    
        
        bad_files = np.load('./data/badfiles.npy')
        if filename.split('.')[0] in bad_files:
            print "bad file detected. Continuing."
            continue
        
        filepath = BASE_PATH +'/' + filename
        utils.readSkeletonFiles(filepath)

        if count >= 1:
            break
    

    data = np.zeros((BATCH_SIZE, MAX_FRAME_LEN, NUM_FEATURES))
    for i in range(BATCH_SIZE):
        
        data[i,:,:] =  readSkeletonFiles(filepath)



         





# define Model

# gru = GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', 
# 						recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, 
# 						bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
# 						dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, 
# 						unroll=False, reset_after=False)

# lstm = LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
# 						bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
# 						kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, 
# 						return_state=False, go_backwards=False, stateful=False, unroll=False)

model = Sequential()
model.add(Bidirectional(GRU(HIDDEN_LAYER_1, return_sequences=True), input_shape=INPUT_SIZE))
model.add(BatchNormalization(epsilon=1e-5))
model.add(Bidirectional(GRU(HIDDEN_LAYER_2, return_sequences=True)))
model.add(BatchNormalization(epsilon=1e-5))
model.add(Dropout(0.25))
model.add(TimeDistributed(Dense(OUTPUT_SIZE, activation='relu')))    #Fully Connected layer

rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.9)

model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc'])


for epoch in range(NUM_EPOCHS):

	X,y #= get_sequence(INPUT_SIZE)
	

	model.fit(X, y, epochs=1, batch_size=1, verbose=2)

model.save('bi-GRU.h5')

	# # evaluate LSTM
# X,y = get_sequence(INPUT_SIZE)
# yhat = model.predict_classes(X, verbose=0)
# for i in range(n_timesteps):
# 	print('Expected:', y[0, i], 'Predicted', yhat[0, i])	