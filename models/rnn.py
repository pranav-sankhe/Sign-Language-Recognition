# All Includes

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics

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


# # Those are separate normalised input features for the neural network
# INPUT_SIGNAL_TYPES = [
#     "body_acc_x_",
#     "body_acc_y_",
#     "body_acc_z_",
#     "body_gyro_x_",
#     "body_gyro_y_",
#     "body_gyro_z_",
#     "total_acc_x_",
#     "total_acc_y_",
#     "total_acc_z_"
# ]

# # Output classes to learn how to classify
# LABELS = [
#     "WALKING", 
#     "WALKING_UPSTAIRS", 
#     "WALKING_DOWNSTAIRS", 
#     "SITTING", 
#     "STANDING", 
#     "LAYING"
# ]


# TRAIN = "train/"
# TEST = "test/"


# # Load "X" (the neural network's training and testing inputs)


# def load_X(X_signals_paths):
#     X_signals = []
    
#     for signal_type_path in X_signals_paths:
#         file = open(signal_type_path, 'r')
#         # Read dataset from disk, dealing with text files' syntax
#         X_signals.append(
#             [np.array(serie, dtype=np.float32) for serie in [
#                 row.replace('  ', ' ').strip().split(' ') for row in file
#             ]]
#         )
#         file.close()
    
#     return np.transpose(np.array(X_signals), (1, 2, 0))

# X_train_signals_paths = [
#     DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
# ]
# X_test_signals_paths = [
#     DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
# ]

# X_train = load_X(X_train_signals_paths)
# X_test = load_X(X_test_signals_paths)
data = np.load('data.npy')
X_train = data[0:40000]
X_test =  data[40000:]



# Load "y" (the neural network's training and testing outputs)

# def load_y(y_path):
#     file = open(y_path, 'r')
#     # Read dataset from disk, dealing with text file's syntax
#     y_ = np.array(
#         [elem for elem in [
#             row.replace('  ', ' ').strip().split(' ') for row in file
#         ]], 
#         dtype=np.int32
#     )
#     file.close()
    
#     # Substract 1 to each output class for friendly 0-based indexing 
#     return y_ - 1
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
# y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
# y_test_path = DATASET_PATH + TEST + "y_test.txt"
labels = np.load('labels.npy')
y_train = labels[0:40000]
y_test = labels[40000:]





training_data_count = len(X_train)  # 40000 training series
test_data_count = len(X_test)  # 6200 testing series
n_steps = len(X_train[0])  # 429 timesteps per series
n_input = len(X_train[0][0])  # 45 input parameters per timestep

print training_data_count, n_steps, n_input

# LSTM Neural Network's internal structure

n_hidden = 256 # Hidden layer num of features
n_classes = len(np.unique(labels)) # Total classes (should go up, or should go down)


# Training 

learning_rate = 0.0001
lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset
batch_size = 100
display_iter = 30000  # To show test set accuracy during training


# Some debugging info

print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_train.shape, y_train.shape, np.mean(X_train), np.std(X_train))
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")


def BiLSTM(_X, _weights, _biases):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters. 
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network. 
    # Note, some code of this notebook is inspired from an slightly different 

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) 
    # new shape: (n_steps*batch_size, n_input)
    
    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])          # y = x * W + b  
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 
    # new shape: n_steps * (batch_size, n_hidden)


    #First BLSTM
    cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-dropout)
    (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs=_X,
                                             dtype=tf.float32,scope='BLSTM_1')
    outputs = tf.concat([forward_output, backward_output], axis=2)

    #Second BLSTM using the output of previous layer as an input.
    cell2 = tf.nn.rnn_cell.GRUCell(n_hidden)
    cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=1-dropout)
    (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(cell2, cell2, inputs=outputs,
                                        sequence_length=lengths, dtype=tf.float32,scope='BLSTM_2')
    outputs = tf.concat([forward_output, backward_output], axis=2)

    lstm_last_output = outputs[-1]
    
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


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
l.compile(loss='categorical_crossentropy', optimizer=rmsprop)
print("Train...")


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

X_train = np.load('data.npy')
Y_train = np.load('labels.npy')

print type(X_train)
print type(Y_train)
print X_train.shape
print Y_train.shape
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, verbose=2,callbacks=[plot_losses])

    return batch_s
model.save('bi-GRU.h5')

def one_hot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS



# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'BN':     tf.Variable(tf.random_normal([n_input, n_hidden])),  #BatchNormalization weights
    'full_connected': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = BiLSTM(x, weights, biases)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs =         extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))

    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs, 
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        
        # To not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

print("Optimization Finished!")

# Accuracy for test data

one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))