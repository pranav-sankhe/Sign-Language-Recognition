# All Includes
import os
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics




# create a sequence classification instance



# # Useful Constants

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

# y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
# y_test_path = DATASET_PATH + TEST + "y_test.txt"
labels = np.load('labels.npy')
y_train = labels[0:40000]
y_test =  labels[40000:]


# Input Data 

training_data_count = len(X_train)  # 40000 training series
test_data_count = len(X_test)  # 6200 testing series
n_steps = len(X_train[0])  # 429 timesteps per series
n_input = len(X_train[0][0])  # 45 input parameters per timestep

print training_data_count, n_steps, n_input

# LSTM Neural Network's internal structure

n_hidden = 256 # Hidden layer num of features
n_classes = len(np.unique(labels)) # Total classes (should go up, or should go down)

num_fully_connected = 1024

# Training 

learning_rate = 0.001
lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset
batch_size = 128
display_iter = 30000  # To show test set accuracy during training


# Some debugging info

print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_train.shape, y_train.shape, np.mean(X_train), np.std(X_train))
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")


def BiLSTM(_X, _weights, _biases):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters. 
    
    #_X = tf.unstack(_X, n_steps, 1)
    
    # # (NOTE: This step could be greatly optimised by shaping the dataset once
    # # input shape: (batch_size, n_steps, n_input)
    # _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    # _X = tf.reshape(_X, [-1, n_input]) 
    # new shape: (n_steps*batch_size, n_input)
    #print "INPUT", _X 
    # Linear activation
    # _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])          # y = x * W + b  
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    # _X = tf.split(_X, n_steps, 0)

    # # new shape: n_steps * (batch_size, n_hidden)
    dropout = 0.25
    #First BLSTM
    cell1 = tf.contrib.rnn.GRUCell(n_hidden)
    cell1 = tf.contrib.rnn.DropoutWrapper(cell1, output_keep_prob=1)
    cell2 = tf.contrib.rnn.GRUCell(n_hidden)
    cell2 = tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob=1)
    cell3 = tf.contrib.rnn.GRUCell(n_hidden)
    cell3 = tf.contrib.rnn.DropoutWrapper(cell3, output_keep_prob=1)
    cells_fw = [cell1, cell2, cell3]
    cells_bw = [cell1, cell2, cell3]

    output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
    cells_fw,
    cells_bw,
    _X,
    dtype=tf.float32,
    sequence_length=np.random.rand(batch_size))

    
    epsilon = 1e-3

    lstm_last_output = output[:,-1,:]
    batch_mean1, batch_var1 = tf.nn.moments(lstm_last_output,[0])

    BN_layer =  tf.nn.batch_normalization(
    lstm_last_output,
    batch_mean1,
    batch_var1,
    _biases['beta'],
    _weights['scale'],
    epsilon)
 
    BN_layer  = tf.layers.dropout(
        BN_layer,
        rate=0.25,
        noise_shape=None,
        seed=None,
        training=False,
        name=None
    )


    fc = tf.layers.dense(
        BN_layer,
        num_fully_connected,
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None
    )


    # Linear activation
    return tf.matmul(fc, _weights['out']) + _biases['out']



def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)
    
    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


# def one_hot(y_):
#     # Function to encode output labels from number indexes 
#     # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
#     y_ = y_.reshape(len(y_))
#     n_values = 49
#     return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def one_hot(y, num_classes):
    encoded =  np.zeros((len(y), num_classes))
    
    for i in range(encoded.shape[0]):
        j =  int(y[i])
        encoded[i][j-1] = 1 
    return encoded    




# Graph input/output
x = tf.placeholder(tf.float32, [batch_size, n_steps, n_input])
y = tf.placeholder(tf.float32, [batch_size, n_classes])

# Graph weights
weights = {
    'scale':     tf.Variable(tf.ones([batch_size, 2*n_hidden])),  #BatchNormalization weights
    'out': tf.Variable(tf.random_normal([num_fully_connected, n_classes], mean=1.0))
}
biases = {
    'beta':     tf.Variable(tf.zeros([2*n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = BiLSTM(x, weights, biases)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)) + l2 # Softmax loss
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=2))
init = tf.global_variables_initializer()
sess.run(init)

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs =         extract_batch_size(X_train, step, batch_size)
    
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size), n_classes)

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
                x: extract_batch_size(X_test, step, batch_size),
                y: one_hot(extract_batch_size(y_test, step, batch_size), n_classes)
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
        x: extract_batch_size(X_test, step, batch_size),
        y: one_hot(extract_batch_size(y_test, step, batch_size), n_classes)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))