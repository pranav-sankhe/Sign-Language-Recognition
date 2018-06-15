import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.ops import embedding_ops
FC_SIZE = 1024
DTYPE = tf.float32

BATCH_SIZE = 1
NUM_EPOCHS = 100
IMG_HEIGHT = 220
IMG_WIDTH = 220 
IN_CHANNELS = 1
NUM_FRAMES = 858
NUM_LSTMCells = 2048


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))             # function to intilaize weights for each layer

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))               # fucntion to intiliaze bias vector for each layer


def model(inputs_vid):       
    prev_layer = inputs_vid

    in_filters = 2
    with tf.variable_scope('conv1_vid') as scope:                                                          # name of the block  
        out_filters = 16                                                                               # number of input channels for conv1     
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])                       # (kernels = filters as defined in TF doc). kernel size = 5 (5*5*5) 
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')                       # stride = 1          
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)                                                            # define biases for conv1 
        conv1 = tf.nn.relu(bias, name=scope.name)                                                      # define the activation for conv1 

        prev_layer = conv1                                                                              
        in_filters = out_filters                                    

    pool1 = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')        
    norm1 = pool1  # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')

    prev_layer = norm1

    with tf.variable_scope('conv2_vids') as scope:
        out_filters = 32
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

        prev_layer = conv2
        in_filters = out_filters

    # normalize prev_layer here
    prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv3_1_vids') as scope:
        out_filters = 64
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias, name=scope.name)
        in_filters = out_filters

    with tf.variable_scope('conv3_2_vids') as scope:
        out_filters = 64
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias, name=scope.name)
        in_filters = out_filters

    with tf.variable_scope('conv3_3_vids') as scope:
        out_filters = 32
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias, name=scope.name)
        in_filters = out_filters

    # normalize prev_layer here
    prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


    with tf.variable_scope('local3_vids') as scope:                                     # FULLY CONNECTED LAYER 
        dim = np.prod(prev_layer.get_shape().as_list()[1:]) 
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])
        local3 = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)

    prev_layer = local3

    with tf.variable_scope('local4_vids') as scope:                                      # ANOTHER FULLLY CONNECTED LAYER     
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])
        local4_vid = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)


    local4_vid = tf.reshape(local4_vid, [FC_SIZE, BATCH_SIZE, 1])

    # Build RNN cell
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(200)
    init_state = encoder_cell.zero_state(BATCH_SIZE, tf.float32)
    encoder_emb_inp = local4_vid
    # Run Dynamic RNN
    #   encoder_outputs: [max_time, batch_size, num_units]
    #   encoder_state: [batch_size, num_units]
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_emb_inp, initial_state=init_state,
         time_major=True)

    print encoder_outputs
    print encoder_state
    

# data = np.random.rand(1,858,256,256,2)
# data = data.astype(np.float32)
# model(data)        


def word_to_int():
    filepath = 'label.csv'
    data = pd.read_csv(filepath, header=None, names=['0','1','2'])
    data = data.values
    data = data[:,(1,2)]

EMBEDDING_SIZE = 512
VOCAB_SIZE = 279

def embeddings(decoder_inputs):
    # Embedding
    # variable_scope = tf.variable_scope('embedding_encoder')
    with tf.variable_scope('embedding_decoder'):
        embedding_decoder = tf.get_variable(
            "embedding_decoder", [VOCAB_SIZE, EMBEDDING_SIZE])
    # Look up embedding:
    #   encoder_inputs: [max_time, batch_size]
    #   encoder_emb_inp: [max_time, batch_size, embedding_size]
    decoder_emb_inp = embedding_ops.embedding_lookup(
        embedding_decoder, decoder_inputs)
    return decoder_emb_inp



# Build RNN cell
num_units = 1024
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

# Helper
helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_emb_inp, decoder_lengths, time_major=True)
# Decoder
decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, encoder_state,
    output_layer=projection_layer)
# Dynamic decoding
outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations,
                                                output_time_major=self.time_major,
                                                swap_memory=True)
logits = outputs.rnn_output



projection_layer = layers_core.Dense(
    tgt_vocab_size, use_bias=False)

crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=decoder_outputs, logits=logits)
train_loss = (tf.reduce_sum(crossent * target_weights) /
    batch_size)



# Calculate and clip gradients
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(
    gradients, max_gradient_norm)


# Optimization
optimizer = tf.train.AdamOptimizer(learning_rate)
update_step = optimizer.apply_gradients(
    zip(clipped_gradients, params))

