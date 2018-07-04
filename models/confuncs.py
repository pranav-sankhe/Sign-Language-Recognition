import tensorflow
import numpy as np 
import pandas as pd
import os









def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))             # function to intilaize weights for each layer

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))               # fucntion to intiliaze bias vector for each layer


def get_max_time(tensor):
    time_axis = 0 if TIME_MAJOR else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

def conv_layer(prev_layer, in_filters, out_filters, Ksize, poolTrue, name_scope)

    # in_filters = 2
    with tf.variable_scope(name_scope) as scope:                                                          # name of the block  
        # out_filters = 8                                                                               # number of input channels for conv1     
        kernel = _weight_variable('weights', [Ksize, Ksize, Ksize, in_filters, out_filters])                       # (kernels = filters as defined in TF doc). kernel size = 5 (5*5*5) 
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')                       # stride = 1          
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)                                                            # define biases for conv1 
        conv1 = tf.nn.relu(bias, name=scope.name)                                                      # define the activation for conv1 
        prev_layer = conv1                                                                              
        # in_filters = out_filters                                    
    if poolTrue:    
        pool1 = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')        
        norm1 = pool1  # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')
        prev_layer = norm1

    return prev_layer    


def fully_connected(prev_layer,Fc_size, name_scope)
    with tf.variable_scope(name_scope) as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, Fc_size])
        biases = _bias_variable('biases', [Fc_size])
        output = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)
    return output    



def readOpflow(dirpath, sync, prescaler):
    dirpath = BASE_VIDEO_FILE + '/' + dirpath
    data = np.zeros((NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 2))
    files = os.listdir(dirpath)
    files = np.sort(files)
    num_frames = len(files) 
    x_files = files[0:num_frames/2]
    y_files = files[num_frames/2:num_frames]
    count = 0 

    for i in range(NUM_FRAMES):
        if i < num_frames/2:
            if i%2 == 0:
                data[i,:,:,0] = np.array(Image.open(dirpath + '/' + x_files[i])) 
                data[i,:,:,1] = np.array(Image.open(dirpath + '/' + y_files[i]))
    for i in range(sync/2):
        data[i,:,:,:] = np.zeros((IMG_HEIGHT, IMG_WIDTH, 2))            
    return data

def embeddings(decoder_inputs):
    with tf.variable_scope('embedding_decoder'):
        embedding_decoder = tf.get_variable(
            "embedding_decoder", [VOCAB_SIZE, EMBEDDING_SIZE])

    decoder_emb_inp = embedding_ops.embedding_lookup(embedding_decoder, decoder_inputs)
    return decoder_emb_inp

def word_to_int(l):
    result = []
    filepath = '/home/data/01_Label/label.csv'
    data = pd.read_csv(filepath, header=None, names=['0','1','2'])
    data = data.values
    data = data[:,0]
    data = data[:-2]
    data = np.append(data,'<s>')
    data = np.append(data,'</s>')
    
    for x in l:
        idx = np.where(data == x)[0][0]
        result.append(idx)
    return result


def readlabels(filepath):
    filepath = BASE_ANNOT_FILE + '/' + filepath + '.csv'
    data = pd.read_csv(filepath)
    data = data.values
    sync = data[0][1] 
    data_shape = data.shape 
    target_output = []
    target_input = []
    frame_start = []
    frame_end = []
 
    target_input.append(SOS)
    for i in range(data_shape[0]/3):
        j = 2 + i*3
        frame_start.append(data[j-1][1])
        target_input.append(data[j][0])
        target_output.append(data[j][0])
        frame_end.append(data[j+1][2])
    target_output.append(EOS)

    target_input = word_to_int(target_input)
    target_output = word_to_int(target_output)
    while(len(target_input) < 16):
        target_input.append(278)
    while(len(target_output) < 16):
        target_output.append(278)
    return target_input, target_output, frame_start, frame_end, sync


def getOpflowBatch(file_list, syncs):
    data = np.zeros((BATCH_SIZE, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 2))
    for i in range(len(file_list)):
        data[i,:,:,:,:] = readOpflow(file_list[i], syncs[0], prescaler)
    return data    


def getLabelbatch(file_list):
    target_inputs = []
    target_outputs = []
    syncs = []

    for i in range(len(file_list)):

        target_input, target_output, frame_start, frame_end, sync = readlabels(file_list[i])
        target_inputs.append(target_input)
        target_outputs.append(target_output)
        syncs.append(sync)
    return target_inputs, target_outputs, syncs      


