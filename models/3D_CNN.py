import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.ops import embedding_ops
import acc_metric
import beamSearch
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.python.layers import core as layers_core

EMBEDDING_SIZE = 512
VOCAB_SIZE = 279
FC_SIZE = 1024
DTYPE = tf.float32
MAX_SENT_LENGTH = 16
BATCH_SIZE = 8
IMG_HEIGHT = 256
IMG_WIDTH = 256
IN_CHANNELS = 2
NUM_FRAMES = 858
NUM_LSTM_CELLS = 2048
NUM_ENCODER_LAYERS = 2
NUM_ITERATIONS = 1000
beam_width = 5 #10 
TIME_MAJOR = False
optimizer = "sgd"
SOS = '<s>'
EOS = '</s>'


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))             # function to intilaize weights for each layer

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))               # fucntion to intiliaze bias vector for each layer


def get_max_time(tensor):
    time_axis = 0 if TIME_MAJOR else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]


def _build_encoder(inputs_vid):
    """Build an encoder."""
    bi_flag = False
    prev_layer = inputs_vid   # size = [BATCH_SIZE, NUM_FRAMES, IMG_HEIGHT. IMG_WIDTH, 2]

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
    


    encoder_emb_inp = tf.tile(tf.expand_dims(local4_vid, 2), [1, 1, 128])

    # create 2 LSTMCells
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=encoder_emb_inp,
                                       dtype=tf.float32)


    return encoder_outputs, encoder_state

    if bi_flag:


        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            encoder_cell,
            encoder_cell,
            encoder_emb_inp,
            dtype=dtype,
            sequence_length=BATCH_SIZE,
            time_major=True,
            swap_memory=True)

        return tf.concat(bi_outputs, -1), bi_state


    
def _build_decoder(encoder_outputs, encoder_state, target_input):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      hparams: The Hyperparameters configurations.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
    """


    # maximum_iteration: The maximum decoding steps.
    #maximum_iterations = 
    mode = 'train'
    ## Decoder.
    output_layer = layers_core.Dense(VOCAB_SIZE, use_bias=False)                          # Define projection layer

    if mode == 'infer' and beam_width > 0:
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
    else:
        decoder_initial_state = encoder_state    
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_LSTM_CELLS)    
      ## Train or eval
    if mode != 'infer':
        # decoder_emp_inp: [max_time, batch_size, num_units]
        
        if TIME_MAJOR:
            target_input = tf.transpose(target_input)
        decoder_emb_inp = embeddings(target_input)
        sequence_length = tf.placeholder(tf.int32, [None])
        # Helper
        output_seq_len = 16
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp, sequence_length=[output_seq_len for _ in range(BATCH_SIZE)])

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell,
            helper,
            decoder_initial_state, output_layer=output_layer)


        

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, maximum_iterations=16, swap_memory=True)
        import pdb; pdb.set_trace()
        sample_id = outputs.sample_id

        # Note: there's a subtle difference here between train and inference.
        # We could have set output_layer when create my_decoder
        #   and shared more code between train and inference.
        # We chose to apply the output_layer to all timesteps for speed:
        #   10% improvements for small models & 20% for larger ones.
        # If memory is a concern, we should apply output_layer per timestep.
        logits = output_layer(outputs.rnn_output)
        

      # ## Inference
      # else:
      #   #beam_width = hparams.beam_width
      #   length_penalty_weight = 0#hparams.length_penalty_weight
      #   start_tokens = tf.fill([self.batch_size], tgt_sos_id)
      #   end_token = tgt_eos_id

      #   if beam_width > 0:
      #     my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
      #         cell=cell,
      #         embedding=self.embedding_decoder,
      #         start_tokens=start_tokens,
      #         end_token=end_token,
      #         initial_state=decoder_initial_state,
      #         beam_width=beam_width,
      #         output_layer=self.output_layer,
      #         length_penalty_weight=length_penalty_weight)
      #   else:
      #     # Helper
      #     sampling_temperature = hparams.sampling_temperature
      #     if sampling_temperature > 0.0:
      #       helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
      #           self.embedding_decoder, start_tokens, end_token,
      #           softmax_temperature=sampling_temperature,
      #           seed=hparams.random_seed)
      #     else:
      #       helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
      #           self.embedding_decoder, start_tokens, end_token)

      #     # Decoder
      #     my_decoder = tf.contrib.seq2seq.BasicDecoder(
      #         cell,
      #         helper,
      #         decoder_initial_state,
      #         output_layer=self.output_layer  # applied per timestep
      #     )

        # Dynamic decoding
        # outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
        #     my_decoder,
        #     maximum_iterations=maximum_iterations,
        #     output_time_major=self.time_major,
        #     swap_memory=True,
        #     scope=decoder_scope)

        # if beam_width > 0:
        #   logits = tf.no_op()
        #   sample_id = outputs.predicted_ids
        # else:
        #   logits = outputs.rnn_output
        #   sample_id = outputs.sample_id

    return logits, sample_id, final_context_state


    

def _compute_loss(target_output, logits):
    """Compute optimization loss."""
    if TIME_MAJOR:
        target_output = tf.transpose(target_output)
    max_time = get_max_time(target_output)
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_output, logits=logits)
    target_weights = tf.sequence_mask(
        BATCH_SIZE, max_time, dtype=logits.dtype)
    if TIME_MAJOR:
        target_weights = tf.transpose(target_weights)

    loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size)
    return loss



def readOpflow(dirpath, sync):
    data = np.zeros((MAX_VFRAME_LEN, IMG_HEIGHT, IMG_WIDTH, 2))

    files = os.listdir(dirpath)
    files = np.sort(files)
    num_frames = len(files) 
    x_files = files[0:num_frames/2]
    y_files = files[num_frames/2:num_frames]
    for i in range(num_frames/2):
        # print i
        data[i,:,:,0] = np.array(Image.open(dirpath + '/' + x_files[i])) 
        data[i,:,:,1] = np.array(Image.open(dirpath + '/' + y_files[i]))
    data = data[sync:,:,:,:]
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
    for x in l:
        idx = np.where(data == x)[0][0]
        result.append(idx)
    return result


def readlabels(filepath):
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
    return target_input, target_output, frame_start, frame_end, sync


def getOpflowBatch(file_list, syncs):
    data = np.zeros((BATCH_SIZE, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 2))
    for i in range(len(file_list)):
        data[i,:,:,:,:] = readOpflow(dirpath, sync[0])
    return data    


def getLabelbatch(file_list):
    target_inputs = []
    target_output = []
    syncs = []

    for i in range(len(file_list)):

        target_input, target_output, frame_start, frame_end, sync = readlabels(filepath)
        target_inputs.append(target_input)
        target_outputs.append(target_output)
        syncs.append(sync)
    return target_inputs, target_outputs, syncs    

max_gradient_norm = 1  #5

def gradient_clip(gradients, max_gradient_norm):
  """Clipping gradients of a model."""
  clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
  gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
  gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

  return clipped_gradients, gradient_norm_summary, gradient_norm



# CHANGE THE BATCH SIZE
inputs_vid = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 2])
target_inputs = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_SENT_LENGTH])
target_outputs = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_SENT_LENGTH])


encoder_outputs, encoder_state = _build_encoder(inputs_vid)

logits, sample_id, final_context_state = _build_decoder(encoder_outputs, encoder_state, target_inputs)


# Gradients
params = tf.trainable_variables()
train_loss = _compute_loss(target_outputs, logits)
gradients = tf.gradients(train_loss, params, colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

clipped_grads, grad_norm_summary, grad_norm = gradient_clip(gradients, max_gradient_norm=max_gradient_norm)


#Optimizer
if optimizer == "sgd":
    opt = tf.train.GradientDescentOptimizer(learning_rate)

elif optimizer == "adam":
    opt = tf.train.AdamOptimizer(learning_rate)


pred_sequences = beam_search_decoder(logits, k=5)
score_list = []
for i in range(len(pred_sequences)):
    reference = target_outputs

    score = sentence_bleu(reference, pred_sequences[i])
    score_list.append(score)

accuracy = score_list


# for i in range(5):
#     pred_sequences[i]    


#accuracy = acc_metric.wer(logits, )
# beamSearch.ctc_beamsearch(dist, '-ab', k=100)

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=2))
init = tf.global_variables_initializer()
sess.run(init)

step = 1

for i in range(NUM_ITERATIONS):

    BASE_VIDEO_FILE = '/home/psankhe/sign-lang-recog/data/opflow_xy'
    BASE_ANNOT_FILE = '/home/data/00_Annotations_20180427'

    annot_files = os.listdir(BASE_ANNOT_FILE)
    for i in range(len(annot_files)):
        annot_files[i] = os.path.basename(annot_files[i]).split('.')[0]


    vid_files = os.listdir(BASE_VIDEO_FILE)

    for i in range(len(vid_files)):
        vid_files[i] = os.path.basename(vid_files[i]).split('.')[0]

    filenames = np.intersect1d(vid_files, annot_files)        

    filenames = np.random.shuffle()
    epochs = len(filenames)/BATCH_SIZE
    for i in range(epochs):
        file_list = filenames[i:i+BATCH_SIZE]
        target_batch_data = getLabelbatch(file_list)
        tgt_batch_input = target_batch_data[0]
        tgt_batch_input = target_batch_data[1]
        syncs = target_batch_data[2]
        batch_vids = getOpflowBatch(file_list, syncs)


        # Fit training using batch data
        _, loss = sess.run(
            [optimizer, train_loss],
            feed_dict={
                inputs_vid: inputs_vid, 
                target_input: tgt_input,
                target_output: tgt_output
            }
        )
    print (loss)
