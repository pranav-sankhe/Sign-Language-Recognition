import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.ops import embedding_ops
import acc_metric
import beamSearch
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.python.layers import core as layers_core
import os
import random
from PIL import Image
import confuncs

NUM_SEGEMENTS = 9
EMBEDDING_SIZE = 256
VOCAB_SIZE = 279
FC_SIZE = 256
DTYPE = tf.float32
MAX_SENT_LENGTH = 16
BATCH_SIZE = 8
IMG_HEIGHT = 256
IMG_WIDTH = 256
IN_CHANNELS = 2
prescaler = 2
NUM_FRAMES = 858/prescaler
NUM_LSTM_CELLS = 256
NUM_ENCODER_LAYERS = 2
NUM_ITERATIONS = 1000
beam_width = 5 #10 
TIME_MAJOR = False
optimizer = "sgd"
SOS = '<s>'
EOS = '</s>'
LR = 0.0001



BASE_VIDEO_FILE = '/home/psankhe/sign-lang-recog/data/opflow_xy'
BASE_ANNOT_FILE = '/home/psankhe/sign-lang-recog/annotations'

def _build_encoder(inputs_vid):
    """Build an encoder."""
    bi_flag = False
    prev_layer = inputs_vid   # size = [BATCH_SIZE, NUM_FRAMES, IMG_HEIGHT. IMG_WIDTH, 2]

    in_filters = 2
    out_filters = 8
    conv1 = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=5, poolTrue=True, name_scope ='conv1')
    prev_layer = conv1
    
    in_filters = out_filters
    out_filters = 16
    conv2 = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=5, poolTrue=True, name_scope='conv2')
    prev_layer = conv2


    in_filters = out_filters
    out_filters = 32
    conv3a = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=5, poolTrue=False, name_scope='conv3a')
    prev_layer = conv3a


    in_filters = out_filters
    out_filters = 64
    conv3b = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=5, poolTrue=False, name_scope='conv3b')
    prev_layer = conv3b

    in_filters = out_filters
    out_filters = 128
    conv3c = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=5, poolTrue=True, name_scope='conv3c')
    prev_layer = conv3c

    prev_layer = tf.split(prev_layer, NUM_SEGEMENTS, axis=1)       #split into segments in time dimensions
    output_fc = []
    flag = True 
    for i in range(len(prev_layer)):
        output_fc.append( confuncs.fully_connected(prev_layer[i], Fc_size=FC_SIZE, name_scope='FC_' + str(i)) )

    
    
    output_fc = tf.convert_to_tensor(output_fc)
    output_fc = tf.transpose(output_fc, [1, 0, 2])
    
    encoder_emb_inp = output_fc
    # encoder_emb_inp = tf.tile(tf.expand_dims(prev_layer, 2), [1, 1, NUM_LSTM_CELLS])

    # create 2 LSTMCells
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [NUM_LSTM_CELLS, NUM_LSTM_CELLS]]

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


    
def _build_decoder(encoder_outputs, encoder_state, target_input, target_sequence_length,mode):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      hparams: The Hyperparameters configurations.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
    """


    ## Decoder.
    output_layer = layers_core.Dense(VOCAB_SIZE, use_bias=False)                          # Define projection layer

    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [NUM_LSTM_CELLS, NUM_LSTM_CELLS]]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    maximum_iterations = confuncs._get_infer_maximum_iterations(target_sequence_length)
    if mode == 'train':
        # decoder_emp_inp: [max_time, batch_size, num_units]
        decoder_initial_state = encoder_state
        decoder_emb_inp = confuncs.embeddings(target_input)
        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp, sequence_length=target_sequence_length)

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            multi_rnn_cell,
            helper,
            decoder_initial_state)

        
        
        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, maximum_iterations=maximum_iterations, swap_memory=True)

        sample_id = outputs.sample_id
        
        logits = output_layer(outputs.rnn_output)
        

    if mode == 'infer':
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)        
        start_tokens = tf.fill([self.batch_size], SOS)
        end_token = EOS
        length_penalty_weight = 0 
        beam_width = 5
        my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                                                          cell=cell,
                                                          embedding=confuncs.embedding_for_beamSearch,
                                                          start_tokens=start_tokens,
                                                          end_token=end_token,
                                                          initial_state=decoder_initial_state,
                                                          beam_width=beam_width,
                                                          output_layer=output_layer,
                                                          length_penalty_weight=length_penalty_weight)

        
        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            swap_memory=True)

        if beam_width > 0:
          logits = tf.no_op()
          sample_id = outputs.predicted_ids
        else:
          logits = outputs.rnn_output
          sample_id = outputs.sample_id


    return logits, sample_id, final_context_state


def core_model(input_videos, target_inputs, target_sequence_length, mode):              # mode can be infer or train 
    encoder_outputs, encoder_state = _build_encoder(input_videos)
    logits, sample_id, final_context_state = _build_decoder(encoder_outputs, encoder_state, target_inputs, target_sequence_length, mode)
    return logits, sample_id, final_context_state
        


PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]
    

def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign



def create_parallel_optimization(logits, target_outputs, optimizer, controller="/cpu:0"):               # returns operation to apply gradients and return loss
    
    # This function is defined below; it returns a list of device ids like
    # `['/gpu:0', '/gpu:1']`
    devices = get_available_gpus()
        
    # This list keeps track of the gradients per tower and the losses
    tower_grads = []
    losses = []
    
    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id_ in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.
            with tf.device(assign_to_device(id_, controller)), tf.name_scope(name):
                
                # Compute loss and gradients, but don't apply them yet
                loss = confuncs._compute_loss(target_outputs, logits)  # model_fn(input_fn)
                
                with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    params = tf.trainable_variables()
                    max_gradient_norm = 1
                    grads = tf.gradients(loss, params, colocate_gradients_with_ops=True)
                    clipped_grads, grad_norm_summary, grad_norm = confuncs.gradient_clip(grads, max_gradient_norm=max_gradient_norm)
                    tower_grads.append(clipped_grads)
                    
                losses.append(loss)
            
            # After the first iteration, we want to reuse the variables.
            outer_scope.reuse_variables()
                
    # Apply the gradients on the controlling device
    with tf.name_scope("apply_gradients"), tf.device(controller):
        # Note that what we are doing here mathematically is equivalent to returning the
        # average loss over the towers and compute the gradients relative to that.
        # Unfortunately, this would place all gradient-computations on one device, which is
        # why we had to compute the gradients above per tower and need to average them here.
        
        # This function is defined below; it takes the list of (gradient, variable) lists
        # and turns it into a single (gradient, variables) list.
        gradients = average_gradients(tower_grads)
        global_step = tf.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
        avg_loss = tf.reduce_mean(losses)

    return apply_gradient_op, avg_loss


def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']



def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
        over the devices. The inner list ranges over the different variables.
    Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        import pdb; pdb.set_trace()
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads



def train(update_op, loss):

    #build model 


    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    for i in range(NUM_ITERATIONS):


        annot_files = os.listdir(BASE_ANNOT_FILE)
        for i in range(len(annot_files)):
            annot_files[i] = os.path.basename(annot_files[i]).split('.')[0]


        vid_files = os.listdir(BASE_VIDEO_FILE)

        for i in range(len(vid_files)):
            vid_files[i] = os.path.basename(vid_files[i]).split('.')[0]

        filenames = np.intersect1d(vid_files, annot_files) 
         
        np.random.shuffle(filenames)
              
        epochs = len(filenames)/BATCH_SIZE
        for i in range(epochs):
            file_list = filenames[i:i+BATCH_SIZE]
            
            target_batch_data = confuncs.getLabelbatch(file_list)
            tgt_batch_input = target_batch_data[0]
            tgt_batch_output = target_batch_data[1]
            syncs = target_batch_data[2]
            tgt_sequence_length =[]
            for j in range(len(tgt_batch_input)):
                tgt_sequence_length.append(len(tgt_batch_input[i]))
            len(tgt_batch_input[0] +1)
            batch_vids = confuncs.getOpflowBatch(file_list, syncs)


            # Fit training using batch data
            _, loss_val = sess.run(
                [update_op, loss],
                feed_dict={
                    inputs_vid: batch_vids, 
                    target_inputs: tgt_batch_input,
                    target_outputs: tgt_batch_output,
                    target_sequence_length: tgt_sequence_length
                }
            )
            print (loss_val)                                # Print loss. Add inference model as well.     
    

    save_path = saver.save(sess, "saved_model.ckpt")    #Save trained model
    print("Model saved in path: %s" % save_path)






def parallel_training():

    
    input_vids = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 2])
    target_inputs = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_SENT_LENGTH])
    target_outputs = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_SENT_LENGTH])
    target_sequence_length = tf.placeholder(tf.int32, [BATCH_SIZE])
    
    mode = 'train'
    logits, sample_id, final_context_state = core_model(input_vids, target_inputs, target_sequence_length, mode)

    optimizer = tf.train.GradientDescentOptimizer(LR)
    update_op, loss = create_parallel_optimization(logits, target_outputs, optimizer, controller="/cpu:0")

    train(update_op, loss)



tf.reset_default_graph()
parallel_training()