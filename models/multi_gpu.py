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
from tensorflow.python.ops import gradients
import hparams
import math

def _build_encoder(inputs_vid):
    """Build an encoder."""
    bi_flag = False
    prev_layer = inputs_vid   # size = [BATCH_SIZE, NUM_FRAMES, IMG_HEIGHT. IMG_WIDTH, 2]

    in_filters = 2
    out_filters = 8
    conv1 = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=3, poolTrue=True, name_scope ='conv1')
    prev_layer = conv1
    
    in_filters = out_filters
    out_filters = 16
    conv2 = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=3, poolTrue=True, name_scope='conv2')
    prev_layer = conv2


    in_filters = out_filters
    out_filters = 32
    conv3a = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=3, poolTrue=False, name_scope='conv3a')
    prev_layer = conv3a


    in_filters = out_filters
    out_filters = 32
    conv3b = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=3, poolTrue=True, name_scope='conv3b')
    prev_layer = conv3b

    in_filters = out_filters
    out_filters = 64
    conv3c = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=5, poolTrue=True, name_scope='conv3c')
    prev_layer = conv3c
    prev_layer = tf.split(prev_layer, hparams.NUM_SEGMENTS, axis=1)       #split into segments in time dimensions
    output_fc = []
    flag = True 
    for i in range(len(prev_layer)):
        output_fc.append( confuncs.fully_connected(prev_layer[i], name_scope='FC_' + str(i)) )

    
    
    output_fc = tf.convert_to_tensor(output_fc)
    output_fc = tf.transpose(output_fc, [1, 0, 2])
    
    encoder_emb_inp = output_fc
    # encoder_emb_inp = tf.tile(tf.expand_dims(prev_layer, 2), [1, 1, NUM_LSTM_CELLS])

    # create 2 LSTMCells
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [hparams.NUM_LSTM_CELLS, hparams.NUM_LSTM_CELLS]]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=encoder_emb_inp,
                                       dtype=tf.float32)

    return encoder_outputs, encoder_state

    # if bi_flag:


    #     bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
    #         encoder_cell,
    #         encoder_cell,
    #         encoder_emb_inp,
    #         dtype=dtype,
    #         sequence_length=BATCH_SIZE,
    #         time_major=True,
    #         swap_memory=True)

    #     return tf.concat(bi_outputs, -1), bi_state

def _build_encoder_resnet(inputs):

    num_filters = hparams.resnet_num_filters
    kernel_size = hparams.resnet_kernel_size
    conv_stride = hparams.resnet_conv_stride
    training    = hparams.resnet_training_bool
    first_pool_size = hparams.resnet_first_pool_size
    first_pool_stride = hparams.resnet_first_pool_stride
    bottleneck = hparams.bottleneck_bool
    resnet_version = hparams.resnet_version
    pre_activation = hparams.pre_activation
    block_sizes = hparams.block_sizes
    block_strides = hparams.block_strides

    if resnet_version == 1:
        if bottleneck:
            block_fn = confuncs._bottleneck_block_v1
        else:
            block_fn = confuncs._building_block_v1     
    else:
        if bottleneck:
            block_fn = confuncs._bottleneck_block_v2
        else:
            block_fn = confuncs._building_block_v2        


    in_filters = 2
    out_filters = 8
    inputs = confuncs.conv_layer(inputs, in_filters, out_filters, Ksize=3, poolTrue=True, name_scope ='conv1')
    in_filters = out_filters
    out_filters = 8
    inputs = confuncs.conv_layer(inputs, in_filters, out_filters, Ksize=3, poolTrue=True, name_scope ='conv2')
    in_filters = out_filters
    out_filters = 16
    inputs = confuncs.conv_layer(inputs, in_filters, out_filters, Ksize=3, poolTrue=True, name_scope ='conv3')

    inputs = confuncs.conv3d_fixed_padding(inputs=inputs, filters=num_filters, kernel_size=kernel_size,strides=conv_stride)
    inputs = tf.identity(inputs, 'initial_conv')

    if resnet_version == 1:
        inputs = confuncs.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

    if first_pool_size:
        inputs = tf.layers.max_pooling3d(
            inputs=inputs, pool_size=first_pool_size,
            strides=first_pool_stride, padding='VALID')
        inputs = tf.identity(inputs, 'initial_max_pool')
    

    for i, num_blocks in enumerate(block_sizes):
        
        num_filters = num_filters * (2**i)
        inputs = confuncs.block_layer(
            inputs=inputs, filters=num_filters, bottleneck=bottleneck,
            block_fn=block_fn, blocks=num_blocks,
            strides=block_strides[i], training=training,
            name='block_layer{}'.format(i + 1))

    # Only apply the BN and ReLU for model that does pre_activation in each
    # building/bottleneck block, eg resnet V2.
    if pre_activation:
        inputs = confuncs.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)


    inputs = tf.split(inputs, hparams.NUM_SEGMENTS, axis=1)       #split into segments in time dimensions
    output_fc = []
    flag = True 
    for i in range(len(inputs)):
        output_fc.append( confuncs.fully_connected(inputs[i], name_scope='FC_' + str(i)) )

    output_fc = tf.convert_to_tensor(output_fc)
    output_fc = tf.transpose(output_fc, [1, 0, 2])
    
    encoder_emb_inp = output_fc

    # create 2 LSTMCells
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [hparams.NUM_LSTM_CELLS, hparams.NUM_LSTM_CELLS]]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=encoder_emb_inp,
                                       dtype=tf.float32)
    #import pdb; pdb.set_trace()
    return encoder_outputs, encoder_state



    
def _build_decoder(encoder_outputs, encoder_state, target_input, target_sequence_length, mode):
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
    batch_size = encoder_outputs.get_shape().as_list()[0]
    maximum_iterations = confuncs._get_infer_maximum_iterations(target_sequence_length)
   
#    projection_layer = layers_core.Dense(hparams.VOCAB_SIZE, use_bias=False, name="output_projection", reuse=True)                          # Define projection layer
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size, reuse=tf.AUTO_REUSE) for size in [hparams.NUM_LSTM_CELLS, hparams.NUM_LSTM_CELLS]]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

     
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

        logits = tf.layers.dense(inputs=outputs.rnn_output, units=hparams.VOCAB_SIZE)
        
        #logits = projection_layer(outputs.rnn_output)
        

    if mode == 'infer':
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)        
        start_tokens = tf.fill([batch_size], hparams.SOS)
        end_token = EOS
        length_penalty_weight = 0 
        beam_width = hparams.beam_width
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


def core_model(input_videos, target_inputs, target_sequence_length, mode, reuse=True):              # mode can be infer or train 
    with tf.variable_scope("core_model", reuse=reuse):
        # encoder_outputs, encoder_state = _build_encoder(input_videos)
        encoder_outputs, encoder_state = _build_encoder_resnet(input_videos)
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



def create_parallel_optimization(input_vids, target_inputs, target_sequence_length, target_outputs, optimizer, mode, controller="/cpu:0"):

    # returns operation to apply gradients and return loss
    
    # This function is defined below; it returns a list of device ids like
    # `['/gpu:0', '/gpu:1']`

    ndevices = len(get_available_gpus())
    with tf.name_scope("split_batches"):
        try:
            input_vids_list                 = tf.split(input_vids,              ndevices, axis=0)
            target_inputs_list              = tf.split(target_inputs,            ndevices, axis=0)
            target_outputs_list             = tf.split(target_outputs,           ndevices, axis=0)
            target_sequence_length_list     = tf.split(target_sequence_length,   ndevices, axis=0)
        except Exception as e:
            raise Exception("Batch size not a multiple of %d" % ndevices)
        
    # This list keeps track of the gradients per tower and the losses
    tower_grads = []
    losses = []
    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i in range(ndevices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.
            with tf.device("/gpu:%d"%i), tf.name_scope(name):
            # with tf.device(assign_to_device(id_, controller)), tf.name_scope(name):
                
                logits, sample_id, final_context_state = core_model(input_vids_list[i], target_inputs_list[i], target_sequence_length_list[i], mode, i!=0)
                # Compute loss and gradients, but don't apply them yet
                batch_size = input_vids_list[i].get_shape().as_list()[0]
                loss = confuncs._compute_loss(target_outputs_list[i], logits, batch_size)  
                
                with tf.name_scope("compute_gradients"):
                    # compute_gradients` returns a list of (gradient, variable) pairs
                    params = tf.trainable_variables()

                    for var in params:
                        tf.summary.histogram(var.name, var)
                    
                    grads = tf.gradients(loss, params, colocate_gradients_with_ops=True)    # optimizer.compute_gradients(loss)
                    clipped_grads, grad_norm_summary, grad_norm = confuncs.gradient_clip(grads, max_gradient_norm=hparams.max_gradient_norm)
                    grad_and_vars = zip(clipped_grads, params)
                    tower_grads.append(grad_and_vars)    
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
        tf.summary.scalar('loss', avg_loss)


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
        
        grads = [g for g, _ in grad_and_vars]
        #gradients._IndexedSlicesToTensor
        for i in range(len(grads)):
            if isinstance(grads[i], tf.IndexedSlices):
                grads[i] = tf.convert_to_tensor(grads[i])
        grad = tf.reduce_mean(grads, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

# def train(update_op, loss):

#     #build model 


#     sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     saver = tf.train.Saver()

#     for i in range(NUM_ITERATIONS):


#         annot_files = os.listdir(BASE_ANNOT_FILE)
#         for i in range(len(annot_files)):
#             annot_files[i] = os.path.basename(annot_files[i]).split('.')[0]


#         vid_files = os.listdir(BASE_VIDEO_FILE)

#         for i in range(len(vid_files)):
#             vid_files[i] = os.path.basename(vid_files[i]).split('.')[0]

#         filenames = np.intersect1d(vid_files, annot_files) 
         
#         np.random.shuffle(filenames)
              
#         epochs = len(filenames)/BATCH_SIZE
#         for i in range(epochs):
#             file_list = filenames[i:i+BATCH_SIZE]
            
#             target_batch_data = confuncs.getLabelbatch(file_list)
#             tgt_batch_input = target_batch_data[0]
#             tgt_batch_output = target_batch_data[1]
#             syncs = target_batch_data[2]
#             tgt_sequence_length =[]
#             for j in range(len(tgt_batch_input)):
#                 tgt_sequence_length.append(len(tgt_batch_input[j]))
            
#             batch_vids = confuncs.getOpflowBatch(file_list, syncs, hparams.prescaler)


#             # Fit training using batch data
#             _, loss_val, summary = sess.run(
#                 [update_op, loss, merged],
#                 feed_dict={
#                     input_vids: batch_vids, 
#                     target_inputs: tgt_batch_input,
#                     target_outputs: tgt_batch_output,
#                     target_sequence_length: tgt_sequence_length
#                 }
#             )
#             train_writer.add_summary(summary, i)

#             print (loss_val)                                # Print loss. Add inference model as well.     
    

#     save_path = saver.save(sess, "saved_model.ckpt")    #Save trained model
#     print("Model saved in path: %s" % save_path)


def parallel_training():

    
    input_vids = tf.placeholder(tf.float32, [hparams.BATCH_SIZE, hparams.NUM_FRAMES, hparams.IMG_HEIGHT, hparams.IMG_WIDTH, hparams.IN_CHANNELS])
    target_inputs = tf.placeholder(tf.int32, [hparams.BATCH_SIZE, hparams.MAX_SENT_LENGTH])
    target_outputs = tf.placeholder(tf.int32, [hparams.BATCH_SIZE, hparams.MAX_SENT_LENGTH])
    target_sequence_length = tf.placeholder(tf.int32, [hparams.BATCH_SIZE])
    
    mode = 'train'

    optimizer = tf.train.GradientDescentOptimizer(hparams.LR)
    update_op, loss = create_parallel_optimization(input_vids, target_inputs, target_sequence_length, target_outputs, optimizer, mode, controller="/cpu:0")

    #train(update_op, loss)


    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        )
    session_config.gpu_options.allow_growth = True
    
    sess = tf.InteractiveSession(config=session_config)

    initializer = tf.random_normal_initializer(0.0, hparams.init_std)
    with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
        # with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
        init = tf.global_variables_initializer()
        sess.run(init)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train',sess.graph)

        saver = tf.train.Saver()
        ndevices = len(get_available_gpus())


        for i in range(hparams.NUM_ITERATIONS):
            annot_files = os.listdir(hparams.BASE_ANNOT_FILE)
            for i in range(len(annot_files)):
                annot_files[i] = os.path.basename(annot_files[i]).split('.')[0]


            vid_files = os.listdir(hparams.BASE_VIDEO_FILE)

            for i in range(len(vid_files)):
                vid_files[i] = os.path.basename(vid_files[i]).split('.')[0]

            filenames = np.intersect1d(vid_files, annot_files) 
             
            np.random.shuffle(filenames)
            mini_batch_size = hparams.BATCH_SIZE/ndevices      
            
            epochs = len(filenames)/hparams.BATCH_SIZE
            for j in range(epochs):
                
                file_list = filenames[j: j + hparams.BATCH_SIZE]
                
                target_batch_data = confuncs.getLabelbatch(file_list)
                tgt_batch_input = target_batch_data[0]
                tgt_batch_output = target_batch_data[1]
                syncs = target_batch_data[2]
                tgt_sequence_length =[]
                for j in range(len(tgt_batch_input)):
                    tgt_sequence_length.append(len(tgt_batch_input[j]))
                
                batch_vids = confuncs.getOpflowBatch(file_list, syncs, hparams.prescaler)

                # Fit training using batch data
                _, loss_val, summary = sess.run(
                    [update_op, loss, merged],
                    feed_dict={
                        input_vids: batch_vids, 
                        target_inputs: tgt_batch_input,
                        target_outputs: tgt_batch_output,
                        target_sequence_length: tgt_sequence_length
                    }
                )
                print loss_val
                train_writer.add_summary(summary, j)

            save_path = saver.save(sess, "./saved_model.ckpt")    #Save trained model
            print("Model saved in path: %s" % save_path)




parallel_training()