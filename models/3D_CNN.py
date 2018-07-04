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
    prev_layer = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=5, poolTrue=True, 'conv1')

    in_filters = 2
    out_filters = 8
    prev_layer = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=5, poolTrue=True, 'conv2')

    in_filters = 2
    out_filters = 8
    prev_layer = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=5, poolTrue=False, 'conv3a')

    in_filters = 2
    out_filters = 8
    prev_layer = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=5, poolTrue=False, 'conv3b')

    in_filters = 2
    out_filters = 8
    prev_layer = confuncs.conv_layer(prev_layer, in_filters, out_filters, Ksize=5, poolTrue=True, 'conv3c')

    prev_layer = confuncs.fully_connected(prev_layer, Fc_size=FC_SIZE, 'FC_1')
    prev_layer = confuncs.fully_connected(prev_layer, Fc_size=FC_SIZE, 'FC_2')


    encoder_emb_inp = tf.tile(tf.expand_dims(prev_layer, 2), [1, 1, NUM_LSTM_CELLS])

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


    
def _build_decoder(encoder_outputs, encoder_state, target_input, target_sequence_length):
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
    # create 2 LSTMCells
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [256, 256]]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

 


      ## Train or eval
    if mode != 'infer':
        # decoder_emp_inp: [max_time, batch_size, num_units]
        
        if TIME_MAJOR:
            target_input = tf.transpose(target_input)
        decoder_emb_inp = embeddings(target_input)
        sequence_length = tf.placeholder(tf.int32, [None])
        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp, sequence_length=target_sequence_length)

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            multi_rnn_cell,
            helper,
            decoder_initial_state)


        
        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, swap_memory=True)

        sample_id = outputs.sample_id
        
        # Note: there's a subtle difference here between train and inference.
        # We could have set output_layer when create my_decoder
        #   and shared more code between train and inference.
        # We chose to apply the output_layer to all timesteps for speed:
        #   10% improvements for small models & 20% for larger ones.
        # If memory is a concern, we should apply output_layer per timestep.
        
        logits = output_layer(outputs.rnn_output)
        # import pdb; pdb.set_trace()

        

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


def core_model(input_videos, target_inputs=None, mode):              # mode can be infer or train 
        

    

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

    loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(BATCH_SIZE)
    return loss





max_gradient_norm = 1  #5

def gradient_clip(gradients, max_gradient_norm):
  """Clipping gradients of a model."""
  clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
  gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
  gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

  return clipped_gradients, gradient_norm_summary, gradient_norm




inputs_vid = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 2])
target_inputs = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_SENT_LENGTH])
target_outputs = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_SENT_LENGTH])
target_sequence_length = tf.placeholder(tf.int32, BATCH_SIZE)


encoder_outputs, encoder_state = _build_encoder(inputs_vid)

logits, sample_id, final_context_state = _build_decoder(encoder_outputs, encoder_state, target_inputs, target_sequence_length)


# Gradients
params = tf.trainable_variables()
train_loss = _compute_loss(target_outputs, logits)
gradients = tf.gradients(train_loss, params, colocate_gradients_with_ops=True)

clipped_grads, grad_norm_summary, grad_norm = gradient_clip(gradients, max_gradient_norm=max_gradient_norm)


#Optimizer
# if optimizer == "sgd":
optimizer = tf.train.GradientDescentOptimizer(LR).minimize(train_loss)

# elif optimizer == "adam":
#     opt = tf.train.AdamOptimizer(LR)

# import pdb; pdb.set_trace()

# pred_sequences = beam_search_decoder(logits, k=5)
# score_list = []
# for i in range(len(pred_sequences)):
#     reference = target_outputs

#     score = sentence_bleu(reference, pred_sequences[i])
#     score_list.append(score)

# accuracy = score_list


# for i in range(5):
#     pred_sequences[i]    


#accuracy = acc_metric.wer(logits, )
# beamSearch.ctc_beamsearch(dist, '-ab', k=100)

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
step = 1

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
        tgt_sequence_length = len(tgt_batch_input[0] +1)
        batch_vids = confuncs.getOpflowBatch(file_list, syncs)




        # Fit training using batch data
        _, loss = sess.run(
            [optimizer, train_loss],
            feed_dict={
                inputs_vid: batch_vids, 
                target_inputs: tgt_batch_input,
                target_outputs: tgt_batch_output
                target_sequence_length: tgt_sequence_length
            }
        )
        print (loss)
save_path = saver.save(sess, "saved_model.ckpt")
print("Model saved in path: %s" % save_path)
