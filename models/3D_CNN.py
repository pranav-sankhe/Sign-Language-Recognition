ef word_to_int():
    filepath = 'label.csv'
    data = pd.read_csv(filepath, header=None, names=['0','1','2'])
    data = data.values
    data = data[:,(1,2)]

EMBEDDING_SIZE = 512

def embeddings():
    # Embedding
    embedding_decoder = variable_scope.get_variable(
        "embedding_encoder", [VOCAB_SIZE, EMBEDDING_SIZE])
    # Look up embedding:
    #   encoder_inputs: [max_time, batch_size]
    #   encoder_emb_inp: [max_time, batch_size, embedding_size]
    decoder_emb_inp = embedding_ops.embedding_lookup(
        embedding_encoder, decoder_inputs)
    return decoder_emb_inp


word_to_int()


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
outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
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
