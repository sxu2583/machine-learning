# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 21:48:03 2017

@author: shihab uddin

Deep learning based chat bot
"""

import tensorflow as tf
import data_utils
import seq2seq_model

def train():
    #Prepares the dataset
    enc_train, dec_train = data_utils.prepare_custom_data(gConfig['working_directory'])

    train_set = read_data(enc_train, dec_train)
    
    
#create model
def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
    return tf.nn.seq2seq.embedding_attention_seq2seq(
    encoder_inputs, decoder_inputs, cell, num_encoder_symbols = source_vocab_size,
    num_decoder_symbols = target_vocab_size, embedding_size = size, output_projection=output_projection,
    feed_previous=do_decode)

with tf.session(config=config) as sess:
    model = create_model(sess, False)
    while True:
        sess.run(model)
        
        checkpoint_path = os.path.join(gConfig['working_directory'], "seq2seq.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        
