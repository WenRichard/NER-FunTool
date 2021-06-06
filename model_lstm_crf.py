import tensorflow as tf
import random
import numpy as np


class MyModel(object):
    
    def __init__(self, 
                 embedding_dim, 
                 hidden_dim, 
                 vocab_size_char, 
                 vocab_size_bio, 
                 use_crf,
                 embeddings
                 ):

        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_x_len = tf.placeholder(tf.int32, [None], name="input_x_len")
        self.input_y = tf.placeholder(tf.int32, [None, None], name='input_y')  # 句子的label
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        
        with tf.variable_scope('embedding_layer'):
            if embeddings is not None:
                embedding_matrix = tf.Variable(tf.to_float(embeddings), trainable=False, name='embedding_matrix')
            else:
                embedding_matrix = tf.get_variable("embedding_matrix", [vocab_size_char, embedding_dim], dtype=tf.float32)
            embedded = tf.nn.embedding_lookup(embedding_matrix, self.input_x)
        
        with tf.variable_scope('encoder'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_dim)
            cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_dim)
            ((rnn_fw_outputs, rnn_bw_outputs), (rnn_fw_final_state, rnn_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, 
                cell_bw=cell_bw, 
                inputs=embedded, 
                sequence_length=self.input_x_len,
                dtype=tf.float32
            )
            rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs) # B * S1 * D

        # Add dropout
        with tf.variable_scope("dropout"):
            rnn_outputs_dropout = tf.nn.dropout(rnn_outputs, self.dropout_keep_prob)

        with tf.variable_scope('projection'):
            logits_seq = tf.layers.dense(rnn_outputs_dropout, vocab_size_bio) # B * S * V
            self.probs_seq = tf.nn.softmax(logits_seq)
            
            if not use_crf:
                preds_seq = tf.argmax(self.probs_seq, axis=-1, name="preds_seq") # B * S
            else:
                log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(inputs=logits_seq,
                                                                                      tag_indices=self.input_y,
                                                                                      sequence_lengths=self.input_x_len)
                preds_seq, crf_scores = tf.contrib.crf.crf_decode(logits_seq, transition_matrix, self.input_x_len)
            
        self.outputs = preds_seq
        
        with tf.variable_scope('loss'):
            if not use_crf: 
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_seq, labels=self.input_y) # B * S
                masks = tf.sequence_mask(self.input_x_len, dtype=tf.float32) # B * S
                loss = tf.reduce_sum(loss * masks, axis=-1) / tf.cast(self.input_x_len, tf.float32) # B
            else:
                loss = -log_likelihood / tf.cast(self.input_x_len, tf.float32) # B
            
        self.loss = tf.reduce_mean(loss)





    
