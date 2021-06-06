import tensorflow as tf
import random
import numpy as np


class MyModel(object):
    
    def __init__(self, 
                 embedding_dim, 
                 hidden_dim,
                 vocab_size_char, 
                 vocab_size_bmeo, 
                 vocab_size_attr,
                 O_tag_index,
                 use_crf,
                 embeddings,):
        
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x") # B * S
        self.input_x_len = tf.placeholder(tf.int32, [None], name="input_x_len") # B
        self.input_y_bmeo = tf.placeholder(tf.int32, [None, None], name='input_y_bmeo') # B * S
        self.input_y_attr = tf.placeholder(tf.int32, [None, None], name='input_y_attr') # B * S
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
            rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs) # B * S * D

        # Add dropout
        with tf.variable_scope("dropout"):
            rnn_outputs_dropout = tf.nn.dropout(rnn_outputs, self.dropout_keep_prob)

        with tf.variable_scope('bmeo_projection'):
            logits_bmeo = tf.layers.dense(rnn_outputs_dropout, vocab_size_bmeo) # B * S * V
            probs_bmeo = tf.nn.softmax(logits_bmeo, axis=-1)
            
            if not use_crf:
                preds_bmeo = tf.argmax(probs_bmeo, axis=-1, name="preds_bmeo") # B * S
            else:
                log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_bmeo, 
                                                                                      self.input_y_bmeo, 
                                                                                      self.input_x_len)
                preds_bmeo, crf_scores = tf.contrib.crf.crf_decode(logits_bmeo, transition_matrix, self.input_x_len)    
        
        with tf.variable_scope('attr_projection'):
            logits_attr = tf.layers.dense(rnn_outputs, vocab_size_attr) # B * S * V
            probs_attr = tf.nn.softmax(logits_attr, axis=-1)
            preds_attr = tf.argmax(probs_attr, axis=-1, name="preds_attr") # B * S
        
        self.outputs = (preds_bmeo, preds_attr)
        
        with tf.variable_scope('loss'):
            if not use_crf:
                loss_bmeo = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_bmeo, labels=self.input_y_bmeo) # B * S
                masks_bmeo = tf.sequence_mask(self.input_x_len, dtype=tf.float32) # B * S
                loss_bmeo = tf.reduce_sum(loss_bmeo * masks_bmeo, axis=-1) / tf.cast(self.input_x_len, tf.float32) # B
            else:
                loss_bmeo = -log_likelihood / tf.cast(self.input_x_len, tf.float32)
    
            loss_attr = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_attr, labels=self.input_y_attr) # B * S
            masks_attr = tf.cast(tf.not_equal(preds_bmeo, O_tag_index), tf.float32) # B * S
            loss_attr = tf.reduce_sum(loss_attr * masks_attr, axis=-1) / (tf.reduce_sum(masks_attr, axis=-1) + 1e-5) # B
            
            loss = loss_bmeo + loss_attr # B
        
        self.loss = tf.reduce_mean(loss)
            



    
