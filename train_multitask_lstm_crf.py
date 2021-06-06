import logging
import tensorflow as tf
import numpy as np
import os
import pickle
import time
import datetime

from model_multitask_lstm import MyModel
from public_tools.data_preprocess_multitask import read_corpus, load_vocab, load_tag2label, load_attr2label, batch_yield, pad_sequences
from public_tools.tag_evaluating import Metrics
from public_tools.entity_evaluating import entity_metrics, entity_metrics_without_lableid
from public_tools.ner_utils import trans_label

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat


# Logging Set
# ==================================================

log_file_path = "./log/multitask/lstm_run.log"
if os.path.exists(log_file_path): os.remove(log_file_path)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
fhlr = logging.FileHandler(log_file_path)
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("train_data_file", "./data/clue_ner/train.txt", "Data source for the train data.")
tf.flags.DEFINE_string("test_data_file", "./data/clue_ner/dev.txt", "Data source for the test data.")
tf.flags.DEFINE_string("char_vocab_file", "./data/vocab_cn.txt", "bert char vocab.")
tf.flags.DEFINE_string("tag2label_file", "./data/clue_ner/multitask/bmeo2label.txt", "bmeo2label dic.")
tf.flags.DEFINE_string("attr2label_file", "./data/clue_ner/multitask/attr2label.txt", "attr2label dic.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 768, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("max_length", 64, "Length of sentence (default: 160)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate (default: 0.001)")
tf.flags.DEFINE_float("clip_grade", 5.0, "clip_grad (default: 5.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("use_clip_grad", False, "clip grad")


tf.flags.DEFINE_string('model_version', '1', 'version number of the model.')
FLAGS = tf.flags.FLAGS


# Data Preparation
# ==================================================


# Load data
logger.info("Loading data...")

train_data = read_corpus(FLAGS.train_data_file)
logger.info(train_data)
test_data = read_corpus(FLAGS.test_data_file)
char2id, id2char = load_vocab(FLAGS.char_vocab_file)
bmeo2id, id2bmeo = load_tag2label(FLAGS.tag2label_file)
attr2id, id2attr = load_attr2label(FLAGS.attr2label_file)

# Load embeddings
# BERT字向量比随机初始化效果好4个点左右
with open('./embedding/new_bert_embedding.pkl', 'rb') as f:
    char_embeddings = pickle.load(f)

# char_embeddings = None

export_path_base = "D:/Expriment/model_output/ner_tool/lstm_crf/multi_task/clue_ner/"
export_path = os.path.join(
      compat.as_bytes(export_path_base),
      compat.as_bytes(str(FLAGS.model_version)))
logger.info('Exporting trained model to', export_path)

# Output directory for models and summaries
timestamp = str(int(time.time()))
checkpoint_out_dir = os.path.join("D:/Expriment/model_output/ner_tool/lstm_crf/multi_task/clue_ner/runs/", timestamp)
if not os.path.exists(checkpoint_out_dir):
    os.makedirs(checkpoint_out_dir)
logger.info("Writing to {}\n".format(checkpoint_out_dir))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.compat.v1.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        logger.info("building model...")

        model = MyModel(embedding_dim=768,
                        hidden_dim=300,
                        vocab_size_char=len(char2id),
                        vocab_size_bmeo=len(bmeo2id),
                        vocab_size_attr=len(attr2id),
                        O_tag_index=bmeo2id["O"],
                        use_crf=False,
                        embeddings=char_embeddings)
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        if FLAGS.use_clip_grad:
            grads_and_vars = [[tf.clip_by_value(g, FLAGS.clip_grad, FLAGS.clip_grad), v] for g, v in
                              grads_and_vars]
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.compat.v1.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.compat.v1.summary.scalar("{}/grad/sparsity".format(v.name),
                                                               tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(checkpoint_out_dir, "summaries", "train")
        if not os.path.exists(train_summary_dir):
            os.makedirs(train_summary_dir)
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary])
        dev_summary_dir = os.path.join(checkpoint_out_dir, "summaries", "dev")
        if not os.path.exists(dev_summary_dir):
            os.makedirs(dev_summary_dir)
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(checkpoint_out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, input_x_len, y_batch_bmeo, y_batch_attr):
            """
            A single training step
            """
            feed_dict = {
                model.input_x: x_batch,
                model.input_x_len: input_x_len,
                model.input_y_bmeo: y_batch_bmeo,
                model.input_y_attr: y_batch_attr,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                model.lr: FLAGS.learning_rate,
            }
            _, step, summaries, loss = sess.run(
                [train_op, global_step, train_summary_op, model.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logger.info("{}: step {}, loss {:g}".format(time_str, step, loss))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, input_x_len, y_batch_bmeo, y_batch_attr,  writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                model.input_x: x_batch,
                model.input_x_len: input_x_len,
                model.input_y_bmeo: y_batch_bmeo,
                model.input_y_attr: y_batch_attr,
                model.dropout_keep_prob: 1.0,
                model.lr: FLAGS.learning_rate,
            }
            step, summaries, (preds_bmeo, preds_attr), loss = sess.run(
                [global_step, dev_summary_op, model.outputs, model.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logger.info("{}: step {}, loss {:g}".format(time_str, step, loss))

            if writer:
                writer.add_summary(summaries, step)

            preds_bmeo = preds_bmeo.tolist()
            preds_attr = preds_attr.tolist()
            return preds_bmeo, preds_attr


        logger.info("model params:")
        params_num_all = 0
        for variable in tf.trainable_variables():
            params_num = 1
            for dim in variable.shape:
                params_num *= dim
            params_num_all += params_num
            logger.info("\t {} {} {}".format(variable.name, variable.shape, params_num))
        logger.info("all params num: " + str(params_num_all))

        logger.info("start training...")

        best_f1 = 0.0
        for epoch in range(FLAGS.num_epochs):
            num_batches = (len(train_data) + FLAGS.batch_size - 1) // FLAGS.batch_size

            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            is_shuffule = True
            batches = batch_yield(train_data, FLAGS.batch_size, char2id, bmeo2id, attr2id, shuffle=is_shuffule)
            for step, (seqs, tag_labels, attr_labels) in enumerate(batches):
                logger.info(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
                step_num = epoch * num_batches + step + 1
                x_train, train_seq_len_list = pad_sequences(seqs, max_len=FLAGS.max_length, pad_mark=0)
                tag_labels_train, _ = pad_sequences(tag_labels, max_len=FLAGS.max_length, pad_mark=0)
                attr_labels_train, _ = pad_sequences(attr_labels, max_len=FLAGS.max_length, pad_mark=0)
                train_step(x_train, train_seq_len_list, tag_labels_train, attr_labels_train)

                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    logger.info("\nEvaluation:")
                    test_batches = batch_yield(test_data, FLAGS.batch_size, char2id, bmeo2id, attr2id, shuffle=is_shuffule)
                    test_labels_all_bmeo = []
                    test_labels_all_attr = []
                    test_labels_pred_bmeo = []
                    test_labels_pred_attr = []
                    x_all = []
                    for _, (test_seqs, test_tag_labels, test_attr_labels) in enumerate(test_batches):
                        x_test, test_seq_len_list = pad_sequences(test_seqs, FLAGS.max_length, pad_mark=0)
                        y_test_tag, _ = pad_sequences(test_tag_labels, FLAGS.max_length, pad_mark=0)
                        y_test_attr, _ = pad_sequences(test_attr_labels, FLAGS.max_length, pad_mark=0)
                        test_preds_bmeo, test_preds_attr = dev_step(x_test, test_seq_len_list, y_test_tag, y_test_attr, writer=dev_summary_writer)
                        test_labels_all_bmeo.extend(y_test_tag)
                        test_labels_all_attr.extend(y_test_attr)
                        test_labels_pred_bmeo.extend(test_preds_bmeo)
                        test_labels_pred_attr.extend(test_preds_attr)
                        x_all.extend(x_test)

                    # 将bmeo标签和attr标签拼接
                    true_labels_all = trans_label(test_labels_all_bmeo, test_labels_pred_attr, id2bmeo, id2attr)
                    pred_labels_all = trans_label(test_labels_pred_bmeo, test_labels_pred_attr, id2bmeo, id2attr)
                    
                    logger.info('预测标签与真实标签评价结果......')
                    # logger.info("仅bmeo部分预测评价结果")
                    # metrics_tag = Metrics(test_labels_all_bmeo, test_labels_pred_bmeo, id2bmeo, remove_O=True)
                    # metrics_tag.report_scores()
                    # metrics_tag.report_confusion_matrix()
                    #
                    # logger.info("仅attr部分预测评价结果")
                    # metrics_attr = Metrics(test_labels_all_attr, test_labels_pred_attr, id2attr, remove_O=True)
                    # metrics_attr.report_scores()
                    # metrics_attr.report_confusion_matrix()

                    bmeo_attr_metrics = Metrics(true_labels_all, pred_labels_all, id2bmeo, remove_O=True,
                                                use_id2tag=False)
                    bmeo_attr_metrics.report_scores()
                    

                    logger.info('预测实体与真实实体评价结果......')
                    precision, recall, f1 = entity_metrics_without_lableid(x_all, true_labels_all, pred_labels_all, id2char)
                    logger.info("Test P/R/F1: {} / {} / {}".format(round(precision, 2), round(recall, 2), round(f1, 2)))
                    if f1 > best_f1:
                        best_f1 = f1
                    logger.info("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {}\n".format(path))
        logger.info('best_f1: {}'.format(best_f1))




            
            