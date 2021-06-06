# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import bert.modeling_bert as modeling_bert
import bert.modeling_google_albert as modeling_albert
from bert import optimization
from bert import tokenization
from layers.lstm_crf_layer import BLSTM_CRF
import tensorflow as tf
import pickle
from tensorflow.contrib.layers.python.layers import initializers
from public_tools import tf_metrics
from public_tools.entity_evaluating import entity_metrics


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", "./data/clue_ner",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", "D:/Expriment/pretrain_model_tf/bert/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "ner", "The name of the task to train.")

flags.DEFINE_string("vocab_file", "D:/Expriment/pretrain_model_tf/bert/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("bmeo_tag_file", "./data/clue_ner/multitask/bmeo2label.txt",
                    "The label set of ner dataset.")
flags.DEFINE_string("attr_tag_file", "./data/clue_ner/multitask/attr2label.txt",
                    "The label set of ner dataset.")

flags.DEFINE_string(
    "output_dir", "D:/Expriment/model_output/ner_tool/bert_crf/multi_task/clue_ner/runs/checkpoints",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "export_dir", None,
    "The dir where the exported model will be written.")


# lstm parameters
flags.DEFINE_bool(
    "use_lstm", True,
    "Whether to use lstm.")
flags.DEFINE_bool(
    "use_crf", True,
    "Whether to use crf.")
flags.DEFINE_integer('lstm_size', 128, 'size of lstm units')
flags.DEFINE_integer('num_layers', 1, 'number of rnn layers, default is 1')
flags.DEFINE_string('cell', 'lstm', 'which rnn cell used')


## Other parameters
flags.DEFINE_string(
    "init_checkpoint", "D:/Expriment/pretrain_model_tf/bert/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("use_albert", False, "Whether to run training.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "do_export", False,
    "Whether to export the model.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("bert_learning_rate", 5e-5, "The initial learning rate of bert for Adam.")

flags.DEFINE_float("others_learning_rate", 2e-4, "The initial learning rate of others model for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float('droupout_rate', 0.5, 'Dropout rate')

flags.DEFINE_float('clip', 5, 'Gradient clip')

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, bmeo_label=None, attr_label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.bmeo_label = bmeo_label
    self.attr_label = attr_label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               bmeo_label_ids,
               attr_label_ids,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.bmeo_label_ids = bmeo_label_ids
    self.attr_label_ids = attr_label_ids
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  # def get_labels_a(self):
  #   """Gets the list of labels for this data set."""
  #   raise NotImplementedError()
  #
  # def get_labels_b(self):
  #   """Gets the list of labels for this data set."""
  #   raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  @classmethod
  def _read_data(cls, input_file):
    """
    file format:
    中	B-ORG
    共	M-ORG
    中	M-ORG
    央	E-ORG
    致	O
    中	B-ORG
    国	M-ORG
    致	M-ORG
    公	M-ORG
    党	M-ORG
    十	M-ORG
    一	M-ORG
    大	E-ORG
    的	O
    贺	O
    词	O

    各	O
    位	O
    代	O
    表	O
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = []
        words = []
        bmeo_labels = []
        attr_labels = []
        for line in f:
            contends = line.strip()
            tokens = contends.split('\t')
            if len(tokens) == 2:
                word = line.strip().split('\t')[0]
                label = line.strip().split('\t')[-1]
            else:
                if len(contends) == 0:
                    # L: 'B-ORG M-ORG M-ORG M-ORG'
                    # W: '中 共 中 央'
                    bmeo_s = ' '.join([label for label in bmeo_labels if len(label) > 0])
                    attr_s = ' '.join([label for label in attr_labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])

                    lines.append([w, bmeo_s, attr_s])
                    words = []
                    bmeo_labels = []
                    attr_labels = []
                    continue
            words.append(word)
            bmeo = label.split('-')[0]
            # 将s的attr置为'O'
            if len(label.split('-')) > 1:
                attr = label.split('-')[1]
            else:
                attr = 'O'
            bmeo_labels.append(bmeo)
            attr_labels.append(attr)
        return lines

  @classmethod
  def _read_label(cls, bmeo_tag_file, attr_tag_file):
      # 读取bmeo的标签 和 attr的标签
      bmeo_tags = []
      attr_tags = []
      with open(bmeo_tag_file, 'r', encoding='utf-8') as f:
          lines = f.readlines()
          for line in lines:
              line = line.strip('\n').split('\t')
              tag = line[0]
              bmeo_tags.append(tag)
      with open(attr_tag_file, 'r', encoding='utf-8') as f:
          lines = f.readlines()
          for line in lines:
              line = line.strip('\n').split('\t')
              tag = line[0]
              attr_tags.append(tag)
      bmeo_tags.append("X")
      bmeo_tags.append("[CLS]")
      bmeo_tags.append("[SEP]")
      attr_tags.append("X")
      attr_tags.append("[CLS]")
      attr_tags.append("[SEP]")
      return bmeo_tags, attr_tags


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return self._read_label(FLAGS.bmeo_tag_file, FLAGS.attr_tag_file)

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[0])
            bmeo_label = tokenization.convert_to_unicode(line[1])
            attr_label = tokenization.convert_to_unicode(line[2])
            if i == 0:
                print('bmeo_label示例：{}'.format(bmeo_label))
                print('attr_label示例：{}'.format(attr_label))
            examples.append(InputExample(guid=guid, text=text, bmeo_label=bmeo_label, attr_label=attr_label))
        return examples

##
def convert_single_example(ex_index, example, bmeo_label_list, attr_label_list, max_seq_length,
                           tokenizer):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    bmeo_label_map = {}
    attr_label_map = {}
    for (i, label) in enumerate(bmeo_label_list):
        bmeo_label_map[label] = i
    for (i, label) in enumerate(attr_label_list):
        attr_label_map[label] = i

    bmeo_O_index = bmeo_label_map["O"]
    attr_O_index = attr_label_map["O"]

    # 保存label->index 的map
    if not os.path.exists(os.path.join(FLAGS.output_dir, 'bmeo_label2id.pkl')):
        with open(os.path.join(FLAGS.output_dir, 'bmeo_label2id.pkl'), 'wb') as w:
            pickle.dump(bmeo_label_map, w)
    if not os.path.exists(os.path.join(FLAGS.output_dir, 'attr_label2id.pkl')):
        with open(os.path.join(FLAGS.output_dir, 'attr_label2id.pkl'), 'wb') as w:
            pickle.dump(attr_label_map, w)

    # bmeo_L: ['B', 'M', 'M', 'E']
    # attr_L: ['ORG', 'ORG', 'ORG', 'ORG']
    # W: ['黑', '龙', '江', '省']
    textlist = example.text.split(' ')
    bmeo_labellist = example.bmeo_label.split(' ')
    attr_labellist = example.attr_label.split(' ')
    tokens = []
    bmeo_labels = []
    attr_labels = []
    for i, word in enumerate(textlist):
        # 对每个字进行tokenize，返回list
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        bmeo_label_1 = bmeo_labellist[i]
        attr_label_1 = attr_labellist[i]
        for m in range(len(token)):
            if m == 0:
                bmeo_labels.append(bmeo_label_1)
                attr_labels.append(attr_label_1)
            else:  # 一般不会出现else
                bmeo_labels.append("X")
                attr_labels.append("X")

    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        bmeo_labels = bmeo_labels[0:(max_seq_length - 2)]
        attr_labels = attr_labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    bmeo_label_ids = []
    attr_label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    bmeo_label_ids.append(bmeo_label_map["[CLS]"])  #
    attr_label_ids.append(attr_label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        bmeo_label_ids.append(bmeo_label_map[bmeo_labels[i]])
        attr_label_ids.append(attr_label_map[attr_labels[i]])
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    bmeo_label_ids.append(bmeo_label_map["[SEP]"])
    attr_label_ids.append(attr_label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # bmeo_label用BMEO中的O去padding，可以忽略此时的padding
        bmeo_label_ids.append(bmeo_O_index)
        # attr_label用attr中的O去padding，可以忽略此时的padding
        attr_label_ids.append(attr_O_index)
        ntokens.append("[PAD]")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(bmeo_label_ids) == max_seq_length
    assert len(attr_label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("bmeo_label_ids: %s" % " ".join([str(x) for x in bmeo_label_ids]))
        tf.logging.info("attr_label_ids: %s" % " ".join([str(x) for x in attr_label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        bmeo_label_ids=bmeo_label_ids,
        attr_label_ids=attr_label_ids,
        # label_mask = label_mask
    )
    return feature


def file_based_convert_examples_to_features(
        examples, bmeo_label_list, attr_label_list, max_seq_length, tokenizer, output_file):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  样本
    :param label_list:标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :param mode:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        # 对于每一个训练样本,
        feature = convert_single_example(ex_index, example, bmeo_label_list, attr_label_list, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["bmeo_label_ids"] = create_int_feature(feature.bmeo_label_ids)
        features["attr_label_ids"] = create_int_feature(feature.attr_label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "bmeo_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "attr_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, bmeo_label2id, attr_label2id,
                 bmeo_labels, attr_labels, num_bmeo_labels, num_attr_bmeo_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  if FLAGS.use_albert:
      model = modeling_albert.AlbertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=segment_ids,
          use_one_hot_embeddings=use_one_hot_embeddings)
  else:
      model = modeling_bert.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=segment_ids,
          use_one_hot_embeddings=use_one_hot_embeddings)

  # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
  bert_outputs = model.get_sequence_output()
  max_seq_length = bert_outputs.shape[1].value
  bmeo_o_index = bmeo_label2id["O"]
  # 以下与传入的 num_bmeo_labels, num_attr_bmeo_labels 等价
  vocab_size_attr = len(attr_label2id)
  vocab_size_bmeo = len(bmeo_label2id)

  used = tf.sign(tf.abs(input_ids))
  # lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
  lengths = tf.reduce_sum(input_mask, axis=-1) # B
  if not FLAGS.use_lstm:
      hiddens = bert_outputs
  else:
      with tf.variable_scope('bilstm'):
          cell_fw = tf.nn.rnn_cell.LSTMCell(300)
          cell_bw = tf.nn.rnn_cell.LSTMCell(300)
          (
          (rnn_fw_outputs, rnn_bw_outputs), (rnn_fw_final_state, rnn_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
              cell_fw=cell_fw,
              cell_bw=cell_bw,
              inputs=bert_outputs,
              sequence_length=lengths,
              dtype=tf.float32
          )
          rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs)  # B * S * D
      hiddens = rnn_outputs

  with tf.variable_scope('bmeo_projection'):
      logits_bmeo = tf.layers.dense(hiddens, vocab_size_bmeo)  # B * S * V
      probs_bmeo = tf.nn.softmax(logits_bmeo, axis=-1)

      if not FLAGS.use_crf:
          preds_bmeo = tf.argmax(probs_bmeo, axis=-1, name="preds_bmeo")  # B * S
      else:
          log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_bmeo,
                                                                                bmeo_labels,
                                                                                lengths)
          preds_bmeo, crf_scores = tf.contrib.crf.crf_decode(logits_bmeo, transition_matrix, lengths)

  with tf.variable_scope('attr_projection'):
      logits_attr = tf.layers.dense(hiddens, vocab_size_attr)  # B * S * V
      probs_attr = tf.nn.softmax(logits_attr, axis=-1)
      preds_attr = tf.argmax(probs_attr, axis=-1, name="preds_attr")  # B * S

  with tf.variable_scope('loss'):
      if not FLAGS.use_crf:
          loss_bmeo = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_bmeo,
                                                                    labels=bmeo_labels)  # B * S
          masks_bmeo = tf.sequence_mask(lengths, dtype=tf.float32)  # B * S
          loss_bmeo = tf.reduce_sum(loss_bmeo * masks_bmeo, axis=-1) / tf.cast(lengths, tf.float32)  # B
      else:
          loss_bmeo = -log_likelihood / tf.cast(lengths, tf.float32)

      loss_attr = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_attr,
                                                                 labels=attr_labels)  # B * S
      # 不将BMEO中的O所对应的的attr属性的loss计算在内
      masks_attr = tf.cast(tf.not_equal(preds_bmeo, bmeo_o_index), tf.float32)  # B * S
      loss_attr = tf.reduce_sum(loss_attr * masks_attr, axis=-1) / (tf.reduce_sum(masks_attr, axis=-1) + 1e-5)  # B
      loss = tf.reduce_mean(loss_bmeo + loss_attr)  # B

  return (loss, preds_bmeo, preds_attr)


def model_fn_builder(bert_config, num_bmeo_labels, num_attr_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, char2id):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    bmeo_label_ids = features["bmeo_label_ids"]
    attr_label_ids = features["attr_label_ids"]
    print('shape of input_ids', input_ids.shape)

    with open(os.path.join(FLAGS.output_dir, 'bmeo_label2id.pkl'), 'rb') as f:
        bmeo_label2id = pickle.load(f)
    with open(os.path.join(FLAGS.output_dir, 'attr_label2id.pkl'), 'rb') as f:
        attr_label2id = pickle.load(f)
    print('load bmeo_label2id success!')
    print('load attr_label2id success!')

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, bmeo_pred_ids, attr_pred_ids) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, bmeo_label2id, attr_label2id, bmeo_label_ids,
        attr_label_ids, num_bmeo_labels, num_attr_labels, use_one_hot_embeddings)


    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if FLAGS.use_albert:
        modeling = modeling_albert
    else:
        modeling = modeling_bert
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      # bert原始采用的lr衰减策略
      train_op = optimization.create_optimizer(
           total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      # 本项目bert采用更小的lr，后接模型采用稍大的lr
      # train_op = optimization.create_optimizer_multitask(total_loss, bert_lr=FLAGS.bert_learning_rate,
      #                                                    others_lr=FLAGS.others_learning_rate)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      # 针对NER ,进行了修改,预测脚本有点问题
      def metric_fn(bmeo_label_ids, bmeo_pred_ids, attr_label_ids, attr_pred_ids):
        bmeo_indices = []
        bmeo_calulate_label = []
        bmeo_id2label = {}

        attr_indices = []
        attr_calulate_label = []
        attr_id2label = {}

        for k, v in bmeo_label2id.items():
            bmeo_id2label[v] = k
            if k == 'O' or k == '[CLS]' or k== '[SEP]':
                pass
            else:
                bmeo_calulate_label.append(k)
                bmeo_indices.append(v)
        print('bmeo_calulate_label: {}'.format(bmeo_calulate_label))
        print('bmeo_indices: {}'.format(bmeo_indices))
        weight = tf.sequence_mask(FLAGS.max_seq_length)
        bmeo_precision = tf_metrics.precision(bmeo_label_ids, bmeo_pred_ids, num_bmeo_labels, bmeo_indices, weight)
        bmeo_recall = tf_metrics.recall(bmeo_label_ids, bmeo_pred_ids, num_bmeo_labels, bmeo_indices, weight)
        bmeo_f = tf_metrics.f1(bmeo_label_ids, bmeo_pred_ids, num_bmeo_labels, bmeo_indices, weight)

        for k, v in attr_label2id.items():
            attr_id2label[v] = k
            if k == 'O' or k == '[CLS]' or k == '[SEP]':
                pass
            else:
                attr_calulate_label.append(k)
                attr_indices.append(v)
        print('attr_calulate_label: {}'.format(attr_calulate_label))
        print('attr_indices: {}'.format(attr_indices))
        weight = tf.sequence_mask(FLAGS.max_seq_length)
        attr_precision = tf_metrics.precision(attr_label_ids, attr_pred_ids, num_attr_labels, attr_indices, weight)
        attr_recall = tf_metrics.recall(attr_label_ids, attr_pred_ids, num_attr_labels, attr_indices, weight)
        attr_f = tf_metrics.f1(attr_label_ids, attr_pred_ids, num_attr_labels, attr_indices, weight)

        return {
            "bmeo_eval_precision": bmeo_precision,
            "bmeo_eval_recall": bmeo_recall,
            "bmeo_eval_f": bmeo_f,
            "attr_eval_precision": attr_precision,
            "attr_eval_recall": attr_recall,
            "attr_eval_f": attr_f,
            # "eval_loss": loss,
        }
      eval_metrics = (metric_fn, [bmeo_label_ids, bmeo_pred_ids, attr_label_ids, attr_pred_ids])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=bmeo_pred_ids,
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn

def serving_input_fn():
    bmeo_label_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='bmeo_label_ids')
    attr_label_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='attr_label_ids')
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'bmeo_label_ids': bmeo_label_ids,
        'attr_label_ids': attr_label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()
    return input_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "ner": NerProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  if FLAGS.use_albert:
    bert_config = modeling_albert.AlbertConfig.from_json_file(FLAGS.bert_config_file)
  else:
    bert_config = modeling_bert.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  # 获取bmeo和attr各自的label集合
  bmeo_label_list, attr_label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  # 获取id2char字典
  id2char = tokenizer.inv_vocab
  # 获取char2id字典
  char2id = tokenizer.vocab

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_bmeo_labels=len(bmeo_label_list),
      num_attr_labels=len(attr_label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      char2id=char2id)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, bmeo_label_list, attr_label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, bmeo_label_list, attr_label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, bmeo_label_list,
                                            attr_label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples

  if FLAGS.do_export:
      estimator._export_to_tpu = False
      estimator.export_savedmodel(FLAGS.export_dir, serving_input_fn)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()


