# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : WenRichard
# @Email : RichardXie1205@gmail.com
# @File : predict_bert_crf.py
# @Software: PyCharm

"""
由于 BIO 词表得到了缩减，CRF 运行时间以及消耗内存迅速减少，cascade_bert_ner训练速度得到提高
"""
import pandas as pd
import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
from datetime import time, timedelta, datetime
import copy

from run_multitask_bert_crf import create_model, InputFeatures, InputExample
from bert import tokenization
from bert import modeling_bert
from public_tools.ner_utils import get_entity, get_entity_without_labelid
from public_tools.tag_evaluating import Metrics
from public_tools.entity_evaluating import entity_metrics, entity_metrics_without_lableid

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "do_predict_outline", False,
    "Whether to do predict outline."
)
flags.DEFINE_bool(
    "do_predict_online", True,
    "Whether to do predict online."
)


gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True


class Args():
    def __init__(self):
        self.kflod = 2
        self.is_training =False
        self.use_one_hot_embeddings = False
        self.batch_size = 1
        self.dev_file = './data/clue_ner/dev.txt'
        self.test_file = './data/clue_ner/test.txt'

        self.bert_config_file = 'D:/Expriment/pretrain_model_tf/bert/bert_config.json'
        self.output_dir = 'D:/Expriment/model_output/ner_tool/bert_crf/multi_task/clue_ner/runs/checkpoints'
        self.vocab_file = 'D:/Expriment/pretrain_model_tf/bert/vocab.txt'


args = Args()

# 加载label->id的词典
with open(os.path.join(FLAGS.output_dir, 'bmeo_label2id.pkl'), 'rb') as f:
    bmeo_label2id = pickle.load(f)
    bmeo_id2label = {value: key for key, value in bmeo_label2id.items()}
with open(os.path.join(FLAGS.output_dir, 'attr_label2id.pkl'), 'rb') as f:
    attr_label2id = pickle.load(f)
    attr_id2label = {value: key for key, value in attr_label2id.items()}

num_bmeo_labels = len(bmeo_label2id)
num_attr_labels = len(attr_label2id)


global graph
graph = tf.get_default_graph()
sess = tf.Session(config=gpu_config)


def parse_file(input_file):
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


def trans_label(bmeo_labels_id, attr_labels_id):
    """
    答案拼接2种方式：
    例子：
    bmeo：[B, M, E]
    attr: [LOC, LOC, ORG]
    1. 实体词 bmeo 与 attr 的属性全都对应起来
    result: [B_LOC, M_LOC, E_ORG]
    2. 实体词 bmeo 的e所对应的attr作为该实体的attr
    result: [B_ORG, M_ORG, E_ORG]
    目前采用第一种
    """
    std_labels = []
    for index, bmeo_line in enumerate(bmeo_labels_id):
        bmeo_attr_label = []
        attr_line = attr_labels_id[index]
        for item in list(zip(bmeo_line, attr_line)):
            bmeo_id = item[0]
            attr_id = item[1]
            if bmeo_id2label[bmeo_id] == "O":
                bmeo_attr = "O"
            else:
                bmeo_attr = bmeo_id2label[bmeo_id] + "_" + attr_id2label[attr_id]
            bmeo_attr_label.append(bmeo_attr)
        std_labels.append(bmeo_attr_label)
    # print("std labels")
    # print(std_labels)
    return std_labels


def dev_offline(file):
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line, bmeo_label, attr_label):
        feature = convert_single_example_dev(2, line, bmeo_label, attr_label, bmeo_label2id, attr_label2id, FLAGS.max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids], (1, FLAGS.max_seq_length))
        input_mask = np.reshape([feature.input_mask], (1, FLAGS.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (1, FLAGS.max_seq_length))
        bmeo_label_ids = np.reshape([feature.bmeo_label_ids], (1, FLAGS.max_seq_length))
        attr_label_ids = np.reshape([feature.attr_label_ids], (1, FLAGS.max_seq_length))
        return input_ids, input_mask, segment_ids, bmeo_label_ids, attr_label_ids

    global graph
    with graph.as_default():
        # sess.run(tf.global_variables_initializer())
        input_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="input_ids")
        input_mask_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="input_mask")
        bmeo_label_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="bmeo_label_ids")
        attr_label_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="attr_label_ids")
        segment_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="segment_ids")

        bert_config = modeling_bert.BertConfig.from_json_file(args.bert_config_file)
        (total_loss, bmeo_pred_ids, attr_pred_ids) = create_model(
            bert_config, args.is_training, input_ids_p, input_mask_p, segment_ids_p, bmeo_label2id, attr_label2id,
            bmeo_label_ids_p, attr_label_ids_p, num_bmeo_labels, num_attr_labels, args.use_one_hot_embeddings)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(args.output_dir))

        tokenizer = tokenization.FullTokenizer(
            vocab_file=args.vocab_file, do_lower_case=FLAGS.do_lower_case)
        # 获取id2char字典
        id2char = tokenizer.inv_vocab

        dev_texts, dev_bmeo_labels, dev_attr_labels = zip(*parse_file(file))
        start = datetime.now()

        bmeo_pred_labels_all = []
        bmeo_true_labels_all = []
        attr_pred_labels_all = []
        attr_true_labels_all = []
        x_all = []
        for index, text in enumerate(dev_texts):
            sentence = str(text)
            input_ids, input_mask, segment_ids, bmeo_label_ids, attr_label_ids = convert(sentence, dev_bmeo_labels[index], dev_attr_labels[index])

            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask,
                         segment_ids_p: segment_ids,
                         bmeo_label_ids_p: bmeo_label_ids,
                         attr_label_ids_p: attr_label_ids}
            # run session get current feed_dict result
            bmeo_y_pred, attr_y_pred = sess.run([bmeo_pred_ids, attr_pred_ids], feed_dict)
            # print(bmeo_y_pred)
            # print(list(bmeo_y_pred[0]))
            # print(list(attr_y_pred[0]))
            # print(len(list(y_pred[0][0])))

            bmeo_sent_tag = []
            attr_sent_tag = []
            bmeo_y_pred_clean = []
            attr_y_pred_clean = []
            input_ids_clean = []
            bmeo_y_true_clean = []
            attr_y_true_clean = []
            # 去除 [CLS] 和 [SEP]获取正确的tag范围
            for index_b, id in enumerate(list(np.reshape(input_ids, -1))):
                char = id2char[id]
                bmeo_tag = bmeo_id2label[list(bmeo_y_pred[0])[index_b]]
                attr_tag = attr_id2label[list(attr_y_pred[0])[index_b]]
                if char == "[CLS]":
                    continue
                if char == "[SEP]":
                    break
                input_ids_clean.append(id)
                bmeo_sent_tag.append(bmeo_tag)
                attr_sent_tag.append(attr_tag)
                bmeo_y_pred_clean.append(list(bmeo_y_pred[0])[index_b])
                attr_y_pred_clean.append(list(attr_y_pred[0])[index_b])
                bmeo_y_true_clean.append(bmeo_label_ids[0][index_b])
                attr_y_true_clean.append(attr_label_ids[0][index_b])

            bmeo_pred_labels_all.append(bmeo_y_pred_clean)
            bmeo_true_labels_all.append(bmeo_y_true_clean)
            attr_pred_labels_all.append(attr_y_pred_clean)
            attr_true_labels_all.append(attr_y_true_clean)
            x_all.append(input_ids_clean)
        true_labels_all = trans_label(bmeo_true_labels_all, attr_true_labels_all)
        pred_labels_all = trans_label(bmeo_pred_labels_all, attr_pred_labels_all)
        print('预测标签与真实标签评价结果......')
        print(true_labels_all)
        print(len(true_labels_all))
        print(pred_labels_all)
        print(len(pred_labels_all))

        bmeo_attr_metrics = Metrics(true_labels_all, pred_labels_all, bmeo_id2label, remove_O=True, use_id2tag=False)
        bmeo_attr_metrics.report_scores()
        # attr_metrics = Metrics(attr_true_labels_all, attr_pred_labels_all, attr_id2label, remove_O=True)
        # attr_metrics.report_scores()
        # metrics.report_confusion_matrix()

        print('预测实体与真实实体评价结果......')
        bmeo_attr_precision, bmeo_attr_recall, bmeo_attr_f1 = entity_metrics_without_lableid(x_all, true_labels_all, pred_labels_all,
                                                              id2char)
        print("BMEO_ATTR Dev P/R/F1: {} / {} / {}".format(round(bmeo_attr_precision, 2), round(bmeo_attr_recall, 2), round(bmeo_attr_f1, 2)))
        print('Time used: {} sec'.format((datetime.now() - start).seconds))


def predict_online():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.
    """
    def convert(line):
        feature = convert_single_example(line, bmeo_label2id, attr_label2id, FLAGS.max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids], (args.batch_size, FLAGS.max_seq_length))
        input_mask = np.reshape([feature.input_mask], (args.batch_size, FLAGS.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (args.batch_size, FLAGS.max_seq_length))
        bmeo_label_ids =np.reshape([feature.bmeo_label_ids], (args.batch_size, FLAGS.max_seq_length))
        attr_label_ids = np.reshape([feature.attr_label_ids], (args.batch_size, FLAGS.max_seq_length))
        return input_ids, input_mask, segment_ids, bmeo_label_ids, attr_label_ids

    global graph
    with graph.as_default():
        print("going to restore checkpoint")
        # sess.run(tf.global_variables_initializer())
        input_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="input_ids")
        input_mask_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="input_mask")
        bmeo_label_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="bmeo_label_ids")
        attr_label_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="attr_label_ids")
        segment_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="segment_ids")

        bert_config = modeling_bert.BertConfig.from_json_file(args.bert_config_file)
        (total_loss, bmeo_pred_ids, attr_pred_ids) = create_model(
            bert_config, args.is_training, input_ids_p, input_mask_p, segment_ids_p, bmeo_label2id, attr_label2id,
            bmeo_label_ids_p, attr_label_ids_p, num_bmeo_labels, num_attr_labels, args.use_one_hot_embeddings)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(args.output_dir))

        tokenizer = tokenization.FullTokenizer(
            vocab_file=args.vocab_file, do_lower_case=FLAGS.do_lower_case)
        # 获取id2char字典
        id2char = tokenizer.inv_vocab

        while True:
            print('input the test sentence:')
            sentence = str(input())
            start = datetime.now()
            if len(sentence) < 2:
                print(sentence)
                continue
            # print('your input is:{}'.format(sentence))
            input_ids, input_mask, segment_ids, bmeo_label_ids, attr_label_ids = convert(sentence)

            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask,
                         segment_ids_p: segment_ids,
                         bmeo_label_ids_p: bmeo_label_ids,
                         attr_label_ids_p: attr_label_ids}
            # run session get current feed_dict result
            bmeo_y_pred, attr_y_pred = sess.run([bmeo_pred_ids, attr_pred_ids], feed_dict)

            bmeo_sent_tag = []
            attr_sent_tag = []
            bmeo_y_pred_clean = []
            attr_y_pred_clean = []
            input_ids_clean = []
            # 去除 [CLS] 和 [SEP]获取正确的tag范围
            for index_b, id in enumerate(list(np.reshape(input_ids, -1))):
                char = id2char[id]
                bmeo_tag = bmeo_id2label[list(bmeo_y_pred[0])[index_b]]
                attr_tag = attr_id2label[list(attr_y_pred[0])[index_b]]
                if char == "[CLS]":
                    continue
                if char == "[SEP]":
                    break
                input_ids_clean.append(id)
                bmeo_sent_tag.append(bmeo_tag)
                attr_sent_tag.append(attr_tag)
                bmeo_y_pred_clean.append(list(bmeo_y_pred[0])[index_b])
                attr_y_pred_clean.append(list(attr_y_pred[0])[index_b])

            pred_sent_label = trans_label([bmeo_y_pred_clean], [attr_y_pred_clean])
            sent_tag = ' '.join(pred_sent_label[0])
            print(sentence + '\n' + sent_tag)
            entity = get_entity_without_labelid([sentence], pred_sent_label)
            print('predict_result:')
            print(entity)
            print('Time used: {} sec'.format((datetime.now() - start).seconds))


def predict_outline():
    """
    do offline prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    """
    # TODO 以文件形式预测结果,暂未开发，目前保持和 predict_online 一致
    def convert(line):
        feature = convert_single_example(line, bmeo_label2id, attr_label2id, FLAGS.max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids], (args.batch_size, FLAGS.max_seq_length))
        input_mask = np.reshape([feature.input_mask], (args.batch_size, FLAGS.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (args.batch_size, FLAGS.max_seq_length))
        bmeo_label_ids = np.reshape([feature.bmeo_label_ids], (args.batch_size, FLAGS.max_seq_length))
        attr_label_ids = np.reshape([feature.attr_label_ids], (args.batch_size, FLAGS.max_seq_length))
        return input_ids, input_mask, segment_ids, bmeo_label_ids, attr_label_ids

    global graph
    with graph.as_default():
        print("going to restore checkpoint")
        # sess.run(tf.global_variables_initializer())
        input_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="input_ids")
        input_mask_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="input_mask")
        bmeo_label_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="bmeo_label_ids")
        attr_label_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="attr_label_ids")
        segment_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="segment_ids")

        bert_config = modeling_bert.BertConfig.from_json_file(args.bert_config_file)
        (total_loss, bmeo_pred_ids, attr_pred_ids) = create_model(
            bert_config, args.is_training, input_ids_p, input_mask_p, segment_ids_p, bmeo_label2id, attr_label2id,
            bmeo_label_ids_p, attr_label_ids_p, num_bmeo_labels, num_attr_labels, args.use_one_hot_embeddings)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(args.output_dir))

        tokenizer = tokenization.FullTokenizer(
            vocab_file=args.vocab_file, do_lower_case=FLAGS.do_lower_case)
        # 获取id2char字典
        id2char = tokenizer.inv_vocab

        while True:
            print('input the test sentence:')
            sentence = str(input())
            start = datetime.now()
            if len(sentence) < 2:
                print(sentence)
                continue
            # print('your input is:{}'.format(sentence))
            input_ids, input_mask, segment_ids, bmeo_label_ids, attr_label_ids = convert(sentence)

            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask,
                         segment_ids_p: segment_ids,
                         bmeo_label_ids_p: bmeo_label_ids,
                         attr_label_ids_p: attr_label_ids}
            # run session get current feed_dict result
            bmeo_y_pred, attr_y_pred = sess.run([bmeo_pred_ids, attr_pred_ids], feed_dict)

            bmeo_sent_tag = []
            attr_sent_tag = []
            bmeo_y_pred_clean = []
            attr_y_pred_clean = []
            input_ids_clean = []
            # 去除 [CLS] 和 [SEP]获取正确的tag范围
            for index_b, id in enumerate(list(np.reshape(input_ids, -1))):
                char = id2char[id]
                bmeo_tag = bmeo_id2label[list(bmeo_y_pred[0])[index_b]]
                attr_tag = attr_id2label[list(attr_y_pred[0])[index_b]]
                if char == "[CLS]":
                    continue
                if char == "[SEP]":
                    break
                input_ids_clean.append(id)
                bmeo_sent_tag.append(bmeo_tag)
                attr_sent_tag.append(attr_tag)
                bmeo_y_pred_clean.append(list(bmeo_y_pred[0])[index_b])
                attr_y_pred_clean.append(list(attr_y_pred[0])[index_b])

            pred_sent_label = trans_label([bmeo_y_pred_clean], [attr_y_pred_clean])
            sent_tag = ' '.join(pred_sent_label[0])
            print(sentence + '\n' + sent_tag)
            entity = get_entity_without_labelid([sentence], pred_sent_label)
            print('predict_result:')
            print(entity)
            print('Time used: {} sec'.format((datetime.now() - start).seconds))


def convert_single_example_dev(ex_index, text, bmeo_label, attr_label, bmeo_label2id, attr_label2id, max_seq_length,
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
    bmeo_label_map = bmeo_label2id
    attr_label_map = attr_label2id
    bmeo_O_index = bmeo_label_map["O"]
    attr_O_index = attr_label_map["O"]
    # bmeo_L: ['B', 'M', 'M', 'E']
    # attr_L: ['ORG', 'ORG', 'ORG', 'ORG']
    # W: ['黑', '龙', '江', '省']
    textlist = text.split(' ')
    bmeo_labellist = bmeo_label.split(' ')
    attr_labellist = attr_label.split(' ')
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


def convert_single_example(example, bmeo_label2id, attr_label2id, max_seq_length, tokenizer):
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
    bmeo_label_map = bmeo_label2id
    attr_label_map = attr_label2id
    bmeo_O_index = bmeo_label_map["O"]
    attr_O_index = attr_label_map["O"]
    tokens = tokenizer.tokenize(example)
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    bmeo_label_ids = []
    attr_label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    bmeo_label_ids.append(bmeo_label_map["[CLS]"])  #
    attr_label_ids.append(attr_label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        bmeo_label_ids.append(0)
        attr_label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    bmeo_label_ids.append(bmeo_label_map["[SEP]"])
    attr_label_ids.append(attr_label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # bmeo_label用BMEO中的O去padding，可以忽略此时的padding
        bmeo_label_ids.append(bmeo_O_index)
        # attr_label用attr中的O去padding，可以忽略此时的padding
        attr_label_ids.append(attr_O_index)
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(bmeo_label_ids) == max_seq_length
    assert len(attr_label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

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


if __name__ == "__main__":
    dev_texts, dev_bmeo_labels, dev_attr_labels = zip(*parse_file(args.dev_file))
    print('dev_texts')
    print(dev_texts)
    dev_offline(args.dev_file)

    # dev_offline(args.test_file)

    # if FLAGS.do_predict_outline:
    #     predict_outline()
    # if FLAGS.do_predict_online:
    #     predict_online()
            
            