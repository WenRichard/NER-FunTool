# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : WenRichard
# @Email : RichardXie1205@gmail.com
# @File : predict_bert_crf.py
# @Software: PyCharm

import pandas as pd
import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
from datetime import time, timedelta, datetime
import copy

from run_bert_crf import create_model, InputFeatures, InputExample
from bert import tokenization
from bert import modeling_bert
from public_tools.ner_utils import get_entity
from public_tools.tag_evaluating import Metrics
from public_tools.entity_evaluating import entity_metrics

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "do_predict_outline", True,
    "Whether to do predict outline."
)
flags.DEFINE_bool(
    "do_predict_online", False,
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
        self.output_dir = 'D:/Expriment/model_output/ner_tool/bert_crf/single_task/clue_ner/runs/checkpoints'
        self.vocab_file = 'D:/Expriment/pretrain_model_tf/bert/vocab.txt'


args = Args()

# 加载label->id的词典
with codecs.open(os.path.join(args.output_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

num_labels = len(label2id)

global graph
graph = tf.get_default_graph()
sess = tf.Session(config=gpu_config)


def parse_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = []
        words = []
        labels = []
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
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([w, l])
                    words = []
                    labels = []
                    continue
            words.append(word)
            labels.append(label)
        return lines


def dev_offline(file):
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line, label):
        feature = convert_single_example_dev(2, line, label, label2id, FLAGS.max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids], (1, FLAGS.max_seq_length))
        input_mask = np.reshape([feature.input_mask], (1, FLAGS.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (1, FLAGS.max_seq_length))
        label_ids =np.reshape([feature.label_ids], (1, FLAGS.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        # sess.run(tf.global_variables_initializer())
        input_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="input_ids")
        input_mask_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="input_mask")
        label_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="label_ids")
        segment_ids_p = tf.placeholder(tf.int32, [1, FLAGS.max_seq_length], name="segment_ids")

        bert_config = modeling_bert.BertConfig.from_json_file(args.bert_config_file)
        (total_loss, logits, trans, pred_ids) = create_model(
            bert_config, args.is_training, input_ids_p, input_mask_p, segment_ids_p,
            label_ids_p, num_labels, args.use_one_hot_embeddings)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(args.output_dir))

        tokenizer = tokenization.FullTokenizer(
            vocab_file=args.vocab_file, do_lower_case=FLAGS.do_lower_case)
        # 获取id2char字典
        id2char = tokenizer.inv_vocab

        dev_texts, dev_labels = zip(*parse_file(file))
        start = datetime.now()

        pred_labels_all = []
        true_labels_all = []
        x_all = []
        for index, text in enumerate(dev_texts):
            sentence = str(text)
            input_ids, input_mask, segment_ids, label_ids = convert(sentence, dev_labels[index])

            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask,
                         segment_ids_p: segment_ids,
                         label_ids_p: label_ids}
            # run session get current feed_dict result
            y_pred = sess.run([pred_ids], feed_dict)
            # print(list(y_pred[0][0]))
            # print(len(list(y_pred[0][0])))

            sent_tag = []
            y_pred_clean = []
            input_ids_clean = []
            y_true_clean = []
            # 去除 [CLS] 和 [SEP]获取正确的tag范围
            for index_b, id in enumerate(list(np.reshape(input_ids, -1))):
                char = id2char[id]
                tag = id2label[list(y_pred[0][0])[index_b]]
                if char == "[CLS]":
                    continue
                if char == "[SEP]":
                    break
                input_ids_clean.append(id)
                sent_tag.append(tag)
                y_pred_clean.append(list(y_pred[0][0])[index_b])
                y_true_clean.append(label_ids[0][index_b])

            pred_labels_all.append(y_pred_clean)
            true_labels_all.append(y_true_clean)
            x_all.append(input_ids_clean)

        print('预测标签与真实标签评价结果......')
        print(pred_labels_all)
        print(len(pred_labels_all))
        print(true_labels_all)
        print(len(true_labels_all))

        metrics = Metrics(true_labels_all, pred_labels_all, id2label, remove_O=True)
        metrics.report_scores()
        # metrics.report_confusion_matrix()

        print('预测实体与真实实体评价结果......')
        precision, recall, f1 = entity_metrics(x_all, pred_labels_all, true_labels_all, id2char, id2label)
        print("Dev P/R/F1: {} / {} / {}".format(round(precision, 2), round(recall, 2), round(f1, 2)))
        print('Time used: {} sec'.format((datetime.now() - start).seconds))


def predict_online():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.
    """
    def convert(line):
        feature = convert_single_example(line, label2id, FLAGS.max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids], (args.batch_size, FLAGS.max_seq_length))
        input_mask = np.reshape([feature.input_mask], (args.batch_size, FLAGS.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (args.batch_size, FLAGS.max_seq_length))
        label_ids =np.reshape([feature.label_ids], (args.batch_size, FLAGS.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        print("going to restore checkpoint")
        # sess.run(tf.global_variables_initializer())
        input_ids_p = tf.placeholder(tf.int32, [args.batch_size, FLAGS.max_seq_length], name="input_ids")
        input_mask_p = tf.placeholder(tf.int32, [args.batch_size, FLAGS.max_seq_length], name="input_mask")
        label_ids_p = tf.placeholder(tf.int32, [args.batch_size, FLAGS.max_seq_length], name="label_ids")
        segment_ids_p = tf.placeholder(tf.int32, [args.batch_size, FLAGS.max_seq_length], name="segment_ids")

        bert_config = modeling_bert.BertConfig.from_json_file(args.bert_config_file)
        (total_loss, logits, trans, pred_ids) = create_model(
            bert_config, args.is_training, input_ids_p, input_mask_p, segment_ids_p,
            label_ids_p, num_labels, args.use_one_hot_embeddings)

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
            input_ids, input_mask, segment_ids, label_ids = convert(sentence)

            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask,
                         segment_ids_p:segment_ids,
                         label_ids_p:label_ids}
            # run session get current feed_dict result
            y_pred = sess.run([pred_ids], feed_dict)

            sent_tag = []
            y_pred_clean = []
            input_ids_clean = []
            # 去除 [CLS] 和 [SEP]获取正确的tag范围
            print([id2char[i] for i in list(np.reshape(input_ids, -1))])
            print(len(list(np.reshape(input_ids, -1))))
            print([id2label[i] for i in list(y_pred[0][0])])
            print(len(list(y_pred[0][0])))
            for index, id in enumerate(list(np.reshape(input_ids, -1))):
                char = id2char[id]
                tag = id2label[list(y_pred[0][0])[index]]
                if char == "[CLS]":
                    continue
                if char == "[SEP]":
                    break
                input_ids_clean.append(id)
                sent_tag.append(tag)
                y_pred_clean.append(list(y_pred[0][0])[index])

            sent_tag = ' '.join(sent_tag)
            print(sentence + '\n' + sent_tag)
            entity = get_entity([sentence], [y_pred_clean], id2label)
            print('predict_result:')
            print(entity)
            print('Time used: {} sec'.format((datetime.now() - start).seconds))


def predict_outline():
    """
    do offline prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    """
    # TODO 以文件形式预测结果
    def convert(line):
        feature = convert_single_example(line, label2id, FLAGS.max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids], (args.batch_size, FLAGS.max_seq_length))
        input_mask = np.reshape([feature.input_mask], (args.batch_size, FLAGS.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (args.batch_size, FLAGS.max_seq_length))
        label_ids = np.reshape([feature.label_ids], (args.batch_size, FLAGS.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        print("going to restore checkpoint")
        # sess.run(tf.global_variables_initializer())
        input_ids_p = tf.placeholder(tf.int32, [args.batch_size, FLAGS.max_seq_length], name="input_ids")
        input_mask_p = tf.placeholder(tf.int32, [args.batch_size, FLAGS.max_seq_length], name="input_mask")
        label_ids_p = tf.placeholder(tf.int32, [args.batch_size, FLAGS.max_seq_length], name="label_ids")
        segment_ids_p = tf.placeholder(tf.int32, [args.batch_size, FLAGS.max_seq_length], name="segment_ids")

        bert_config = modeling_bert.BertConfig.from_json_file(args.bert_config_file)
        (total_loss, logits, trans, pred_ids) = create_model(
            bert_config, args.is_training, input_ids_p, input_mask_p, segment_ids_p,
            label_ids_p, num_labels, args.use_one_hot_embeddings)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(args.output_dir))

        tokenizer = tokenization.FullTokenizer(
            vocab_file=args.vocab_file, do_lower_case=FLAGS.do_lower_case)
        # 获取id2char字典
        id2char = tokenizer.inv_vocab

        # TODO 以文件形式预测结果
        while True:
            print('input the test sentence:')
            sentence = str(input())
            start = datetime.now()
            if len(sentence) < 2:
                print(sentence)
                continue
            # print('your input is:{}'.format(sentence))
            input_ids, input_mask, segment_ids, label_ids = convert(sentence)

            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask,
                         segment_ids_p: segment_ids,
                         label_ids_p: label_ids}
            # run session get current feed_dict result
            y_pred = sess.run([pred_ids], feed_dict)

            sent_tag = []
            y_pred_clean = []
            input_ids_clean = []
            # 去除 [CLS] 和 [SEP]获取正确的tag范围
            for index, id in enumerate(list(np.reshape(input_ids, -1))):
                char = id2char[id]
                tag = id2label[list(y_pred[0][0])[index]]
                if char == "[CLS]":
                    continue
                if char == "[SEP]":
                    break
                input_ids_clean.append(id)
                sent_tag.append(tag)
                y_pred_clean.append(list(y_pred[0][0])[index])

            sent_tag = ' '.join(sent_tag)
            print(sentence + '\n' + sent_tag)
            entity = get_entity([sentence], [y_pred_clean], id2label)
            print('predict_result:')
            print(entity)
            print('Time used: {} sec'.format((datetime.now() - start).seconds))


def convert_single_example_dev(ex_index, text, label, label2id, max_seq_length,
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
    O_index = label2id["O"]
    # L: ['B-ORG', 'M-ORG', 'M-ORG', 'M-ORG']
    # W: ['中', '共', '中', '央']
    textlist = text.split(' ')
    labellist = label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        # 对每个字进行tokenize，返回list
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:  # 一般不会出现else
                labels.append("X")

    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    label_ids.append(label2id["[CLS]"])  #
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label2id[labels[i]])
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    label_ids.append(label2id["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # label用O去padding
        label_ids.append(O_index)
        ntokens.append("[PAD]")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


def convert_single_example(example, label2id, max_seq_length, tokenizer):
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
    label_map = label2id
    tokens = tokenizer.tokenize(example)
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


if __name__ == "__main__":
    # dev_texts, dev_labels = zip(*parse_file(args.dev_file))
    # print('dev_texts')
    # print(dev_texts)
    # dev_offline(args.dev_file)

    dev_offline(args.dev_file)
    # if FLAGS.do_predict_outline:
    #     predict_outline()
    # if FLAGS.do_predict_online:
    #     predict_online()

            
            