#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author： WenRichard
# ide： PyCharm

import sys, pickle, os, random
import numpy as np


## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


def read_corpus(corpus_path, save_tags=False):
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
    # sent_ = [a, b, c, ...]
    # tag_ = [B, M, 0,...]
    # [(sent_, tag_),...]
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    tags = []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()

            sent_.append(char)
            tag_.append(label)
            tags.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    tags_set = list(set(tags))
    tags_dic = {}
    for index, tag in enumerate(tags_set):
        tags_dic[tag] = index

    if save_tags:
        # boson_ner/clue_ner/general_ner/msra_ner/renMinRiBao_ner/resume_ner
        with open('../data/clue_ner/tag2label.txt', 'w', encoding='utf-8') as f:
            for tag, index in tags_dic.items():
                f.write(tag + '\t' + str(index) + '\n')
    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """
    构建词表
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    char2id = {}
    for sent_, tag_ in data:
        for char in sent_:
            if char.isdigit():
                char = '<NUM>'
            elif ('\u0041' <= char <='\u005a') or ('\u0061' <= char <='\u007a'):
                char = '<ENG>'
            if char not in char2id:
                char2id[char] = [len(char2id)+1, 1]
            else:
                char2id[char][1] += 1
    low_freq_chars = []
    for char, [char_id, char_freq] in char2id.items():
        if char_freq < min_count and char != '<NUM>' and char != '<ENG>':
            low_freq_chars.append(char)
    for char in low_freq_chars:
        del char2id[char]

    new_id = 1
    for char in char2id.keys():
        char2id[char] = new_id
        new_id += 1
    char2id['<UNK>'] = new_id
    char2id['<PAD>'] = 0

    print(len(char2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(char2id, fw)


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        char2id = pickle.load(fr)
    print('vocab_size:', len(char2id))
    return char2id


def read_vocab(vocab_path):
    """
    读取词表
    :param vocab_path:
    :return:
    """
    dic = {}
    with open(vocab_path, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
        for index, line in enumerate(lines):
            char = line.strip('\n')
            dic[char] = index
    return dic


def sentence2id(sent, char2id):
    """

    :param sent:
    :param char2id:
    :return:
    """
    sentence_id = []
    for char in sent:
        # if char.isdigit():
        #     char = '<NUM>'
        # elif ('\u0041' <= char <= '\u005a') or ('\u0061' <= char <= '\u007a'):
        #     char = '<ENG>'
        if char not in char2id:
            char = '<UNK>'
        sentence_id.append(char2id[char])
    return sentence_id


def gen_char_embedding(vocab_path, raw_embedding_path, new_embedding_path):
    vector_dim = 768
    embedding_dic = {}
    vocab_ls = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>', '<NUM>', '<ENG>']
    with open(raw_embedding_path, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
        for line in lines:
            line = line.strip('\n').split(' ')
            char = line[0]
            embedding = [float(i) for i in line[1:]]
            # print(embedding)
            if char != ' ':
                embedding_dic[char] = embedding
                vector_dim = len(embedding)
                vocab_ls.append(char)

    print('raw_embedding_len: {}'.format(len(embedding_dic)))
    # print(embedding_dic)
    print('vector_dim: {}'.format(vector_dim))

    if '<PAD>' not in embedding_dic:
        embedding_dic['<PAD>'] = np.zeros((1, vector_dim)).reshape(-1).tolist()
    if '<UNK>' not in embedding_dic:
        embedding_dic['<UNK>'] = np.random.normal(loc=0.0, scale=1, size=vector_dim).reshape(-1).tolist()
    if '<CLS>' not in embedding_dic:
        embedding_dic['<CLS>'] = np.random.normal(loc=0.0, scale=1, size=vector_dim).reshape(-1).tolist()
    if '<SEP>' not in embedding_dic:
        embedding_dic['<SEP>'] = np.random.normal(loc=0.0, scale=1, size=vector_dim).reshape(-1).tolist()
    if '<MASK>' not in embedding_dic:
        embedding_dic['<MASK>'] = np.random.normal(loc=0.0, scale=1, size=vector_dim).reshape(-1).tolist()
    if '<NUM>' not in embedding_dic:
        embedding_dic['<NUM>'] = np.random.normal(loc=0.0, scale=1, size=vector_dim).reshape(-1).tolist()
    if '<ENG>' not in embedding_dic:
        embedding_dic['<ENG>'] = np.random.normal(loc=0.0, scale=1, size=vector_dim).reshape(-1).tolist()

    print('new_embedding_len: {}'.format(len(embedding_dic)))
    # print(embedding_dic)

    with open(vocab_path, 'w', encoding='utf-8') as f2:
        for index, char in enumerate(vocab_ls):
            f2.write(char + '\t' + str(index) + '\n')

    bert_embeddings = []
    for char in vocab_ls:
        bert_embeddings.append(embedding_dic[char])
    # print(bert_embeddings)
    bert_embeddings = np.array(bert_embeddings)
    print(bert_embeddings)
    print(np.shape(bert_embeddings))
    with open(new_embedding_path, 'wb') as f3:
        pickle.dump(bert_embeddings, f3)


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, max_len, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def load_vocab(vocab_path):
    char2id = {}
    id2char = {}
    with open(vocab_path, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
        for line in lines:
            line = line.strip('\n').split('\t')
            char = line[0]
            index = int(line[1])
            char2id[char] = index
            id2char[index] = char
    print('char2id: {}'.format(char2id))
    print('id2char: {}'.format(id2char))
    return char2id, id2char


def load_tag2label(tag2label_path):
    tag2id = {}
    id2tag = {}
    with open(tag2label_path, 'r', encoding='utf-8') as f2:
        lines = f2.readlines()
        for line in lines:
            line = line.strip('\n').split('\t')
            tag = line[0]
            index = int(line[1])
            tag2id[tag] = index
            id2tag[index] = tag
    print('tag2id: {}'.format(tag2id))
    print('id2tag: {}'.format(id2tag))
    return tag2id, id2tag


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


if __name__ == '__main__':
    # resume_ner_corpus_path = '../data/resume_ner/train.char.bmes'
    # clue_ner_corpus_path = '../data/clue_ner/train.txt'
    # read_corpus(clue_ner_corpus_path, save_tags=True)

    # =================
    raw_embedding_path = '../embedding/bert_embedding.txt'
    new_embedding_path = '../embedding/new_bert_embedding.pkl'
    save_path = '../data/vocab_cn.txt'

    gen_char_embedding(save_path, raw_embedding_path, new_embedding_path)

