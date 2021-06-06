#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author： WenRichard
# ide： PyCharm


def clean_bert_vocab(vocab_path, save_path):
    with open(vocab_path, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
        for line in lines:
            if 'unused' in line:
                pass
            else:
                print(line.strip('\n').replace('##', ''))

    with open(save_path, 'w', encoding='utf-8') as f2:
        pass


def extract_bert_vocab(embedding_path, vocab_path):
    vocab_ls = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>', '<NUM>', '<ENG>']
    with open(embedding_path, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
        for line in lines:
            line = line.strip('\n').split(' ')
            char = line[0]
            if char!= ' ':
                vocab_ls.append(char)

    with open(vocab_path, 'w', encoding='utf-8') as f2:
        for char in vocab_ls:
            f2.write(char + '\n')


if __name__ == '__main__':
    # bert_char_path = './vocab.txt'
    # save_path = './vocab_.txt'
    # clean_bert_vocab(bert_char_path, save_path)
    embedding_path = '../embedding/bert_embedding.txt'
    save_path = './vocab_cn.txt'
    extract_bert_vocab(embedding_path, save_path)