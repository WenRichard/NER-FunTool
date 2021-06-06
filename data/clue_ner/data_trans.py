#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author： WenRichard
# ide： PyCharm

import sys, pickle, os, random
import numpy as np
import json


def read_corpus(corpus_path, output_path):
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
    """
    with open(corpus_path, 'r', encoding='utf-8') as fr, open(output_path, 'w', encoding='utf-8') as fo:
        for line in fr.readlines():
            json_line = json.loads(line.strip())
            """
            {'text': '艺术家也讨厌画廊的老板，内心恨他们，这样的话，你是在这样的状态下，两年都是一次性合作，甚至两年、', 
            'label': {'position': {'艺术家': [[0, 2]], '老板': [[9, 10]]}}}
            """
            text_a = json_line['text']

            label = ['O'] * len(text_a)
            if 'label' in json_line:
                for attr, words in json_line['label'].items():
                    for word, indices in words.items():
                        for index in indices:
                            if index[0] == index[1]:
                                label[index[0]] = 'S-' + attr
                            else:
                                label[index[0]] = 'B-' + attr
                                label[index[1]] = 'E-' + attr
                                for i in range(index[0] + 1, index[1]):
                                    label[i] = 'M-' + attr

            for index, char in enumerate(text_a):
                fo.write(char + '\t' + label[index] + '\n')
            fo.write('\n')




if __name__ == '__main__':
    json_path_train = 'train.json'
    json_path_train_output = 'train.txt'
    json_path_dev = 'dev.json'
    json_path_dev_output = 'dev.txt'
    json_path_test = 'test.json'
    json_path_test_output = 'test.txt'

    read_corpus(json_path_train, json_path_train_output)
    read_corpus(json_path_dev, json_path_dev_output)
    read_corpus(json_path_test, json_path_test_output)


