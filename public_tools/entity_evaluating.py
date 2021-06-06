# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : WenRichard
# @Email : RichardXie1205@gmail.com
# @File : entity_evaluating.py
# @Software: PyCharm

import re
import numpy as np


def calculate(x, y, id2word, id2tag, res = []):
    """
    y: 是label id
    """
    entity=[]
    for i in range(len(x)): #for every sen
        for j in range(len(x[i])): #for every word
            if x[i][j]==0 or y[i][j]==0:
                continue
            if id2tag[y[i][j]][0]=='B':
                entity=[id2word[x[i][j]]+'/'+id2tag[y[i][j]]]
            elif id2tag[y[i][j]][0]=='M' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]]+'/'+id2tag[y[i][j]])
            elif id2tag[y[i][j]][0]=='E' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]]+'/'+id2tag[y[i][j]])
                entity.append(str(i))
                entity.append(str(j))
                res.append(entity)
                entity=[]
            else:
                entity=[]
    return res


def calculate_without_lableid(x, y, id2word, res = []):
    """
    y: 不是label id
    """
    entity=[]
    for i in range(len(x)): #for every sen
        for j in range(len(x[i])): #for every word
            if x[i][j]==0 or y[i][j]==0:
                continue
            if y[i][j][0]=='B':
                entity=[id2word[x[i][j]]+'/'+y[i][j]]
            elif y[i][j][0]=='M' and len(entity)!=0 and entity[-1].split('/')[1][1:]==y[i][j][1:]:
                entity.append(id2word[x[i][j]]+'/'+y[i][j])
            elif y[i][j][0]=='E' and len(entity)!=0 and entity[-1].split('/')[1][1:]==y[i][j][1:]:
                entity.append(id2word[x[i][j]]+'/'+y[i][j])
                entity.append(str(i))
                entity.append(str(j))
                res.append(entity)
                entity=[]
            else:
                entity=[]
    return res


def entity_metrics(x, y_true, y_pred, id2word, id2tag):
    """
    计算以实体为单位预测的 precision， recall， f1值
    :param x:
    :param y_true:
    :param y_pred:
    :param id2word:
    :param id2tag:
    :return:
    """
    entity_pred = []
    entity_true = []
    entity_pred = calculate(x, y_pred, id2word, id2tag, entity_pred)
    entity_true = calculate(x, y_true, id2word, id2tag, entity_true)
    entity_intersection = [i for i in entity_pred if i in entity_true]
    if len(entity_intersection) != 0:
        precision = float(len(entity_intersection))/len(entity_pred)
        recall = float(len(entity_intersection))/len(entity_true)
        f1 = (2*precision*recall)/(precision+recall)
        return precision, recall, f1
    else:
        return 0, 0, 0


def entity_metrics_without_lableid(x, y_true, y_pred, id2word):
    """
    计算以实体为单位预测的 precision， recall， f1值
    :param x:
    :param y_true:
    :param y_pred:
    :param id2word:
    :param id2tag:
    :return:
    """
    entity_pred = []
    entity_true = []
    entity_pred = calculate_without_lableid(x, y_pred, id2word, entity_pred)
    entity_true = calculate_without_lableid(x, y_true, id2word, entity_true)
    entity_intersection = [i for i in entity_pred if i in entity_true]
    if len(entity_intersection) != 0:
        precision = float(len(entity_intersection))/len(entity_pred)
        recall = float(len(entity_intersection))/len(entity_true)
        f1 = (2*precision*recall)/(precision+recall)
        return precision, recall, f1
    else:
        return 0, 0, 0






