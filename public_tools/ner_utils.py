# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : WenRichard
# @Email : RichardXie1205@gmail.com
# @File : ner_utils.py
# @Software: PyCharm


def trans_label(bmeo_labels_id, attr_labels_id, id2bmeo, id2attr):
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
            if id2bmeo[bmeo_id] == "O":
                bmeo_attr = "O"
            else:
                if id2attr[attr_id] == "O":
                    bmeo_attr = "O"
                else:
                    bmeo_attr = id2bmeo[bmeo_id] + "-" + id2attr[attr_id]
            bmeo_attr_label.append(bmeo_attr)
        std_labels.append(bmeo_attr_label)
    # print("std labels")
    # print(std_labels)
    return std_labels


def get_entity(x, y, id2tag):
    entity = ""
    res = []
    for i in range(len(x)):  # for every sen
        for j in range(len(x[i])):  # for every word
            if y[i][j] == 0:
                continue
            if id2tag[y[i][j]][0] == 'B':
                entity = id2tag[y[i][j]][1:]+':'+x[i][j]
            elif id2tag[y[i][j]][0] == 'M' and len(entity) != 0 :
                entity += x[i][j]
            elif id2tag[y[i][j]][0] == 'E' and len(entity) != 0 :
                entity += x[i][j]
                res.append(entity)
                entity = []
            else:
                entity = []
    return res


def get_entity_without_labelid(x, y):
    entity = ""
    res = []
    for i in range(len(x)):  # for every sen
        for j in range(len(x[i])):  # for every word
            if y[i][j] == 0:
                continue
            if y[i][j][0] == 'B':
                entity = y[i][j][1:]+':'+x[i][j]
            elif y[i][j][0] == 'M' and len(entity) != 0 :
                entity += x[i][j]
            elif y[i][j][0] == 'E' and len(entity) != 0 :
                entity += x[i][j]
                res.append(entity)
                entity = []
            else:
                entity = []
    return res


def write_entity(outp, x, y, id2tag):
    '''
    注意，这个函数每次使用是在文档的最后添加新信息。
    '''
    entity = ''
    for i in range(len(x)):
        if y[i] == 0:
            continue
        if id2tag[y[i]][0] == 'B':
            entity = id2tag[y[i]][2:]+':'+x[i]
        elif id2tag[y[i]][0] =='M' and len(entity) != 0:
            entity += x[i]
        elif id2tag[y[i]][0] == 'E' and len(entity) != 0:
            entity += x[i]
            outp.write(entity+' ')
            entity = ''
        else:
            entity = ''
    return


def get_result(sentence, label):
    """
    clue_ner 预测结果脚本
    """
    result_words = []
    result_pos = []
    temp_word = []
    temp_pos = ''
    for i in range(min(len(sentence), len(label))):
        if label[i].startswith('O'):
            if len(temp_word) > 0:
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            temp_word = []
            temp_pos = ''
        elif label[i].startswith('S-'):
            if len(temp_word)>0:
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            result_words.append([i, i])
            result_pos.append(label[i].split('-')[1])
            temp_word = []
            temp_pos = ''
        elif label[i].startswith('B-'):
            if len(temp_word)>0:
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            temp_word = [i]
            temp_pos = label[i].split('-')[1]
        elif label[i].startswith('M-'):
            if len(temp_word)>0:
                temp_word.append(i)
                if temp_pos=='':
                    temp_pos = label[i].split('-')[1]
        else:
            if len(temp_word)>0:
                temp_word.append(i)
                if temp_pos=='':
                    temp_pos = label[i].split('-')[1]
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            temp_word = []
            temp_pos = ''
    return result_words, result_pos



if __name__ == '__main__':
    id2tag = {0: 'O', 1: 'E-ORG', 2: 'E-TITLE', 3: 'E-RACE', 4: 'B-LOC', 5: 'M-CONT', 6: 'M-NAME', 7: 'E-PRO',
             8: 'B-TITLE', 9: 'M-ORG', 10: 'B-ORG', 11: 'B-EDU', 12: 'E-LOC', 13: 'S-NAME', 14: 'E-EDU', 15: 'B-NAME',
             16: 'E-NAME', 17: 'M-TITLE', 18: 'M-EDU', 19: 'E-CONT', 20: 'S-ORG', 21: 'B-PRO', 22: 'B-RACE',
             23: 'M-RACE', 24: 'M-PRO', 25: 'M-LOC', 26: 'S-RACE', 27: 'B-CONT'}

    text = ['中国首都是北京']
    tag = ['B-ORG', 'M-ORG', 'M-ORG', 'E-ORG', 'O', 'M-ORG', 'M-ORG']
    y_pred = [[10,  9,  9,  1,  0,  9,  9]]

    # 黄晨担任交通银行行长
    # B - NAME
    # E - NAME
    # O
    # O
    # B - ORG
    # M - ORG
    # M - ORG
    # E - ORG
    # B - TITLE
    # E - TITLE

    # res = get_entity(text, y_pred, id2tag)
    # print(res)
    result_words, result_pos = get_result('中国首都是北京', tag)
    print(result_words)
    print(result_pos)


