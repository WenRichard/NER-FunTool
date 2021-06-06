# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : WenRichard
# @Email : RichardXie1205@gmail.com
# @File : data_trans.py
# @Software: PyCharm


in_file = './test.char.bmes'
out_file = './test.txt'
with open(in_file, 'r', encoding='utf-8') as f1, open(out_file, 'w', encoding='utf-8') as f2:
    for line in f1.readlines():
        s = line.strip('\n').replace(' ', '\t')
        f2.write(s + '\n')


