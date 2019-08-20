#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import pandas as pd
import tqdm
import multiprocessing
from sklearn.metrics import f1_score

try:
    with open('data/stopword.txt') as f:
        contents = f.readlines()
        stop_words = [c.strip() for c in contents]
except:
    stop_words = []


def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


def seg_words(contents):
    contents_segs = list()
    for content in tqdm.tqdm(contents):
        rcontent = content.replace("\r\n", " ").replace("\n", " ")
        segs = [word for word in jieba.cut(rcontent) if word not in stop_words]
        contents_segs.append(" ".join(segs))
    return contents_segs


def seg_word(__content):
    rcontent = __content.replace("\r\n", " ").replace("\n", " ")
    segs = [word for word in jieba.cut(rcontent) if word not in stop_words]
    return " ".join(segs)


def seg_words_multiprocessor(contents):
    with multiprocessing.Pool() as p:
        contents_segs = list(tqdm.tqdm(p.imap(seg_word, contents), total=len(contents)))
    return contents_segs


def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[1, 0, -1, -2], average='macro')
