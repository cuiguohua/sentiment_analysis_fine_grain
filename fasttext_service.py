# -*- coding:utf-8 -*-
import tqdm
import os
import argparse
import config
import logging
import numpy as np
from sklearn.externals import joblib
from util import load_data_from_csv, seg_words, get_f1_score, seg_words_multiprocessor

labels_name = ['交通是否便利',
               '距离商圈远近',
               '是否容易寻找',
               '排队等候时间',
               '服务人员态度',
               '是否容易停车',
               '点菜/上菜速度',
               '价格水平',
               '性价比',
               '折扣力度',
               '装修情况',
               '嘈杂情况',
               '就餐空间',
               '卫生状况',
               '菜品分量',
               '口感',
               '外观',
               '推荐程度',
               '本次消费感受',
               '再次消费意愿']

classes_name = ['中性', '正面', '未提及', '负面']

learning_rate = 1.0
epoch = 10
word_ngrams = 3
min_count = 1
model_name = 'fasttext_model_lr{}_e{}_n{}_c{}.pkl'.format(learning_rate, epoch, word_ngrams, min_count)
model_path = config.model_path
print('Loading fasttext model... ')
classifier_dict = joblib.load(model_path + model_name)
print('Load model done! ')


def human_predict(content: str):
    content_test = seg_words([content])
    test_data_format = np.asarray([content_test]).T

    res = []
    for label_ in labels_name:
        type_idx = classifier_dict[label_].predict(
            test_data_format).astype(int)
        #         print(type_idx)
        res.append(classes_name[type_idx[0]])
    return list(zip(labels_name, res))
