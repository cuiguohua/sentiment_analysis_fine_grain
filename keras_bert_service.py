# -*- coding:utf-8 -*-

from keras import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Flatten
from sklearn import metrics
import numpy as np
from bert_serving.client import BertClient

NUM_CLASSES = 80
TRAIN_FILE_PATH = 'TEXT_DIR/t.tsv'
VALID_FILE_PATH = 'TEXT_DIR/dev.tsv'
TEST_FILE_PATH = 'TEXT_DIR/test.tsv'
NUM_ASPECTS = 20
INPUT_DIM = 768
MODEL_PATH = 'bert_service_model/model.h5'


def get_labels():
    """See base class."""
    label_list = []
    # num_aspect=FLAGS.num_aspects
    aspect_value_list = [-2, -1, 0, 1]
    for i in range(NUM_ASPECTS):
        for value in aspect_value_list:
            label_list.append(str(i) + "_" + str(value))
    return label_list


def get_model():
    model = Sequential()
    # print("create model. feature_dim = %s, label_dim = %s" % (self.feature_dim, self.label_dim))
    model.add(Dense(500, activation='relu', input_dim=INPUT_DIM))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_data(file_path):
    label_list = get_labels()
    y = []
    X = []
    with open(file_path) as f:
        for line in f.readlines():
            l = line.split('\t')
            label_with_comma = l[0]
            label_ids = [0] * NUM_CLASSES
            labels = label_with_comma.split(',')
            for label_ in labels:
                label_ids[label_list.index(label_)] = 1

            feature = l[1]
            X.append(feature)
            y.append(label_ids)
    y = np.array(y, dtype=int)
    return X, y


def encode_content(X):
    from bert_serving.client import BertClient
    bc = BertClient()  # ip address of the GPU machine
    X_encode = bc.encode(X)
    return X_encode


# X_train, y_train = get_data(TRAIN_FILE_PATH)
# X_train_encode = encode_content(X_train)


def train_model(X, y, **kwargs):
    model = get_model()
    model.fit(X, y, **kwargs)
    model.save(MODEL_PATH)
    return model


# model = train_model(X_train_encode, y_train, batch_size=4, epochs=10, verbose=2)


def predict_model(model_, X):
    res = model_.predict(X)
    res[res > 0.5] = 1
    res[res <= 0.5] = 0
    return res.astype('int64')


def multi_label_accuracy(y_true, y_pred):
    acc_list = []
    for i in range(NUM_CLASSES):
        acc = metrics.accuracy_score(y_true[:, i], y_pred[:, i])
        acc_list.append(acc)
    return sum(acc_list) / NUM_CLASSES


# multi_label_accuracy(y_train[:100], predict_model(model, X_train_encode[:100]))

# X_valid, y_valid = get_data(VALID_FILE_PATH)
# X_valid_encode = encode_content(X_valid)

# y_valid_pred = predict_model(model, X_valid_encode)

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

classes_name = ['中性', '积极', '未提到', '消极']


def label_id_to_label(y):
    labels_list = get_labels()
    labels = []
    for i in range(NUM_CLASSES):
        if y[i] == 1:
            labels.append(labels_list[i])
    res = ['Not mentioned'] * NUM_ASPECTS
    for label in labels:
        l = label.split('_')
        aspect_id = int(l[0])
        aspect_type = int(l[1])
        res[aspect_id] = classes_name[aspect_type]
        # res[labels_name[aspect_id]] = classes_name[aspect_type]
    return list(zip(labels_name, res))


def human_predict(content: str):
    print(content)
    X = [content]
    print(X)
    model = load_model(MODEL_PATH)
    bc = BertClient()
    X_encode = bc.encode(X)
    print(X_encode.shape)
    y_pred = predict_model(model, X_encode)
    return label_id_to_label(y_pred[0])


