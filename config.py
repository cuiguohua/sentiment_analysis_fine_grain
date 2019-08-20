#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

data_path = os.path.abspath('.') + "/data"
model_path = data_path + "/model/"
train_data_path = data_path + "/train/sentiment_analysis_trainingset.csv"
validate_data_path = data_path + "/valid/sentiment_analysis_validationset.csv"
test_data_path = data_path + "/test/sentiment_analysis_testa.csv"
test_data_predict_output_path = data_path + "/predict/testa_predict.csv"
