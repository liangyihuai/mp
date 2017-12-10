# coding=utf8

import numpy as np;
import pandas as pd;

import xgboost as xgb;


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# data_path = 'F:/Pusan/kaggle/music-recommendation-data/input/'
data_path = 'D:\\LiangYiHuai\\kaggle\\music-recommendation-data\\input\\'
print('Loading data...')
train = pd.read_csv(data_path + 'median_train.txt')
test = pd.read_csv(data_path + 'median_test.txt')

test = test.drop(['id'], axis=1)
target = train.pop('target')

file_object = open(data_path + 'index_file.txt', 'r')
feature_map = {};
for line in file_object.readlines():
    fields1 = line.rstrip().split('\t&*')
    if(np.size(fields1) < 3): continue;
    feature = fields1[0];
    field = fields1[1];
    no = fields1[2];
    if not feature_map.has_key(feature):
        field_map = {};
        field_map[field] = no;
        feature_map[feature]= field_map;
    else:
        feature_map[feature][field] = no;


for col in train.columns:
    field_map = feature_map[col];
    row_index = 0;
    for field in train[col]:
        if not field_map.has_key(field):
            train[col][row_index] = str(str(1) + ":1");
            continue;
        no = field_map[field];
        train[col][row_index] = str(no)+":1";
        row_index += 1;

train.insert(0, 'target', target);

for col in test.columns:
    field_map = feature_map[col];
    row_index = 0;
    for field in test[col]:
        if not field_map.has_key(field):
            test[col][row_index] = str(str(1) + ":1");
            continue;
        no = field_map[field];
        test[col][row_index] = str(str(no)+':1');
        row_index += 1;


test.to_csv(data_path+'one_hot_test.txt', index=False)
train.to_csv(data_path+'one_hot_train.txt', index=False)

