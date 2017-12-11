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
train = pd.read_csv(data_path + 'median_train.txt', dtype=np.uint32)
test = pd.read_csv(data_path + 'median_test.txt', dtype=np.uint32)

field_nums = [];
for col in test.columns:
    num = test[col].nunique()
    print(col, num);
    field_nums.append(num);

print("change target location");
target = train.pop('target')
train.insert(0, 'target', target)

def to_string_op(x, start_index):
    return str(x+start_index)+':1';

print("apply test elements");
test = test.drop(['id'], axis=1)
one_hot_test = pd.DataFrame()
index = 0;
start_index = 0;
for col in test.columns:
    one_hot_test[col] = test[col].apply(lambda x: to_string_op(x, start_index)).astype(np.str_);
    start_index += field_nums[index]
    index += 1;

print("save");
one_hot_test.to_csv(data_path+'one_hot_test.txt', index=False)

field_nums = [];
train_field_num = 0;
for col in train.columns:
    if col == 'target': continue;
    num = train[col].nunique()
    print(col, num);
    train_field_num += num;
    field_nums.append(num);

print ("train field num = ", train_field_num);

print('Done!')