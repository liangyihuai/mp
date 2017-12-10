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

print ("concat")

concat_data = pd.concat([train, test], axis=0)

print("unique");
unique_map = {};
for col in concat_data.columns:
    unique_map[col] = concat_data[col].unique();

print('construct result');
count = 0;
result = [];
for key, value in unique_map.items():
    for ele in value:
        result.append(str(str(key)+'\t&*'+str(ele)+'\t&*'+str(count)+'\n'))
        count += 1;

file_object = open(data_path + 'index_file.txt', 'w')
file_object.writelines(result)
file_object.flush()
file_object.close()

