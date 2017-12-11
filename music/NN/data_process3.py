# coding=utf8

import numpy as np;
import pandas as pd;

import xgboost as xgb;


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_onehot_data(median_file_path):
    data_path = median_file_path
    print('Loading data...')
    train = pd.read_csv(data_path + 'median_train.txt', delimiter='\t')
    test = pd.read_csv(data_path + 'median_test.txt', delimiter='\t')

    ids = test.pop('id')
    target = train.pop('target')

    print ("concat")

    concat_data = pd.concat([train, test])

    def float_to_int(x):
        if pd.isnull(x):
            return 0;
        elif type(x) == np.float:
            return int(x)
        else: return x;

    print("unique");
    feature_map = {};
    count = 0;
    feature_nums = [];
    for col in concat_data.columns:
        field_map = {}
        uniqs = concat_data[col].unique();
        uniqs = [float_to_int(i) for i in uniqs]
        feature_nums.append(np.size(uniqs))
        for field in uniqs:
            field_map[field] = count;
            count += 1;
        feature_map[col] = field_map;

    print (feature_nums)

    one_hot_train = pd.DataFrame();
    one_hot_test = pd.DataFrame();

    print('process train data');
    for col in train.columns:
        field_map = feature_map[col];
        temp_arr = [];
        for field in train[col]:
            if not field_map.has_key(field):
                temp_arr.append("0:1")
                continue
            no = field_map[field];
            temp_arr.append(str(no)+":1");
        one_hot_train[col] = np.asarray(temp_arr);

    one_hot_train.insert(0, 'target', target);
    print("save one-hot training data");
    one_hot_train.to_csv(data_path+'one_hot_train.txt', index=False, sep='\t')

    print("process test data")
    for col in test.columns:
        field_map = feature_map[col];
        temp_arr = [];
        for field in test[col]:
            if not field_map.has_key(field):
                temp_arr.append('0:1')
                continue
            no = field_map[field];
            temp_arr.append(str(str(no)+':1'));
        one_hot_test[col] = np.asarray(temp_arr)


    one_hot_test.insert(0, 'id', ids)

    print("save one-hot test data");
    one_hot_test.to_csv(data_path+'one_hot_test.txt', index=False, sep='\t')


