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


#
# print("apply train elements");
# one_hot_train = pd.DataFrame();
# index = 0;
# start_index = 0;
# for col in train.columns:
#     if col == 'target':
#         one_hot_train[col] = train[col];
#         continue;
#     one_hot_train[col] = train[col].apply(lambda x: to_string_op(x, start_index)).astype(np.str_)
#     start_index += field_nums[index]
#     index += 1;
#
# print("save");
# one_hot_train.to_csv(data_path + 'one_hot_train.txt', index=False)



#
# random_seed = 12;
#
# X_train = train.drop(['target'], axis=1)
# y_train = train['target'].values
#
# # X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, random_state=0, shuffle=True)
# # X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, shuffle=True)
#
# #
# X_test = test.drop(['id'], axis=1)
# ids = test['id'].values
#
# # del train, test; gc.collect();
# # lgb_train = lgb.Dataset(X_tr, y_tr)
# # lgb_val = lgb.Dataset(X_val, y_val)
# print('Processed data...')
#
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=1729)
#
# params = {
#     'booster': 'gbtree',  # gbtree used
#     'objective': 'binary:logistic',
#     'early_stopping_rounds': 50,
#     'scale_pos_weight': 0.63,  # 正样本权重
#     'eval_metric': 'auc',
#     'gamma': 0,
#     'max_depth': 5,
#     # 'lambda': 550,
#     'subsample': 0.6,
#     'colsample_bytree': 0.9,
#     'min_child_weight': 1,
#     'eta': 0.2,
#     'seed': random_seed,
#     'nthread': 3,
#     'silent': 0
# }
#
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dval = xgb.DMatrix(X_val, label=y_val)
# watchlist = [(dval, 'val'), (dtrain, 'train')]
# model = xgb.train(params, dtrain, num_boost_round=1, evals=watchlist)
#
# # 对测试集进行预测（以上部分之所以划分成验证集，可以用来调参）
# predictions = model.predict(X_test, ntree_limit=model.best_ntree_limit)  # 预测结果为概率
#
# # 输出特征重要性
# print(model.get_fscore())
#
# subm = pd.DataFrame()
# subm['id'] = ids
# subm['target'] = predictions
# subm.to_csv(data_path + 'lgbm_submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
#
print('Done!')

######