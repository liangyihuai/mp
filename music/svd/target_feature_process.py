# coding=utf8

import zipfile
from surprise import Reader, Dataset, SVD, evaluate
import pandas as pd;
import numpy as np;

# Unzip ml-100k.zip
# zipfile = zipfile.ZipFile('D:/LiangYiHuai/kaggle/music-recommendation-data/ml-100k.zip', 'r')
# zipfile.extractall()
# zipfile.close()

u_data = './target_data';
data_path = 'D:\\LiangYiHuai\\kaggle\\music-recommendation-data\\input\\'

# Prepare the data to be used in Surprise
# 'msno song_id target'
print('read data');
reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 10))
data = Dataset.load_from_file(u_data, reader=reader)

# Split the dataset into 5 folds and choose the algorithm
# print('split the dataset');
# data.split(n_folds=5)
algo = SVD(n_factors=100, n_epochs=10000)
#
# # Train and test reporting the RMSE and MAE scores
# print("evaluate");
# evaluate(algo, data, measures=['RMSE'])

# Retrieve the trainset.
print('begin to train');
trainset = data.build_full_trainset()
algo.train(trainset)

# Predict a certain item
# userid = str(196)
# itemid = str(302)
# actual_rating = 4

test = pd.read_csv(data_path + 'test.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                'source_screen_name' : 'category',
                                                'source_type' : 'category',
                                                'song_id' : 'category'})


# Writing output to file
subm = pd.DataFrame()
subm['id'] = test['id'].values

print('predict')
predictions = [];
for index, row in test.iterrows():   # 获取每行的index、row
    temp_user_id = row[test.columns[1]];
    temp_item_id = row[test.columns[2]];
    pre = algo.predict(temp_user_id, temp_item_id)[3] / 10;
    predictions.append(pre);


subm['target'] = np.asarray(predictions);
subm.to_csv('./svd_submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
# subm.to_csv('./svd_submission.txt', index=False, float_format = '%.5f')
print("Done!");