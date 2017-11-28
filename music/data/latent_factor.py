import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import sys

data_path = 'D:\\LiangYiHuai\\kaggle\\music-recommendation-data\\input\\'

train = pd.read_csv(data_path + 'train.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                'source_screen_name' : 'category',
                                                'source_type' : 'category',
                                                'target' : np.uint8,
                                                'song_id' : 'category'})
test = pd.read_csv(data_path + 'test.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                'source_screen_name' : 'category',
                                                'source_type' : 'category',
                                                'song_id' : 'category'})

# train['song_id'] = train['song_id'].map(lambda x: 'song_id'+x);
# test['song_id'] = test['song_id'].map(lambda x: 'song_id'+x);
song_ids = train['song_id'].append(test['song_id'], ignore_index=True)
song_ids = np.unique(song_ids);

member_ids = train['msno'].append(test['msno'], ignore_index=True);
member_ids = np.unique(member_ids)

song_id_map = {};
member_id_map = {};

index = 0;
for key in song_ids:
    song_id_map[key] = index;
    index += 1;

index = 0;
for key in member_ids:
    member_id_map[key] = index;
    index += 1;

print (sys.getsizeof(song_id_map))
print (sys.getsizeof(member_id_map))
print (sys.getsizeof(train))
print (sys.getsizeof(test))

print np.size(song_ids), np.size(member_ids);

# svd_matrix = np.zeros([np.size(song_ids), np.size(member_ids)]);



# for row in train.itertuples():
#     member_id = row[1];
#     song_id = row[2]; # row is a tuple, the index starts from 1.
#     target = row[6];
#     if target > 0: target = 5;
#     svd_matrix[song_id][member_id] = [target];
#
# for row in test.itertuples():
#     member_id = row[2];
#     song_id = row[3];
#     svd_matrix[song_id][member_id] = 0;
#
# print(svd_matrix);








# song_member = np.intersect1d(train['song_id'].unique(), train['msno'].unique())
# print (song_member);






























