import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

data_path = 'D:\\LiangYiHuai\\kaggle\\music-recommendation-data\\input\\'

train = pd.read_csv(data_path + 'train.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                'source_screen_name' : 'category',
                                                'source_type' : 'category',
                                                'target' : np.uint8,
                                                'song_id' : 'category'})

# users_in_train_and_test = np.intersect1d(train['msno'].unique(), train['msno'].unique())
print(np.size(train['msno']), np.size(train['msno'].unique()))
print(np.size(train['song_id']), np.size(train['song_id'].unique()))