
import data_process as process1;

import data_process3 as process3;

data_path = 'D:\\LiangYiHuai\\kaggle\\music-recommendation-data\\input\\'

print('----------begin to generate the feature from original data set-------');
process1.generate_features(data_path);
print('---------begin to generate one hot data ----------');
process3.generate_onehot_data(data_path);