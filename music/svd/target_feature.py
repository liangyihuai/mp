import numpy as np;
import pandas as pd;

data_path = 'D:\\LiangYiHuai\\kaggle\\music-recommendation-data\\input\\'
print('Loading data...')
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


def change_target_op(target):
    if target == 1:
        return 10;
    else:
        return 1;


train['target'] = train['target'].apply(change_target_op);

test.drop(test['id'], inplace=True)

test_target = pd.Series([0 for i in range(np.size(train['msno']))], dtype=np.uint8);
test['target'] = test_target;

concat_result = pd.concat([train, test], ignore_index=True)

# sp_g = lambda x: split_genres(x, i)
# songs['genre_' + str(i)] = songs['genre_ids'].apply(sp_g)

concat_result.to_csv('./target_data', index=False, header=False, columns=['msno', 'song_id', 'target']);

