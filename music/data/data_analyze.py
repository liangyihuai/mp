import numpy as np;
import os;
import csv;

TRAIN_DATA = 'D:\\LiangYiHuai\\kaggle\\music-recommendation-data\\train.csv\\train.csv';
TRAIN_DATA_FILTER = 'D:\\LiangYiHuai\\kaggle\\music-recommendation-data\\train.csv\\filter_train.csv';
# TEST_DATA = 'D:\\LiangYiHuai\\kaggle\\music-recommendation-data\\test.csv\\test.csv';
# TEST_DATA_FILTER = 'D:\\LiangYiHuai\\kaggle\\music-recommendation-data\\test.csv\\filter_train.csv';


def filter_raw_train_data():
    '''extract the valid data, and save it to the hard disk'''
    temp_lines = [];
    field_names = [];
    with open(TRAIN_DATA) as file:
        is_first_line = True;

        for line in file:
            if is_first_line:
                is_first_line = False;
                field_names = line.split(',');
                continue;
            fields = line.split(',');

            if len(fields) < 6: continue;
            is_this_line_is_good = True;
            for i in range(len(fields)):
                if (i == 0 or i == 1 or i == 5) and len(fields[i].strip()) < 1: # the field can not be blank.
                    is_this_line_is_good = False;
                    break;

            if not is_this_line_is_good:continue;

            temp_lines.append(fields);

    with open(TRAIN_DATA_FILTER, 'w') as file:
        ','.join(field_names)
        file.write(line);
        for fields in temp_lines:
            line = ','.join(fields);
            file.write(line);



def count_train_data():
    users, songs, source_system_tabs, source_screen_names, source_types = read_train_in_dict();
    print('user number is: %s'% len(users.keys()));
    count = 0;
    for num in users.values(): count += num;
    print('user appear average number of times is: %f'%(count / len(users.keys())))

    print('song number is: %s' % len(songs.keys()));
    count = 0;
    for num in songs.values(): count += num;
    print('song appear average number of times is: %f'%(count / len(songs.keys())));

    for key, value in source_system_tabs.items():
        print('source_system_tabs id: %s, count: %s' % (key, value));
    for key, value in source_screen_names.items():
        print('source_screen_names id: %s, count: %s' % (key, value));
    for key, value in source_types.items():
        print('source_types id: %s, count: %s' % (key, value));

def read_train_in_dict():
    users = {};  # 0
    songs = {};  # 1
    source_system_tabs = {};  # 2
    source_screen_names = {};  # 3
    source_types = {};  # 4
    with open(TRAIN_DATA_FILTER) as file:
        is_first_line = True;
        while True:
            line = file.readline();
            if not line: break;
            if is_first_line:
                is_first_line = False;
                continue;
            fields = line.split(',');

            if len(fields) < 6: continue;

            field = fields[0];
            if field in users.keys():
                value = users[field] + 1;
                users[field] = value;
            else:
                users[field] = 1;

            field = fields[1];
            if field in songs.keys():
                value = songs[field] + 1;
                songs[field] = value;
            else:
                songs[field] = 1;

            field = fields[2];
            if field in source_system_tabs.keys():
                value = source_system_tabs[field] + 1;
                source_system_tabs[field] = value;
            else:
                source_system_tabs[field] = 1;

            field = fields[3];
            if field in source_screen_names.keys():
                value = source_screen_names[field] + 1;
                source_screen_names[field] = value;
            else:
                source_screen_names[field] = 1;

            field = fields[4];
            if field in source_types.keys():
                value = source_types[field] + 1;
                source_types[field] = value;
            else:
                source_types[field] = 1;
    return (users, songs, source_system_tabs, source_screen_names, source_types);


TEST_DATA_PATH = 'D:\\LiangYiHuai\\kaggle\\music-recommendation-data\\test.csv\\test.csv';


def read_testing_data_into_dict():
    users = {};  # 1
    songs = {};  # 2
    source_system_tabs = {};  # 3
    source_screen_names = {};  # 4
    source_types = {};  # 5
    with open(TRAIN_DATA_FILTER) as file:
        is_first_line = True;
        while True:
            line = file.readline();
            if not line: break;
            if is_first_line:
                is_first_line = False;
                continue;
            fields = line.split(',');

            if len(fields) < 6: continue;

            index = 1;
            field = fields[index];
            if field in users.keys():
                field = field.strip();
                value = users[field] + 1;
                users[field] = value;
            else:
                users[field] = 1;

            index += 1;
            field = fields[index];
            if field in songs.keys():
                value = songs[field] + 1;
                songs[field] = value;
            else:
                songs[field] = 1;

            index += 1;
            field = fields[index];
            if field in source_system_tabs.keys():
                value = source_system_tabs[field] + 1;
                source_system_tabs[field] = value;
            else:
                source_system_tabs[field] = 1;

            index += 1;
            field = fields[index];
            if field in source_screen_names.keys():
                value = source_screen_names[field] + 1;
                source_screen_names[field] = value;
            else:
                source_screen_names[field] = 1;

            index += 1;
            field = fields[index];
            if field in source_types.keys():
                value = source_types[field] + 1;
                source_types[field] = value;
            else:
                source_types[field] = 1;
    return (users, songs, source_system_tabs, source_screen_names, source_types);


def get_song_not_in_training_data():
    train_users, train_songs, _, _, _ = read_train_in_dict();
    test_users, test_songs, _, _, _ = read_testing_data_into_dict();

    count_user_not_in_training_data = 0;
    for test_user_name in test_users.keys():
        if test_user_name not in train_users.keys():
            count_user_not_in_training_data += 1;
    print("count_user_not_in_training_data", count_user_not_in_training_data);

    count_song_not_in_training_data = 0;
    for test_song_name in test_songs.keys():
        if test_song_name not in train_songs.keys():
            count_song_not_in_training_data += 1;
    print("count_song_not_in_training_data", count_song_not_in_training_data);


def count_user_for_each_song():
    '''map<userID, map<songID, count>>, the target is 1'''
    result = {};
    with open(TRAIN_DATA_FILTER) as file:
        is_first_line = True;
        while True:
            line = file.readline();
            if not line: break;

            if is_first_line:
                is_first_line = False;
                continue;
            fields = line.split(',');
            if len(fields) < 6: continue;

            user = fields[0];
            song = fields[1];
            target = fields[5].rstrip();

            if target == '0' or target == 0: continue;

            if user not in result:
                result[user] = {};
                result[user][song] = 1;
            else:
                temp_songs = result[user];
                if song not in temp_songs:
                    temp_songs[song] = 1;
                else:
                    temp_count = temp_songs[song] + 1;
                    temp_songs[song] = temp_count;

    for user_name, songs in result.items():

        # print("----------user_id: ", user_name, "------------");
        for song_name, count in songs.items():
            # print("song name: %s, count: %s"%(song_name, count));
            if count > 1:
                print("user id: %s, song name: %"%(user_name, song_name));


# count_train_data();
# filter_raw_train_data();
# get_song_not_in_training_data();
count_user_for_each_song()















