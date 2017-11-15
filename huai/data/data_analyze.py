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
    with open(TRAIN_DATA) as file:
        is_first_line = True;
        field_names = [];
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

    with open(TRAIN_DATA_FILTER, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',');
        writer.writerow(field_names);
        writer.writerows(temp_lines);


def count_train_data():
    users = {}; # 0
    songs = {}; # 1
    source_system_tabs = {}; # 2
    source_screen_names = {}; # 3
    source_types = {}; # 4
    with open(TRAIN_DATA_FILTER) as file:
        is_first_line = True;
        for line in file:
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


# count_train_data();
# filter_raw_train_data();






