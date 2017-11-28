import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split


# df = pd.DataFrame(columns=('a', 'b'), index=("a", 'bb'));
# df.loc[0] = [1, 2];
# df.loc[1] = [3, 4];
# df.loc['a']['b'] = 3;
# print(df);

#
# df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
# df['col1'] = df['col1'].map(lambda x:x + 100);
# print(df)
#
# s1 = pd.Series([1, 2, 3]);
# s2 = pd.Series([1, 4]);
# print(s1.append(s2));
# print(s1);
# print (s2);
#
# df = pd.DataFrame({'col1': [1, 2], 'col2': [0.1, 0.2]}, index=['a', 'b'])
# for row in df.itertuples():
#     print(row[2]);

# df = pd.DataFrame({'a':[2, 3, 4, 4]});
# # df.drop_duplicates(inplace=True);
# print np.unique(df['a']);
# print (df);
# print (df['a']);

# df0 = pd.DataFrame(['a', 'b', 'c']);
#
# for row in df0.itertuples():
#     print row[0]
#
# df1 = pd.DataFrame({'col1':['a', 'b', 'b'], 'col2':['A', 'B', 'B']});
#
# indexes = np.unique(df1['col1']);
# columns = np.unique(df1['col2']);
# print (indexes)
# print (columns)
# print (np.size(indexes));
#
# for i in indexes:
#     print i;
#
# df = pd.DataFrame(index=indexes, columns=columns, dtype=np.uint8);
# print(df)

df = pd.DataFrame({'key1':['a', 'a', 'b', 'b', 'a'],
        'key2':['one', 'two', 'one', 'two', 'one'],
        'key3':['1', '1', '0', '1', '1']})

map = {};

df = df.loc[df['key3'] == '1', ['key1']]
for name, group in df['key1'].groupby(df['key1']):
    print(name, group.count());


# new_series = df.loc[(df['BBB'] > 25) | (df['CCC'] >= -40), ['AAA', 'BBB']];

# print(df[['key1', 'key3']]);
# print(df[['key1', 'key3']].apply(lambda v: v[1]))

# for name, group in df[['key1', 'key3']].groupby(df['key1']):
#     count = group.count();
#     print(name, count);

