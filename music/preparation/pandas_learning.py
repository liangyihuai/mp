# coding=utf8

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as splt;
from functools import reduce;


def frame():
    s = pd.Series([1, 3, 5, np.nan, 6, 8]);
    print(s);

    dates = pd.date_range('20170102', periods=6)
    print(dates);

    df = pd.DataFrame(np.random.randn(6, 4),
                      index=dates, columns=list('ABCD'))
    print(df);

    df2 = pd.DataFrame({
        'A': 1.,
        'B':pd.Timestamp('20130102'),
        'C':pd.Series(1, index=list(range(4)), dtype='float32'),
        'D':np.array([3]*4, dtype='int32'),
        'E':pd.Categorical(['test', 'train', 'test', 'train']),
        'F':'foo'});
    print(df2);
    print(df2.dtypes);


    print(df.head());
    print(df.tail());
    print(df.index);
    print(df.columns);
    print(df.values);
    print(df.describe());
    print(df.T);
    # 按照第二坐标轴（列）的大小降序排序
    print(df.sort_index(axis=1, ascending=False))
    print(df.sort_values(by=['B'], ascending=True, inplace=False));
    print(df.sort_values(by=['A', 'B'], ascending=[0, 1], inplace=False));
    print(df['A']);
    print(df[0:2]);
    print(df['20170102':'20170104'])
    print(df.loc(dates[0]));

    print('-------------------');
    df3 = pd.DataFrame([[1,2,3], [1, 3, 4], [2, 4, 3]],
                       index=['one', 'two', 'three'],
                       columns=['A', 'B', 'C']);
    print(df3);
    print(df3['A'].isin([1]));
    print(df3.sort_values(by=['A', 'B'], ascending=[0, 1]));

def series_test():
    data = np.array(['a', 'b', 'c'])
    s = pd.Series(data);
    print(s)

    data = np.array(['a', 'b', 'c'])
    s = pd.Series(data, index=[100, 101, 102]);
    print(s)

    data = {'a':0., 'b':1., 'c':2.}
    s = pd.Series(data);
    print(s)

    data = {'a':0., 'b':1., 'c':2.}
    s = pd.Series(data, index=['b', 'c', 'd', 'a']);
    print(s)


    s = pd.Series(5, index=['1', '2', '3'])
    print(s)

    print(s[0])

    s = pd.Series(np.arange(5), index=[1, 2, 3, 4, 5])
    print(s)
    print(s[:3])
    print(s[-3:])
    print(s[1])
    print(s[[1, 2, 3]])


def frame_learning():
    df = pd.DataFrame();
    print(df)

    df = pd.DataFrame([1, 2, 3, 4])
    print(df)

    data = [['alex', 10], ['bob', 12], ['clarke', 13]]
    df = pd.DataFrame(data, columns=['name', 'age'], dtype=float)
    print(df)

    data = {'name':['Tom', 'jack', 'steve'], 'age':[28, 34, 29]}
    df = pd.DataFrame(data, index=['rank1', 'rank2', 'rank3']);
    print(df)


    data = [{'a':1, 'b':2}, {'a':5, 'b':10, 'c':20}]
    df = pd.DataFrame(data, index=['first', 'second']);
    print(df)

    data = [{'a':1, 'b':2}, {'a':5, 'b':10, 'c':20}]
    df = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b']);
    print(df)

    data = [{'a':1, 'b':2}, {'a':5, 'b':10, 'c':20}]
    df = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b1']);
    print(df)

    data = {'one':pd.Series([1, 2, 3], index=['a', 'b', 'c']),
            'two':pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
    df = pd.DataFrame(data);
    print(df)
    print(df['two'])
    df['three'] = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
    print(df)
    print ("Adding a new column by passing as Series:")
    df['four'] = df['one'] + df['three'];
    print(df)

    del(df['one']) # delete a column
    print(df)
    df.pop('two')  # delete a column
    print(df)

    print('-------------')
    d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
         'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
    df = pd.DataFrame(d);
    print(df)
    print(df.loc['b'])
    print(df.iloc[2])

    d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
         'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
    df = pd.DataFrame(d);
    print(df[2:4])

    df = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
    df2 = pd.DataFrame([[5, 6], [7, 8]], columns=['a', 'b'])
    df = df.append(df2)
    print(df)
    print('---------')
    df = df.drop(0) # 0 is label name.
    print(df)


def series_basic_functionality():
    data = np.random.rand(2, 4, 5);
    p = pd.Panel(data)
    print(p)


    data = {'item1': pd.DataFrame(np.random.randn(4, 3)),
            'item2': pd.DataFrame(np.random.randn(4, 2))}
    p = pd.Panel(data);
    print(p)

    s = pd.Series(np.random.randn(4));
    print(s)
    print(s.axes)
    print(s.empty)
    print(s.ndim)
    print(s.size)
    print(s.values)
    print(s.tail(2))
    print(s.head(2))


def frame_basic_functionality():
    # frame basic functionality
    d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
       'Age':pd.Series([25,26,25,23,30,29,23]),
       'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}
    df = pd.DataFrame(d);
    print(df)
    print(df.T)
    print(df)
    print(df.axes)
    print(df.dtypes)
    print(df.empty)
    print(df.ndim)
    print(df.shape)
    print(df.size)
    print(df.values)
    print(df.head(2))
    print(df.tail(2))

def statistics():
    #statistics
    d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
       'Lee','David','Gasper','Betina','Andres']),
       'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
       'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])}
    df = pd.DataFrame(d);
    print(df)
    print(df.sum())
    print(df.sum(1))
    print(df.mean())
    print(df.std())


def adder(ele1, ele2):
    return ele1 + ele2;

df = pd.DataFrame(np.random.randn(5, 3),
                  columns=['col1', 'col2', 'col3'])

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df['col1'].map(lambda x:x + 100);
print(df)

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.applymap(lambda x: x*100);
print(df)


data = np.arange(0, 16).reshape(4, 4);
data = pd.DataFrame(data, columns=['0', '1', '2', '3'])

def f(x):
    return x-1;

def test0():
    print(data)
    print(data.apply(lambda x: x- 2))
    print(data.ix[:, ['0', '1']].apply(f))

    print(data.ix[[0, 1], :].apply(f))
    print(data.apply(f))
    print(data.apply(f, axis=1))

    df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
    print(df['col1'].apply(lambda x: x+100))

    df = pd.DataFrame(
        {'AAA': [4, 5, 6, 7], 'BBB': [10, 20, 30, 40], 'CCC': [100, 50, -30, -50]})
    print(df)

    print(df.loc[df.AAA >= 5, 'BBB'])
    df.loc[df.AAA >= 5, ['BBB', 'CCC']] = 500
    print(df)

    df.loc[df.AAA < 5, ['BBB', 'CCC']] = 2000;
    print(df)

    df_mask = pd.DataFrame({'AAA':[True] * 4, 'BBB':[False]*4, 'CCC':[True, False]*2})

    print(df.where(df_mask, -1000))
    df = pd.DataFrame(
       {'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]});
    print(df)
    df['logic'] = np.where(df['AAA'] > 5, 'high', 'low');
    print(df)

    df = pd.DataFrame(
         {'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]});
    print(df)
    print(df[df.AAA <= 5])
    print(df[df.AAA > 5])
    print("------")
    new_series = df.loc[(df['BBB'] < 25) & (df['CCC'] >= 40), 'AAA']
    print(df)
    print(new_series)

    new_series = df.loc[(df['BBB'] > 25) | (df['CCC'] >= -40), ['AAA', 'BBB']];
    print(new_series)
    df.loc[(df['BBB'] > 25) | (df['CCC'] >= 75), 'AAA'] = 0.1;
    print(df)

    df = pd.DataFrame(
         {'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]});
    print(df)

    print(df.loc[(df.CCC - 43.0).abs().argsort()])


def apply_test():
    frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),
                      index=['Utah', 'Ohio', 'Texas', 'Oregon']);
    print(frame);
    f = lambda x: x.max() - x.min();
    print(frame.apply(f))
    print(frame.apply(f, axis=1));
    format = lambda x: '%.2f'%x;
    # 如果想让方程作用于DataFrame中的每一个元素，可以使用applymap().
    print(frame.applymap(format))

    # map()只要是作用将函数作用于一个Series的每一个元素
    print(frame['e'].map(format))

    #总的来说就是apply()是一种让函数作用于列或者行操作，
    # applymap()是一种让函数作用于DataFrame每一个元素的操作，
    # 而map是一种让函数作用于Series每一个元素的操作

def loc_test():
    df = pd.DataFrame(
        {'AAA': [4, 5, 6, 7], 'BBB':[10, 20, 30, 40], 'CCC':[100, 50, -30, -50]})

    print(df);

    crit1 = df.AAA <= 5.5;
    crit2 = df.BBB == 10.0;
    crit3 = df.CCC > -40.0;

    all_crit = crit1 & crit2 & crit3;
    crit_list = [crit1, crit2, crit3];
    all_crit = reduce(lambda x, y: x & y, crit_list);
    print('all-crit')
    print(all_crit)

    print(df[all_crit])

    df = pd.DataFrame(
       {'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]});
    print(df)
    print(df[(df.AAA <= 6) & (df.index.isin([0, 2, 4]))])
    new_series = df.loc[(df.AAA <= 6) & (df.index.isin([0, 2, 4]))];
    print(new_series);
    print(df)

    data = {'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]}
    df = pd.DataFrame(data=data,index=['foo','bar','boo','kar']);
    print(df.loc['bar': 'kar']) # label oriented


    print(df.iloc[0: 1]) # position oriented
    print(df[0: 1]) # position
    print(df['bar': 'kar']) # label

    df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]})
    print(df)
    print(df[~((df.AAA <= 6) & (df.index.isin([0, 2, 4])))])


def column_test():
    # column test
    df = pd.DataFrame(
         {'AAA' : [1,2,1,3], 'BBB':[1, 1, 2, 2], 'CCC':[2,1,3,1]});
    print(df)
    source_cols = df.columns
    new_cols = [str(x) + '_cat' for x in source_cols]
    categories = {1: 'Alpha', 2: 'Beta', 3: 'Charlie'}
    df[new_cols] = df[source_cols].applymap(categories.get);
    print(df)

    df = pd.DataFrame(
       {'AAA' : [1,1,1,2,2,2,3,3], 'BBB' : [2,1,3,4,5,1,2,3]});
    print(df.loc[df.groupby('AAA')['BBB'].idxmin()])
    # Notice the same results, with the exception of the index.
    print(df.sort_values(by='BBB').groupby('AAA', as_index=False).first())


    df = pd.DataFrame({'key1':['a', 'a', 'b', 'b', 'a'],
        'key2':['one', 'two', 'one', 'two', 'one'],
        'data1':np.random.randn(5),
        'data2':np.random.randn(5)})

    print(df)

    grouped = df['data1'].groupby(df['key1'])
    print(grouped.mean())
    print(grouped.first())
    print(df.groupby(['key1', 'key2']).size())


# multi indexing

df = pd.DataFrame({'row' : [0,1,2],
                    'One_X' : [1.1,1.1,1.1],
                    'One_Y' : [1.2,1.2,1.2],
                    'Two_X' : [1.11,1.11,1.11],
                    'Two_Y' : [1.22,1.22,1.22]});

df = df.set_index('row')
print(df)
df.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in df.columns])
print(df)