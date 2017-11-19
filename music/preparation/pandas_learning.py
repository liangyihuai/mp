import pandas as pd;
import numpy as np;
import matplotlib.pyplot as splt;

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






