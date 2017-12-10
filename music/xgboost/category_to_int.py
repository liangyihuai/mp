import pandas as pd;
import numpy as np;

# If you don't want to modify your DataFrame but simply get the codes:
#
# df.cc.astype('category').cat.codes
# Or use the categorical column as an index:
#
# df2 = pd.DataFrame(df.temp)
# df2.index = pd.CategoricalIndex(df.cc)

df = pd.DataFrame({'A':['a', 'b', 'c']}, dtype='category');
print(df);
df['code'] = df.A.cat.codes;
print (df);

