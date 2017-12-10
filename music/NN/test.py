import numpy as np;
import pandas as pd;


df = pd.DataFrame(np.random.rand(5,3),columns=['col1','col2','col3'])
df = df.applymap(lambda x: x*100);
print(df)

col2 = df.pop('col2')
df.insert(0, 'col2', col2)
print(df);