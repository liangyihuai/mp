import numpy as np;
import pandas as pd;


df = pd.DataFrame({'a':[3, 3, None], 'b':[2, 2, 1], 'c':[5, 6, 8]})

map = {}
for col in df.columns:
    for v in df[col].unique():
        map[v] = 1;

print (map);

np.nan


