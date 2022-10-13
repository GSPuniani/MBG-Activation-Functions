#%%
import os
import re

import pandas as pd

#%%
dfs = {}

for f in os.listdir():
    if not f.endswith('txt'):
        continue
    model = f.split('_')[0]
    count = re.search(r'\d+', f.split('_')[1]).group(0)
    print(model, count)
    df = pd.read_csv(f, skiprows=6, delim_whitespace=True).assign(model=model)
    if count in dfs:
        dfs[count] = pd.concat([dfs[count], df], ignore_index=True)
    else:
        dfs[count] = df

for count, df in dfs.items():
    df.to_csv(f'{count}epochs.csv', index=False)
