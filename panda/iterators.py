'''
Created on Jul. 22, 2021

@author: zollen
@url: https://towardsdatascience.com/heres-the-most-efficient-way-to-iterate-through-your-pandas-dataframe-4dad88ac92ee
@title: examining pandas various iteration performance
'''

import pandas as pd
import time

df = pd.read_csv('../data/petfinder-mini.csv')
df.fillna('NA', inplace=True)


'''
iterrows():
Iterate over DataFrame rows as (index, Series) pairs.
'''
kk = df.iterrows()
start_time = time.time()
for idx, row in kk:
    temp = row['Fee'] + row['PhotoAmt']
    
end_time = time.time()

print("PANDAS iterrows() takes  : ", end_time - start_time)


'''
itertuples():
Iterate over DataFrame rows as namedtuples.
'''
kk = df.itertuples()
start_time = time.time()
for row in kk:
    temp = row.Fee + row.PhotoAmt
    
end_time = time.time()

print("PANDAS itertuples() takes: ", end_time - start_time)


'''
Numpy Array Iteration
'''
start_time = time.time()
for row in df.values:
    temp = row[11] + row[13]
    
end_time = time.time()

print("Numpy Iteration takes    : ", end_time - start_time)


'''
Dictionary Iteration
'''
kk = df.to_dict('records')
start_time = time.time()
for row in kk:
    temp = row['Fee'] + row['PhotoAmt']
    
end_time = time.time()

print("Dict Iteration takes     : ", end_time - start_time)


