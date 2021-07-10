'''
Created on Jul. 3, 2021

@author: zollen
'''

import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

test = pd.read_csv('../data/test.csv')
preds = pd.read_csv('../data/prediction.csv')
train = pd.read_csv('../data/monthly_train.csv')


'''
shop_id, item_id, date_block_num, lg2, lg1, item_cnt_month
1              2          0                   1                   
1              2          1                   2
1              2          2         1    2    1                
1              2          3         2    1    1
1              2          4         1    1    2 
1              2          5         1    2    2
1              2          6         2    2    1

https://stackoverflow.com/questions/20410312/how-to-create-a-lagged-data-structure-using-pandas-dataframe
https://rayheberer.medium.com/generating-lagged-pandas-columns-10397309ccaf    
'''
def lag_features(df, trailing_window_size, columns, targets, no_na=True):
    
    df_lagged = df.copy()
   
    for window in range(1, trailing_window_size + 1):
        shifted = df[columns + targets ].groupby(columns).shift(window)
        shifted.columns = [x + "_lag" + str(window) for x in df[targets]]
        df_lagged = pd.concat((df_lagged, shifted), axis=1)
        
    if no_na:
        df_lagged.dropna(inplace=True)
    
    return df_lagged
    
'''
keys = ['shop_id', 'item_id']

tstart = time.time()
t = lag_features(train, 3, keys, ['item_cnt_month'])
tend = time.time()
print(t.head(500))
print("TIME: ", tend - tstart)
'''

k = pd.DataFrame({
    'A': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2], 
    'B': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
t = lag_features(k, 3, ['A'], ['B'], False)
print(t.dropna())
t = lag_features(t, 2, ['A'], ['C'])
t.drop(columns=['C_lag1'], inplace=True)


print(t)