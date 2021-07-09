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


kk = pd.merge(test, preds, how='inner')
res = kk.loc[kk['item_cnt_month'] < 0, ['shop_id', 'item_id', 'item_cnt_month']]

'''
shop_id, item_id, date_block_num, lg2, lg1, item_cnt_month
1              2          0                   1                   
1              2          1                   2
1              2          2         1    2    1                
1              2          3         2    1    1
1              2          4         1    1    2 
1              2          5         1    2    2
1              2          6         2    2    1


start
https://stackoverflow.com/questions/33907537/groupby-and-lag-all-columns-of-a-dataframe
need more

https://stackoverflow.com/questions/20410312/how-to-create-a-lagged-data-structure-using-pandas-dataframe
https://rayheberer.medium.com/generating-lagged-pandas-columns-10397309ccaf    
'''
def lag_feature(df, trailing_window_size, columns, target):
    
    df_lagged = df.copy()
   
    for window in range(1, trailing_window_size + 1):
        shifted = df[columns + [ target] ].groupby(columns).shift(window)
        shifted.columns = [ target + "_lag" + str(window) ]
        df_lagged = pd.concat((df_lagged, shifted), axis=1)
    df_lagged.dropna(inplace=True)
    
    return df_lagged
    

keys = ['shop_id', 'item_id']

tstart = time.time()
t = lag_feature(train, 3, keys, 'item_cnt_month')
tend = time.time()
print(t.head(500))
print("TIME: ", tend - tstart)