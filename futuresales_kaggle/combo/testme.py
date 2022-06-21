'''
Created on Mar. 10, 2022

@author: zollen
'''

import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

xgb1 = pd.read_csv('xgb1.csv.data')
xgb2 = pd.read_csv('xgb2.csv.data')
xgb3 = pd.read_csv('xgb3.csv.data')
lgbm1 = pd.read_csv('lgbm1.csv.data')
cat1 = pd.read_csv('cat1.csv.data')

df = pd.DataFrame()
df['ID'] = xgb1['ID']
df['item_cnt_month'] = 0.4 * xgb1['item_cnt_month'] + 0.2 * xgb2['item_cnt_month'] + 0.1 * xgb3['item_cnt_month'] + 0.3 * lgbm1['item_cnt_month']



df.to_csv('../data/prediction.csv', index = False)



print("DONE")
