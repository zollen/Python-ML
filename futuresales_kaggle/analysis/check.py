'''
Created on Jul. 3, 2021

@author: zollen
'''

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

test = pd.read_csv('../data/test.csv')
preds = pd.read_csv('../data/prediction.csv')
train = pd.read_csv('../data/sales_train.csv')


kk = pd.merge(test, preds, how='inner')
res = kk.loc[kk['item_cnt_month'] < 0, ['shop_id', 'item_id', 'item_cnt_month']]



'''
k = res.set_index(['shop_id', 'item_id']).T.to_dict('list')
for shopId, itemId in k.keys():
    nn = train[(train['shop_id'] == shopId) & 
                (train['item_id'] == itemId) &
                (train['date_block_num'] == 33 |
                 (train['date_block_num'] == 32))]
    if len(nn) > 0:
        print(nn)
'''



