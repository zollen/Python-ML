'''
Created on Jul. 2, 2021

@author: zollen
'''

import pandas as pd
import numpy as np
import time
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

np.random.seed(0)

features = ['date_block_num', 'shop_id', 'item_id', 'item_price']
label = 'item_cnt_day'

sales = pd.read_csv('../data/monthly_train.csv')
test = pd.read_csv('../data/monthly_test.csv')


ts = time.time()

model = XGBRegressor()
model.fit(sales[features], sales[label])
preds = model.predict(test[features])

test['item_cnt_month'] = preds
test['item_cnt_month'] = test['item_cnt_month'].astype('int64')
test[['ID', label]].to_csv('../data/prediction.csv', index = False)

print("TIME: ", time.time() - ts)

print("Done")



