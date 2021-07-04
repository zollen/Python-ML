'''
Created on Jul. 2, 2021

@author: zollen
'''

import pandas as pd
import numpy as np
import time
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

np.random.seed(0)

features = ['date_block_num', 'shop_id', 'item_id', 'item_price']
label = 'item_cnt_month'

sales = pd.read_csv('../data/monthly_train.csv')
test = pd.read_csv('../data/monthly_test.csv')
items = pd.read_csv('../data/items.csv')
cats = pd.read_csv('../data/monthly_cats.csv')
shops = pd.read_csv('../data/monthly_shops.csv')



ts = time.time()

model = LGBMRegressor()
model.fit(sales[features], sales[label])
preds = model.predict(test[features])

test[label] = preds
test[label] = test[label].astype('int64')

#test.loc[(test['shop_id'] == 55) & (test['item_id'] == 8516), 'item_cnt_month'] = 1
#test.loc[test['item_cnt_month'] < 0, 'item_cnt_month'] = 0


test[['ID', label]].to_csv('../data/prediction.csv', index = False)


print("TIME: ", time.time() - ts)

print("Done")



