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

'''
1. use clip(0, 21), clip(0, 19), clip(0,15) yield lower rmse. Need to revisit
'''

features = ['date_block_num', 'shop_id', 'item_id', 
            'shop_category', 'shop_city',
            'item_price', 'item_category_id', 'name2', 
            'name3', 'item_type', 'item_subtype']
label = 'item_cnt_month'

train = pd.read_csv('../data/monthly_train.csv')
test = pd.read_csv('../data/monthly_test.csv')
items = pd.read_csv('../data/monthly_items.csv')
cats = pd.read_csv('../data/monthly_cats.csv')
shops = pd.read_csv('../data/monthly_shops.csv')

ts = time.time()

items_cats = pd.merge(items, cats, how='left', on='item_category_id')
train_item_cats = pd.merge(train, items_cats, how='left', on='item_id')
test_item_cats = pd.merge(test, items_cats, how='left', on='item_id')
train_item_cats_shops = pd.merge(train_item_cats, shops, how='left', on='shop_id')
test_item_cats_shops = pd.merge(test_item_cats, shops, how='left', on='shop_id')

train_item_cats_shops[label] = train_item_cats_shops[label].clip(0, 20)

print(train_item_cats_shops.head())

model = LGBMRegressor()
model.fit(train_item_cats_shops[features], train_item_cats_shops[label])
preds = model.predict(test_item_cats_shops[features])

test[label] = preds
test[label] = test[label].astype('int64')
test[label] = test[label].clip(0, 20)

test[['ID', label]].to_csv('../data/prediction.csv', index = False)

print("TIME: ", time.time() - ts)

print("Done")



