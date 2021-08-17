'''
Created on Jul. 17, 2021

@author: zollen
@score: 
lag features for the prediction.xsd - output: 1.03740
item_cnt_month_lag1,2,3
date_item_avg_cnt_lag1,2,3
date_shop_item_avg_cnt_lag1,2,3
date_shop_subtype_avg_cnt_lag1,2,3
delta_reveune_lag2
delta_price_lag1,2,3
date_itemtype_avg_cnt_lag1
date_itemcat_avg_cnt_lag1,2,3
date_name3_avg_cnt_lag2
'''

import pandas as pd
import numpy as np
import time
from xgboost import XGBRegressor
import futuresales_kaggle.lib.future_lib as ft
import warnings


warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

np.random.seed(0)

target = 'item_price'
label = 'item_cnt_month'
keys = ['shop_id', 'item_id']
base_features = ['date_block_num', 'shop_id', 'item_id', 
            'shop_category', 'shop_city', 
            'item_category_id', 'name2', 
            'name3', 'item_type', 'item_subtype']

lag_features = []
       

features = base_features + ['item_price_lag1', 
                            'item_price_lag2', 
                            'item_price_lag3',
                            'item_price_lag4',
                            'item_price_lag5'] + lag_features

train = pd.read_csv('../data/monthly_train.csv')
raw = pd.read_csv('../data/sales_train.csv')
test = pd.read_csv('../data/monthly_test.csv')
items = pd.read_csv('../data/monthly_items.csv')
cats = pd.read_csv('../data/monthly_cats.csv')
shops = pd.read_csv('../data/monthly_shops.csv')
preds = pd.read_csv('../data/prediction.csv')


'''
merge cats, shops and items
'''
items_cats = pd.merge(items, cats, how='left', on='item_category_id')
train_item_cats = pd.merge(train, items_cats, how='left', on='item_id')
raw_item_cats = pd.merge(raw, items_cats, how='left', on='item_id')
test_item_cats = pd.merge(test, items_cats, how='left', on='item_id')
train_item_cats_shops = pd.merge(train_item_cats, shops, how='left', on='shop_id')
test_item_cats_shops = pd.merge(test_item_cats, shops, how='left', on='shop_id')
test_item_cats_shops[label] = preds[label]
test_item_cats_shops[label] = test_item_cats_shops[label].clip(0, 20)
train_item_cats_shops[label] = train_item_cats_shops[label].clip(0, 20)



all_df = pd.concat([train_item_cats_shops, test_item_cats_shops])
all_df.drop(columns=['ID'], inplace=True)

pp = ft.add_lag_features(all_df, 5, keys, ['item_price' ] + lag_features)

pp.drop(columns=lag_features, inplace = True)



t1 = pp[pp['date_block_num'] < 34]
t3 = pp[pp['date_block_num'] == 34]


print("TOTAL: ", len(test))
print(t1.head())

start_ts = time.time()

model = XGBRegressor()
model.fit(t1[features], t1[target])


t3[target] = model.predict(t3[features])

test.drop(columns=[target], inplace = True)
test = test.merge(t3[['shop_id', 'item_id', 'item_price']], on=['shop_id', 'item_id'], how='left')
test.fillna(0, inplace=True)

'''
select rows that exists only in test, but not in train
'''
k = test.merge(train[['shop_id', 'item_id']], 
               on=['shop_id', 'item_id'], how='outer', indicator=True).loc[lambda x : x['_merge'] == 'left_only']

test = test.set_index(['shop_id', 'item_id'])
k = k.set_index(['shop_id', 'item_id'])

test.loc[test.index.isin(k.index), 'item_price'] = 0

test = test.reset_index()
k = k.reset_index()

end_ts = time.time() 


print("TOTAL: ", len(test))
print("TIME: ", end_ts - start_ts)

test.to_csv('../data/monthly_test2.csv', index = False)
print("Done")
