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

def lag_features(df, trailing_window_size, columns, targets):
    
    df_lagged = df.copy()
   
    for window in range(1, trailing_window_size + 1):
        shifted = df[columns + targets ].groupby(columns).shift(window)
        shifted.columns = [x + "_lag" + str(window) for x in df[targets]]
        df_lagged = pd.concat((df_lagged, shifted), axis=1)
    df_lagged.dropna(inplace=True)
    
    return df_lagged


'''
1. use clip(0, 21), clip(0, 19), clip(0,15) yield lower rmse. Need to revisit
'''

label = 'item_cnt_month'

raw = pd.read_csv('../data/sales_train.csv')
train = pd.read_csv('../data/monthly_train.csv')
test = pd.read_csv('../data/monthly_test.csv')
items = pd.read_csv('../data/monthly_items.csv')
cats = pd.read_csv('../data/monthly_cats.csv')
shops = pd.read_csv('../data/monthly_shops.csv')

ts = time.time()


'''
merge cats, shops and items
'''
items_cats = pd.merge(items, cats, how='left', on='item_category_id')
train_item_cats = pd.merge(train, items_cats, how='left', on='item_id')
test_item_cats = pd.merge(test, items_cats, how='left', on='item_id')
train_item_cats_shops = pd.merge(train_item_cats, shops, how='left', on='shop_id')
test_item_cats_shops = pd.merge(test_item_cats, shops, how='left', on='shop_id')



'''
clip values between 0 and 20
'''
train_item_cats_shops[label] = train_item_cats_shops[label].clip(0, 20)



'''
adding new features
'''

# 1. groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
f1 = raw.groupby(['date_block_num', 'item_id']).agg({'item_cnt_day': ['mean']})
f1.columns = [ 'date_item_avg_cnt' ]
train_item_cats_shops = train_item_cats_shops.merge(f1, on=['date_block_num', 'item_id'], how='left')
train_item_cats_shops.fillna(0, inplace = True)
test_item_cats_shops = test_item_cats_shops.merge(f1, on=['date_block_num', 'item_id'], how='left')
test_item_cats_shops.fillna(0, inplace = True)

# 2. groupby( ["date_block_num","shop_id","item_id"] ).agg({"item_cnt_month" : ["mean"]})
f2 = raw.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['mean']})
f2.columns = [ 'date_shop_item_avg_cnt' ]
train_item_cats_shops = train_item_cats_shops.merge(f2, on=['date_block_num', 'shop_id', 'item_id'], how='left')
train_item_cats_shops.fillna(0, inplace = True)
test_item_cats_shops = test_item_cats_shops.merge(f2, on=['date_block_num', 'shop_id', 'item_id'], how='left')
test_item_cats_shops.fillna(0, inplace = True)






'''
adding lag features
'''
all_df = pd.concat([train_item_cats_shops, test_item_cats_shops])
all_df.drop(columns=['ID'], inplace=True)

features = ['date_block_num', 'shop_id', 'item_id', 
            'shop_category', 'shop_city',
            'item_price', 'item_category_id', 'name2', 
            'name3', 'item_type', 'item_subtype',
            'item_cnt_month_lag1', 'item_cnt_month_lag2', 'item_cnt_month_lag3',
            'date_item_avg_cnt_lag1', 'date_item_avg_cnt_lag2', 'date_item_avg_cnt_lag3',
            'date_shop_item_avg_cnt_lag1', 'date_shop_item_avg_cnt_lag2', 'date_shop_item_avg_cnt_lag3'
            ]
            
keys = ['shop_id', 'item_id']
targets = ['item_cnt_month', 'date_item_avg_cnt', 'date_shop_item_avg_cnt' ]

all_df.loc[all_df['date_block_num'] == 34, 'item_cnt_month'] = 0

pp = lag_features(all_df, 3, keys, targets)

pp.drop(columns=['date_item_avg_cnt', 'date_shop_item_avg_cnt'], inplace = True)


t1 = pp[pp['date_block_num'] < 34]
t2 = pp.loc[pp['date_block_num'] == 34, 
                [
                    'shop_id', 'item_id', 'item_cnt_month_lag1', 
                    'item_cnt_month_lag2', 'item_cnt_month_lag3',
                    'date_item_avg_cnt_lag1', 'date_item_avg_cnt_lag2', 
                    'date_item_avg_cnt_lag3', 'date_shop_item_avg_cnt_lag1',
                    'date_shop_item_avg_cnt_lag2', 'date_shop_item_avg_cnt_lag3'
                ]]

print(t1.head())

posttest = test_item_cats_shops.merge(t2, on=['shop_id', 'item_id'], how='left')
posttest.fillna(0, inplace=True)


model = LGBMRegressor()
model.fit(t1[features], t1[label])
preds = model.predict(posttest[features])

test[label] = preds
test[label] = test[label].astype('int64')
test[label] = test[label].clip(0, 20)

test[['ID', label]].to_csv('../data/prediction.csv', index = False)

print("TIME: ", time.time() - ts)

print("Done")



