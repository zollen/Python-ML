'''
Created on Jul. 2, 2021

@author: zollen
'''

import pandas as pd
import numpy as np
import time
from lightgbm import LGBMRegressor
import futuresales_kaggle.lib.future_lib as ft
import warnings


warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

np.random.seed(0)



'''
1. use clip(0, 21), clip(0, 19), clip(0,15) yield lower rmse. Need to revisit
2. ship_id, name3
'''
base_features = ['date_block_num', 'shop_id', 'item_id', 
            'shop_category', 'shop_city', 
            'item_category_id', 'name2', 
            'name3', 'item_type', 'item_subtype', 'item_price']
removed_features = ['delta_reveune_lag1', 'delta_reveune_lag3',
                    'date_itemtype_avg_cnt_lag2', 'date_itemtype_avg_cnt_lag3',
                    'date_name3_avg_cnt_lag1', 'date_name3_avg_cnt_lag3']
label = 'item_cnt_month'
keys = ['shop_id', 'item_id']
lag_features = [ label ]
LAGS = 3


raw = pd.read_csv('../data/sales_train.csv')
train = pd.read_csv('../data/monthly_train.csv')
test = pd.read_csv('../data/monthly_test2.csv')
items = pd.read_csv('../data/monthly_items.csv')
cats = pd.read_csv('../data/monthly_cats.csv')
shops = pd.read_csv('../data/monthly_shops.csv')

ts = time.time()



'''
merge cats, shops and items
'''
items_cats = pd.merge(items, cats, how='left', on='item_category_id')
train_item_cats = pd.merge(train, items_cats, how='left', on='item_id')
raw_item_cats = pd.merge(raw, items_cats, how='left', on='item_id')
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
train_item_cats_shops, test_item_cats_shops = ft.add_item_avg_cnt(lag_features, 
                        raw, train_item_cats_shops, test_item_cats_shops)

# 2. groupby( ["date_block_num","shop_id","item_id"] ).agg({"item_cnt_month" : ["mean"]})
train_item_cats_shops, test_item_cats_shops = ft.add_date_item_avg_cnt(lag_features, 
                        raw, train_item_cats_shops, test_item_cats_shops)

# 3. groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_day': ['mean']})
train_item_cats_shops, test_item_cats_shops = ft.add_date_shop_subtype_avg_cnt(lag_features, 
                        raw_item_cats, train_item_cats_shops, test_item_cats_shops)

# 4. groupby( ["date_block_num","shop_id"] ).agg({"revenue": ["sum"] })
# 4. groupby(["shop_id"]).agg({ "revenue":["mean"] })
train_item_cats_shops, test_item_cats_shops = ft.add_delta_revenue(lag_features, 
                        raw_item_cats, train_item_cats_shops, test_item_cats_shops)

# 5. groupby(['item_id']).agg({"item_price": ["mean"]})
# 5. groupby(['date_block_num', 'item_id']).agg({"item_price": ["mean"]})
train_item_cats_shops, test_item_cats_shops = ft.add_delta_price(lag_features, 
                        raw_item_cats, train_item_cats_shops, test_item_cats_shops)

# 6. groupby( ["date_block_num","item_type"] ).agg({"item_cnt_month" : ["mean"]})
train_item_cats_shops, test_item_cats_shops = ft.add_date_itemtype_cnt(lag_features, 
                        raw_item_cats, train_item_cats_shops, test_item_cats_shops)


# 7. groupby( ["date_block_num","item_category_id"] ).agg({"item_cnt_month" : ["mean"]})
train_item_cats_shops, test_item_cats_shops = ft.add_date_itemcat_cnt(lag_features, 
                        raw_item_cats, train_item_cats_shops, test_item_cats_shops)

# 8. groupby( ["date_block_num","name3"] ).agg({"item_cnt_month" : ["mean"]})
train_item_cats_shops, test_item_cats_shops = ft.add_date_name3_avg_cnt(lag_features, 
                        raw_item_cats, train_item_cats_shops, test_item_cats_shops)




all_df = pd.concat([train_item_cats_shops, test_item_cats_shops])
all_df.drop(columns=['ID'], inplace=True)
all_df.loc[all_df['date_block_num'] == 34, 'item_cnt_month'] = 0


del raw
del train
del items
del cats
del shops
del items_cats
del train_item_cats
del raw_item_cats
del test_item_cats
del train_item_cats_shops



'''
adding lag features
'''            
pp = ft.add_lag_features(all_df, LAGS, keys, lag_features)
del all_df





new_features = []
for feature in lag_features:
    for i in range(1, LAGS+1):
        new_features.append(feature + "_lag" + str(i))    


        
for feature in removed_features:   
    if feature in new_features:
        new_features.remove(feature)
pp.drop(columns=lag_features[1:] + removed_features, inplace = True)




features = base_features + new_features



t1 = pp[pp['date_block_num'] < 34]
t2 = pp.loc[pp['date_block_num'] == 34, keys + new_features]
del pp




test_item_cats_shops = test_item_cats_shops.merge(t2, on=keys, how='left')
test_item_cats_shops.fillna(0, inplace=True)


print(t1[features].head())
print(test_item_cats_shops[features].head())


del t2





model = LGBMRegressor()
model.fit(t1[features], t1[label])

print(pd.Series(index=features, data=model.feature_importances_).sort_values(ascending=False))

preds = model.predict(test_item_cats_shops[features])

test[label] = preds
test[label] = test[label].astype('int64')
test[label] = test[label].clip(0, 20)

test[['ID', label]].to_csv('../data/prediction.csv', index = False)

print("TIME: ", time.time() - ts)

print("Done")



