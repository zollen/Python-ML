'''
Created on Jul. 2, 2021

@author: zollen
'''

import pandas as pd
import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
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
train_item_cats_shops = pd.merge(train_item_cats, shops, how='left', on='shop_id')




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

# 2. groupby( ["date_block_num","shop_id","item_id"] ).agg({"item_cnt_month" : ["mean"]})
f2 = raw.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['mean']})
f2.columns = [ 'date_shop_item_avg_cnt' ]
train_item_cats_shops = train_item_cats_shops.merge(f2, on=['date_block_num', 'shop_id', 'item_id'], how='left')
train_item_cats_shops.fillna(0, inplace = True)

# 3. groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_day': ['mean']})
f3 = raw_item_cats.groupby(['date_block_num', 'shop_id', 'item_subtype']).agg({'item_cnt_day': ['mean']})
f3.columns = [ 'date_shop_subtype_avg_cnt' ]
train_item_cats_shops = train_item_cats_shops.merge(f3, on=['date_block_num', 'shop_id', 'item_subtype'], how='left')
train_item_cats_shops.fillna(0, inplace = True)





'''
adding lag features
'''
features = ['date_block_num', 'shop_id', 'item_id', 
            'shop_category', 'shop_city',
            'item_price', 'item_category_id', 'name2', 
            'name3', 'item_type', 'item_subtype',
            'item_cnt_month_lag1', 'item_cnt_month_lag2', 'item_cnt_month_lag3',
            'date_item_avg_cnt_lag1', 'date_item_avg_cnt_lag2', 'date_item_avg_cnt_lag3',
            'date_shop_item_avg_cnt_lag1', 'date_shop_item_avg_cnt_lag2', 'date_shop_item_avg_cnt_lag3',
            'date_shop_subtype_avg_cnt_lag1', 'date_shop_subtype_avg_cnt_lag2', 'date_shop_subtype_avg_cnt_lag3'
            ]
            
keys = ['shop_id', 'item_id']
targets = ['item_cnt_month', 'date_item_avg_cnt', 
           'date_shop_item_avg_cnt', 'date_shop_subtype_avg_cnt' ]


pp = lag_features(train_item_cats_shops, 3, keys, targets)

pp.drop(columns=['date_item_avg_cnt', 'date_shop_item_avg_cnt',
                 'date_shop_subtype_avg_cnt'], inplace = True)


t1 = pp[pp['date_block_num'] < 33]
t2 = pp[pp['date_block_num'] == 33]

print(t1.head())




del raw
del train
del items
del cats
del shops
del items_cats
del train_item_cats
del raw_item_cats
del train_item_cats_shops
del pp
del f1
del f2
del f3


'''
XGBRegressor               :  0.7710
LGBMRegressor              :  0.7608
CatBoostRegressor          :  0.7699
Ridge                      :  0.7567    ***
Lasso                      :  0.8681
ElasticNet                 :  0.8339
PoissonRegressor           :  0.8696
HuberRegressor             :  0.9647
ARDRegression              :  0.7556    ***
TweedieRegressor           :  0.8044
SGDRegressor               :  6943300761544078
PassiveAggressiveRegressor :  0.9282
LinearRegression           :  0.7567    ***
RandomForestRegressor      :  
AdaBoostRegressor          :  1.3426
DecisionTreeRegressor      :  1.0124
GradientBoostingRegressor  :  

'''


model = RandomForestRegressor()
model.fit(t1[features], t1[label])
preds = model.predict(t2[features])

print("TIME: ", time.time() - ts)

print("RMSE   : %0.4f" % np.sqrt(mean_squared_error(t2[label], preds)))



