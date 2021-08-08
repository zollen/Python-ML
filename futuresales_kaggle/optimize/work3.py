'''
Created on Aug. 2, 2021

@author: zollen
@url: https://towardsdatascience.com/why-is-everyone-at-kaggle-obsessed-with-optuna-for-hyperparameter-tuning-7608fdca337c
'''
import optuna 
import joblib
from optuna.samplers import CmaEsSampler
import pandas as pd
import numpy as np
from os import path
import time
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import futuresales_kaggle.lib.future_lib as ft
import warnings


warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

np.random.seed(0)

base_features = ['date_block_num', 'shop_id', 'item_id', 
            'shop_category', 'shop_city', 
            'item_category_id', 'name2', 
            'name3', 'item_type', 'item_subtype', 'item_price']
removed_features = ['delta_reveune_lag1', 'delta_reveune_lag3',
                    'date_itemtype_avg_cnt_lag2', 'date_itemtype_avg_cnt_lag3']
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




datax = pp.loc[pp['date_block_num'] < 33, features].values
datay = pp.loc[pp['date_block_num'] < 33, label].values
testx = pp.loc[pp['date_block_num'] == 33, features].values
testy = pp.loc[pp['date_block_num'] == 33, label].values
del pp



'''
Optimization
'''
def evaluate(trial, datax, datay, testx, testy):
    
    global features, label
    
    params = {
        "n_estimators": trial.suggest_int(name="n_estimators", low=80, high=200),
        "max_depth": trial.suggest_int("max_depth", 2, 50),
        "num_leaves": trial.suggest_int("num_leaves", 5, 100),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 0, 50),
        "max_bin": trial.suggest_int("max_bin", 2, 50),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 5),
        "boosting": trial.suggest_categorical(
            "boosting", choices=["gbdt", "dart", "goss"]
        ),
        "n_jobs": -1,
        "random_state": 0,
    }
    
    model = LGBMRegressor(**params)
    model.fit(datax, datay)
    preds = model.predict(testx)
    preds = preds.astype('int64').clip(0, 20)
    return np.sqrt(mean_squared_error(testy, preds))
    
    

start_st = time.time()

file = "lightgbm.pkl"
# Create study that minimizes
if path.exists(file):
    study = joblib.load(file)
else:
    study = optuna.create_study(
                study_name='lightgbm-study',
                direction="minimize", sampler=CmaEsSampler(seed=int(time.time())))

# Pass additional arguments inside another function
func = lambda trial: evaluate(trial, datax, datay, testx, testy)

# Start optimizing with 100 trials
study.optimize(func, n_trials=100)

end_st = time.time()

print(f"Score: {study.best_value:.4f} params: {study.best_params}")
print("TIME: ", end_st - start_st)

joblib.dump(study, file)

'''

'''


