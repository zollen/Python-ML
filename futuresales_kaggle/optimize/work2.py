'''
Created on Aug. 2, 2021

@author: zollen
@url: https://towardsdatascience.com/why-is-everyone-at-kaggle-obsessed-with-optuna-for-hyperparameter-tuning-7608fdca337c
'''
import optuna 
from optuna.samplers import CmaEsSampler
import pandas as pd
import numpy as np
import time
import warnings


warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)


label = 'item_cnt_month'
base_features = ['date_block_num', 'shop_id', 'item_id', 
            'shop_category', 'shop_city', 
            'item_category_id', 'name2', 
            'name3', 'item_type', 'item_subtype', 'item_price', label]




train = pd.read_csv('../data/monthly_train.csv')
items = pd.read_csv('../data/monthly_items.csv')
cats = pd.read_csv('../data/monthly_cats.csv')
shops = pd.read_csv('../data/monthly_shops.csv')


'''
merge cats, shops and items
'''
items_cats = pd.merge(items, cats, how='left', on='item_category_id')
train_item_cats = pd.merge(train, items_cats, how='left', on='item_id')
train_item_cats_shops = pd.merge(train_item_cats, shops, how='left', on='shop_id')

data = train_item_cats_shops[base_features].values.tolist()




'''
clip values between 0 and 20
'''
train_item_cats_shops[label] = train_item_cats_shops[label].clip(0, 20)



'''
Optimization
'''
def evaluate(trial, data):
    
    p = []
    p.append(trial.suggest_float("p0", -10, 10))
    for i in range(1, 12):
        p.append(trial.suggest_float("p" + str(i), -2, 2))     
    
    return sum(abs(x[11] - (p[0] +      
                         p[1]  * x[0] +    
                         p[2]  * x[1] + 
                         p[3]  * x[2] + 
                         p[4]  * x[3] + 
                         p[5]  * x[4] + 
                         p[6]  * x[5] + 
                         p[7]  * x[6] + 
                         p[8]  * x[7] + 
                         p[9]  * x[8] +
                         p[10] * x[9] + 
                         p[11] * x[10])) for x in data)
    

start_st = time.time()
# Create study that minimizes
study = optuna.create_study(
                study_name='futuresales-study', storage=None, load_if_exists=True,
                direction="minimize", sampler=CmaEsSampler(seed=int(time.time())))

# Pass additional arguments inside another function
func = lambda trial: evaluate(trial, data)

# Start optimizing with 100 trials
study.optimize(func, n_trials=100)

end_st = time.time()

print(f"Optimized Params: {study.best_params}")
print(f"Optimized RMSLE: {study.best_value:.5f}")
print("TIME: ", end_st - start_st)

'''
Score: 2250073130.00394  params:  {'p0': -0.5474754044323396, 'p1': -0.22365215518548298, 'p2': -1.288031780721671, 'p3': -0.00726546261621994, 'p4': -0.37894192162338103, 'p5': 1.1902314904699083, 'p6': -0.39375544834483, 'p7': -0.0683183202926567, 'p8': -0.11570387695028395, 'p9': 0.9350971601602047, 'p10': -0.6414492712711383, 'p11': 0.06585529945306223}
Score: 
'''


