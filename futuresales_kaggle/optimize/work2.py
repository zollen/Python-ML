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
import time
import os.path
from os import path
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

file = "futuresales.pkl"
# Create study that minimizes
if path.exists(file):
    study = joblib.load(file)
else:
    study = optuna.create_study(
                study_name='futuresales-study',
                direction="minimize", sampler=CmaEsSampler(seed=int(time.time())))

# Pass additional arguments inside another function
func = lambda trial: evaluate(trial, data)

# Start optimizing with 100 trials
study.optimize(func, n_trials=200)

end_st = time.time()

print(f"Best Score: {study.best_value:.4f} params: {study.best_params}")
print("TIME: ", end_st - start_st)

joblib.dump(study, file)

'''
Score: 2250073130.00394  params:  {'p0': -0.5474754044323396, 'p1': -0.22365215518548298, 'p2': -1.288031780721671, 'p3': -0.00726546261621994, 'p4': -0.37894192162338103, 'p5': 1.1902314904699083, 'p6': -0.39375544834483, 'p7': -0.0683183202926567, 'p8': -0.11570387695028395, 'p9': 0.9350971601602047, 'p10': -0.6414492712711383, 'p11': 0.06585529945306223}
Score: 1603687314.1622   params:  {'p0': -0.26701899380834504, 'p1': 0.5192701508371832, 'p2': 0.0010182488657887675, 'p3': -0.014977711643763594, 'p4': 0.03285648564724977, 'p5': 0.10203659111934255, 'p6': 0.28920470992600295, 'p7': -0.7602955823756627, 'p8': 0.1361232979885994, 'p9': -1.6865537135981563, 'p10': 1.2379125214377276, 'p11': 0.01903338535365648}
Score: 1208362412.0784   params:  {'p0': -0.30765495158226214, 'p1': -0.029031933525130782, 'p2': -0.30741734047186403, 'p3': 0.005358124757376518, 'p4': -0.017848373531018952, 'p5': 0.719662202065308, 'p6': 0.4838679424152582, 'p7': -1.12995252252948, 'p8': 6.16524535470564e-05, 'p9': -1.1581930974154977, 'p10': 0.8297081140081501, 'p11': -0.05102987682630354}
'''


