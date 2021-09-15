'''
Created on Aug. 2, 2021

@author: zollen
@url: https://towardsdatascience.com/why-is-everyone-at-kaggle-obsessed-with-optuna-for-hyperparameter-tuning-7608fdca337c
'''
import optuna 
import joblib
from optuna.samplers import CmaEsSampler
import pandas as pd
import time
from os import path
import warnings


warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)


base_features = ['date_block_num', 'shop_id', 'item_id', 
            'shop_category', 'shop_city', 
            'item_category_id', 'name2', 
            'name3', 'item_type', 'item_subtype', 'item_price']




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
Optimization
'''
def evaluate(trial, data):
    
    global study, file
    
    if trial.number % 5 == 0:
        joblib.dump(study, file)
    
    p = []
    p.append(trial.suggest_float("p0", -1000, 1000))
    for i in range(1, 11):
        p.append(trial.suggest_float("p" + str(i), -100, 100))  
           
    return sum(abs(x[10] - (p[0] +      
                         p[1]  * x[0] +    
                         p[2]  * x[1] + 
                         p[3]  * x[2] + 
                         p[4]  * x[3] + 
                         p[5]  * x[4] + 
                         p[6]  * x[5] + 
                         p[7]  * x[6] + 
                         p[8]  * x[7] + 
                         p[9]  * x[8] +
                         p[10] * x[9])) for x in data)
    

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
study.optimize(func, n_trials=1200)

end_st = time.time()

print(f"Best Score: {study.best_value:.4f} params: {study.best_params}")
print("TIME: ", end_st - start_st)

joblib.dump(study, file)

'''
TOTAL: 14242872
Best Score: 14503312440.6002 params: {'p0': 176.19860947223722, 'p1': 44.76551279721511, 'p2': 22.187786273234266, 'p3': -0.024627058643813697, 'p4': 78.24969588917398, 'p5': -96.50926558337359, 'p6': 10.233831206929601, 'p7': -6.304993396472517, 'p8': 0.15217650946201097, 'p9': -89.0420029942246, 'p10': 2.9856415053998893}


'''


