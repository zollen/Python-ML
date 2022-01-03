'''
Created on Oct. 16, 2021

@author: zollen
'''

import pandas as pd
import numpy as np
import optuna 
import joblib
from optuna.samplers import CmaEsSampler, TPESampler, RandomSampler
import time
import re
from os import path
from itertools import permutations, product
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings



warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

np.random.seed(0)


def lag_feature( df,lags, cols ):
    for col in cols:
        tmp = df[["date_block_num", "shop_id","item_id",col ]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag"+str(i)]
            shifted.date_block_num = shifted.date_block_num + i
            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df.fillna(0)

def add_one_lag_feature(df, feature1, post):
    
    name = feature1 + '_cnt'
    
    if name in df:
        return df
    
    f1 = df.groupby(['date_block_num', feature1]).agg({'item_cnt_month': [ 'mean' ]})
    f1.columns = [ name ]
    df = df.merge(f1, on=['date_block_num', feature1], how='left')
    df.fillna(0, inplace = True)
    df[name] = df[name].astype(np.float16)
    
    if post == 1:      
        df = lag_feature(df, [1], [ name ])
    if post == 2:
        df = lag_feature(df, [1, 2], [ name ])
    if post == 3:
        df = lag_feature(df, [1, 2, 3], [ name ])
    
    del f1
    
    return df

def add_two_lag_feature(df, feature1, feature2, post):
    
    name = feature1 + '_' + feature2 + '_cnt'
    
    if name in df:
        return df
    
    f1 = df.groupby(['date_block_num', feature1, feature2]).agg({'item_cnt_month': [ 'mean' ]})
    f1.columns = [ name ]
    df = df.merge(f1, on=['date_block_num', feature1, feature2], how='left')
    df.fillna(0, inplace = True)
    df[name] = df[name].astype(np.float16)
    
    if post == 1:      
        df = lag_feature(df, [1], [ name ])
    if post == 2:
        df = lag_feature(df, [1, 2], [ name ])
    if post == 3:
        df = lag_feature(df, [1, 2, 3], [ name ])
    
    del f1
    
    return df
    
def add_three_lag_feature(df, feature1, feature2, feature3, post):
    
    name = feature1 + '_' + feature2 + '_' + feature3 + '_cnt'
    
    if name in df:
        return df
    
    f1 = df.groupby(['date_block_num', feature1, feature2, feature3]).agg({'item_cnt_month': [ 'mean' ]})
    f1.columns = [ name ]
    df = df.merge(f1, on=['date_block_num', feature1, feature2, feature3], how='left')
    df.fillna(0, inplace = True)
    df[name] = df[name].astype(np.float16)
    
    if post == 1:      
        df = lag_feature(df, [1], [ name ])
    if post == 2:
        df = lag_feature(df, [1, 2], [ name ])
    if post == 3:
        df = lag_feature(df, [1, 2, 3], [ name ])
    
    del f1
    
    return df


base_features = [
    'shop_id', 'item_id', 'shop_category', 'shop_city',  
    'item_category_id', 'name2', 'name3', 'item_type', 'item_subtype'
    ]
'''
0. All three lags.
1. First lags.
2. Second Lags.
3. First and Second lags.
'''
tokens = []
for length in [1, 2]:
    tokens += list(permutations(base_features, r=length))
    


   
def display(params):
      
    labels = {
        0 : "123", 1: "1", 2: "2", 3: "12"
        }
    
    for i in range(1, 8):
        print(i, " ==> [", labels[params['action' + str(i)]],  "]",
              tokens[params['param' + str(i)]])
    

      
def evaluate(trial, tokens, df):
    
    joblib.dump(study, file)
    
    data = df.copy()
    
    size = len(tokens)
    params = []
    actions = []
    for i in range(1, 10):
        params.append(trial.suggest_int(name="param" + str(i), low=0, high=size - 1))
        actions.append(trial.suggest_int(name="action" + str(i), low=0, high=3))
        
    for i in range(len(params)):
        option = len(tokens[params[i]])
        if option == 1:
#            print("===> ", tokens[params[i]][0])
            data = add_one_lag_feature(data, tokens[params[i]][0], actions[i])
        if option == 2:
#            print("===> ", tokens[params[i]][0], tokens[params[i]][1])
            data = add_two_lag_feature(data, tokens[params[i]][0], tokens[params[i]][1], actions[i])
        if option == 3: 
#            print("===> ", tokens[params[i]][0], tokens[params[i]][1], tokens[params[i][2]])
            data = add_three_lag_feature(data, tokens[params[i]][0], tokens[params[i]][1], tokens[params[i]][2], actions[i])
    
#    print(data.head())
    
    
    
    datax = data[data['date_block_num'] < 33]
    datax.drop(columns=['date_block_num'], inplace = True)
    datay = data.loc[data['date_block_num'] < 33, 'item_cnt_month'] 
    testx = data[data['date_block_num'] == 33]
    testx.drop(columns=['date_block_num'], inplace = True)
    testy = data.loc[data['date_block_num'] == 33, 'item_cnt_month'] 
    del data
    
    datay = datay.clip(0, 20)
    testy = testy.clip(0, 20)
    
    print(datax.head())
    
       
    model = XGBRegressor(verbosity=0)
    model.fit(datax, datay)
    del datax
    del datay
    preds = model.predict(testx)
    preds = preds.astype('int64').clip(0, 20)
    return np.sqrt(mean_squared_error(testy, preds))


train = pd.read_csv('../data/sales_train.csv')
items = pd.read_csv('../data/items.csv')
cats = pd.read_csv('../data/item_categories.csv')
shops = pd.read_csv('../data/shops.csv')



'''
Remove any outliers
'''
train = train[(train.item_price < 300000 )& (train.item_cnt_day < 1000)]


'''
Remove any rows from train where item prices is negative
'''
train = train[train.item_price > 0].reset_index(drop = True)
train.loc[train.item_cnt_day < 1, "item_cnt_day"] = 0


'''
Remerge duplicated rows
'''
# Якутск Орджоникидзе, 56
train.loc[train.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
train.loc[train.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
train.loc[train.shop_id == 10, 'shop_id'] = 11


'''
Clean up some shop names and add city and category to shops
'''
shops.loc[ shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name" ] = 'СергиевПосад ТЦ "7Я"'
shops["city"] = shops.shop_name.str.split(" ").map( lambda x: x[0] )
shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1] )
shops.loc[shops.city == "!Якутск", "city"] = "Якутск"



'''
Only keep shop category if there are 5 or more shops of that category, 
the rest are grouped as other
'''
category = []
for cat in shops.category.unique():
    if len(shops[shops.category == cat]) >= 5:
        category.append(cat)
shops.category = shops.category.apply( lambda x: x if (x in category) else "other" )
shops["shop_category"] = LabelEncoder().fit_transform( shops.category )
shops["shop_city"] = LabelEncoder().fit_transform( shops.city )
shops = shops[["shop_id", "shop_category", "shop_city"]]


'''
Cleaning item categories data
'''
cats["item_type"] = cats.item_category_name.apply( lambda x: x.split(" ")[0] ).astype(str)
cats.loc[ (cats['item_type'] == "Игровые")| (cats['item_type'] == "Аксессуары"), "category" ] = "Игры"

category = []
for cat in cats['item_type'].unique():
    if len(cats[cats['item_type'] == cat]) >= 5: 
        category.append( cat )
cats['item_type'] = cats['item_type'].apply(lambda x: x if (x in category) else "etc")
cats['item_type'] = LabelEncoder().fit_transform(cats['item_type'])
cats["split"] = cats.item_category_name.apply(lambda x: x.split("-"))
cats["subtype"] = cats.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats["item_subtype"] = LabelEncoder().fit_transform( cats["subtype"] )
cats = cats[["item_category_id", "item_type", "item_subtype"]]


'''
Cleaning item data
'''
def name_correction(x):
    x = x.lower() # all letters lower case
    x = x.partition('[')[0] # partition by square brackets
    x = x.partition('(')[0] # partition by curly brackets
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x) # remove special characters
    x = x.replace('  ', ' ') # replace double spaces with single spaces
    x = x.strip() # remove leading and trailing white space
    return x

# split item names by first bracket
items["name1"], items["name2"] = items.item_name.str.split("[", 1).str
items["name1"], items["name3"] = items.item_name.str.split("(", 1).str

# replace special characters and turn to lower case
items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()

# fill nulls with '0'
items = items.fillna('0')

items["item_name"] = items["item_name"].apply(lambda x: name_correction(x))

# return all characters except the last if name 2 is not "0" - the closing bracket
items.name2 = items.name2.apply( lambda x: x[:-1] if x !="0" else "0")



'''
Cleaning item type
'''
items["type"] = items.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )
items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"
items.loc[ items.type == "", "type"] = "mac"
items.type = items.type.apply( lambda x: x.replace(" ", "") )
items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc"
items.loc[ items.type == 'рs3' , "type"] = "ps3"

group_sum = items.groupby(["type"]).agg({"item_id": "count"})
group_sum = group_sum.reset_index()
drop_cols = []
for cat in group_sum.type.unique():
    if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40:
        drop_cols.append(cat)
items.name2 = items.name2.apply( lambda x: "other" if (x in drop_cols) else x )
items = items.drop(["type"], axis = 1)

items.name2 = LabelEncoder().fit_transform(items.name2)
items.name3 = LabelEncoder().fit_transform(items.name3)

items.drop(["item_name", "name1"],axis = 1, inplace= True)




'''
Preprocessing
'''
ts = time.time()
matrix = []
cols  = ["date_block_num", "shop_id", "item_id"]
for i in range(34):
    sales = train[train.date_block_num == i]
    matrix.append( np.array(list( product( [i], sales.shop_id.unique(), sales.item_id.unique() ) ), dtype = np.int16) )


matrix = pd.DataFrame( np.vstack(matrix), columns = cols )
matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8)
matrix["shop_id"] = matrix["shop_id"].astype(np.int8)
matrix["item_id"] = matrix["item_id"].astype(np.int16)
matrix.sort_values( cols, inplace = True )

'''
summing the monthly counts
'''
group = train.groupby( ["date_block_num", "shop_id", "item_id"] ).agg( {"item_cnt_day": ["sum"]} )
group.columns = ["item_cnt_month"]
group.reset_index( inplace = True)
matrix = pd.merge( matrix, group, on = cols, how = "left" )
matrix["item_cnt_month"] = matrix["item_cnt_month"].fillna(0).astype(np.float16)
del group




'''
Adding shops, items and categories data into the matrix
'''
matrix = pd.merge( matrix, shops, on = ["shop_id"], how = "left" )
matrix = pd.merge(matrix, items, on = ["item_id"], how = "left")
matrix = pd.merge( matrix, cats, on = ["item_category_id"], how = "left" )

matrix["shop_city"] = matrix["shop_city"].astype(np.int16)
matrix["shop_category"] = matrix["shop_category"].astype(np.int16)
matrix["item_category_id"] = matrix["item_category_id"].astype(np.int16)
matrix["item_type"] = matrix["item_type"].astype(np.int16)
matrix["name2"] = matrix["name2"].astype(np.int16)
matrix["name3"] = matrix["name3"].astype(np.int16)
matrix["item_subtype"] = matrix["item_subtype"].astype(np.int16)


'''
add new features
'''
matrix = lag_feature( matrix, [1,2,3], ["item_cnt_month"] )


f1 = matrix.groupby(['date_block_num']).agg({'item_cnt_month': [ 'mean' ]})
f1.columns = [ 'date_avg_cnt' ]
matrix = matrix.merge(f1, on=['date_block_num'], how='left')
matrix.fillna(0, inplace = True)
matrix['date_avg_cnt'] = matrix['date_avg_cnt'].astype(np.float16)
matrix = lag_feature( matrix, [1], ["date_avg_cnt"] )
matrix.drop( ["date_avg_cnt"], axis = 1, inplace = True )
del f1

del train
del items
del cats
del shops


start_st = time.time()

file = "lgbm_features.pkl"
# Create study that minimizes
if path.exists(file):
    study = joblib.load(file)
else:
    study = optuna.create_study(
                study_name='lgbm-features',
                direction="minimize", sampler=CmaEsSampler(seed=int(time.time())))

# Pass additional arguments inside another function
func = lambda trial: evaluate(trial, tokens, matrix)

# Start optimizing with 100 trials
study.optimize(func, n_trials=200)

end_st = time.time()

print(f"Score: {study.best_value:.4f}")
print(display(study.best_params))
print("TIME: ", end_st - start_st)

joblib.dump(study, file)
