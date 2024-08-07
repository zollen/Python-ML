'''
Created on Sep. 24, 2021

@author: zollen
'''

import pandas as pd
import numpy as np
import time
import re
from itertools import product
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import futuresales_kaggle.lib.future_lib as ft
import warnings


warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

np.random.seed(0)


train = pd.read_csv('../data/sales_train.csv')
test = pd.read_csv('../data/test.csv')
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
test.loc[test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11


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



'''
Creating a test set of month 34
'''
test["date_block_num"] = 34
test["date_block_num"] = test["date_block_num"].astype(np.int8)
test["shop_id"] = test.shop_id.astype(np.int8)
test["item_id"] = test.item_id.astype(np.int16)


'''
Concatenating train and test sets
'''
matrix = pd.concat([matrix, test.drop(["ID"],axis = 1)], ignore_index=True, sort=False, keys=cols)
matrix.fillna( 0, inplace = True )


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
LAGS=3
lag_features = [ 'item_cnt_month' ]
removed_features = []


f1 = matrix.groupby(['date_block_num']).agg({'item_cnt_month': [ 'mean' ]})
f1.columns = [ 'date_avg_cnt' ]
matrix = matrix.merge(f1, on=['date_block_num'], how='left')
matrix.fillna(0, inplace = True)
matrix['date_avg_cnt'] = matrix['date_avg_cnt'].astype(np.float16)
lag_features.append('date_avg_cnt')
removed_features.append('date_avg_cnt')
removed_features.append('date_avg_cnt_lag2')
removed_features.append('date_avg_cnt_lag3')
del f1

f1 = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': [ 'mean' ]})
f1.columns = [ 'date_shop_avg_cnt' ]
matrix = matrix.merge(f1, on=['date_block_num', 'shop_id'], how='left')
matrix.fillna(0, inplace = True)
matrix['date_shop_avg_cnt'] = matrix['date_shop_avg_cnt'].astype(np.float16)
lag_features.append('date_shop_avg_cnt')
removed_features.append('date_shop_avg_cnt')
del f1

f1 = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': [ 'mean' ]})
f1.columns = [ 'date_item_avg_cnt' ]
matrix = matrix.merge(f1, on=['date_block_num', 'item_id'], how='left')
matrix.fillna(0, inplace = True)
matrix['date_item_avg_cnt'] = matrix['date_item_avg_cnt'].astype(np.float16)
lag_features.append('date_item_avg_cnt')
removed_features.append('date_item_avg_cnt')
del f1

f1 = matrix.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_month': [ 'mean' ]})
f1.columns = [ 'date_shop_item_avg_cnt' ]
matrix = matrix.merge(f1, on=['date_block_num', 'shop_id','item_id'], how='left')
matrix.fillna(0, inplace = True)
matrix['date_shop_item_avg_cnt'] = matrix['date_shop_item_avg_cnt'].astype(np.float16)
lag_features.append('date_shop_item_avg_cnt')
removed_features.append('date_shop_item_avg_cnt')
del f1

f1 = matrix.groupby(['date_block_num', 'shop_id', 'item_subtype']).agg({'item_cnt_month': [ 'mean' ]})
f1.columns = [ 'date_shop_subtype_avg_cnt' ]
matrix = matrix.merge(f1, on=['date_block_num', 'shop_id','item_subtype'], how='left')
matrix.fillna(0, inplace = True)
matrix['date_shop_subtype_avg_cnt'] = matrix['date_shop_subtype_avg_cnt'].astype(np.float16)
lag_features.append('date_shop_subtype_avg_cnt')
removed_features.append('date_shop_subtype_avg_cnt')
removed_features.append('date_shop_subtype_avg_cnt_lag2')
removed_features.append('date_shop_subtype_avg_cnt_lag3')
del f1

f1 = matrix.groupby(['date_block_num', 'shop_city']).agg({'item_cnt_month': [ 'mean' ]})
f1.columns = [ 'date_city_avg_cnt' ]
matrix = matrix.merge(f1, on=['date_block_num', 'shop_city'], how='left')
matrix.fillna(0, inplace = True)
matrix['date_city_avg_cnt'] = matrix['date_city_avg_cnt'].astype(np.float16)
lag_features.append('date_city_avg_cnt')
removed_features.append('date_city_avg_cnt')
removed_features.append('date_city_avg_cnt_lag2')
removed_features.append('date_city_avg_cnt_lag3')
del f1

f1 = matrix.groupby(['date_block_num', 'item_id', 'shop_city']).agg({'item_cnt_month': [ 'mean' ]})
f1.columns = [ 'date_item_city_avg_cnt' ]
matrix = matrix.merge(f1, on=['date_block_num', 'item_id','shop_city'], how='left')
matrix.fillna(0, inplace = True)
matrix['date_item_city_avg_cnt'] = matrix['date_item_city_avg_cnt'].astype(np.float16)
lag_features.append('date_item_city_avg_cnt')
removed_features.append('date_item_city_avg_cnt')
removed_features.append('date_item_city_avg_cnt_lag2')
removed_features.append('date_item_city_avg_cnt_lag3')
del f1





price_features = True
'''
adding average item price 
adding lag values of item price per month
add delta price values - how current month average price related to global average
'''
if price_features:
    group = train.groupby( ["item_id"] ).agg({"item_price": ["mean"]})
    group.columns = ["item_avg_item_price"]
    group.reset_index(inplace = True)
    
    matrix = matrix.merge( group, on = ["item_id"], how = "left" )
    matrix["item_avg_item_price"] = matrix.item_avg_item_price.astype(np.float16)
    
    
    group = train.groupby( ["date_block_num","item_id"] ).agg( {"item_price": ["mean"]} )
    group.columns = ["date_item_avg_item_price"]
    group.reset_index(inplace = True)
    
    matrix = matrix.merge(group, on = ["date_block_num","item_id"], how = "left")
    matrix["date_item_avg_item_price"] = matrix.date_item_avg_item_price.astype(np.float16)
    
    
    lag_features.append('date_item_avg_item_price')
    removed_features.append('item_avg_item_price')
    removed_features.append('date_item_avg_item_price')
    removed_features.append('date_item_avg_item_price_lag1')
    removed_features.append('date_item_avg_item_price_lag2')
    removed_features.append('date_item_avg_item_price_lag3')
    removed_features.append('delta_price_lag1')
    removed_features.append('delta_price_lag2')
    removed_features.append('delta_price_lag3')



'''
add total shop revenue per month to matrix
add lag values of revenus per month
add delta revenus values - how current month revene related to global average
'''
if price_features:    
    train["revenue"] = train["item_cnt_day"] * train["item_price"]
    
    group = train.groupby( ["date_block_num","shop_id"] ).agg({"revenue": ["sum"] })
    group.columns = ["date_shop_revenue"]
    group.reset_index(inplace = True)
    
    matrix = matrix.merge( group , on = ["date_block_num", "shop_id"], how = "left" )
    matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)
    
    group = group.groupby(["shop_id"]).agg({ "date_block_num":["mean"] })
    group.columns = ["shop_avg_revenue"]
    group.reset_index(inplace = True )
    
    matrix = matrix.merge( group, on = ["shop_id"], how = "left" )
    
    matrix["shop_avg_revenue"] = matrix.shop_avg_revenue.astype(np.float32)
    matrix["delta_revenue"] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
    matrix["delta_revenue"] = matrix["delta_revenue"]. astype(np.float32)
    
    lag_features.append('delta_revenue')
    removed_features.append('delta_revenue')
    removed_features.append('delta_revenue_lag2')
    removed_features.append('delta_revenue_lag3')
    removed_features.append('date_shop_revenue')
    removed_features.append('shop_avg_revenue')







'''
add month and days
'''
matrix["month"] = matrix["date_block_num"] % 12
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix["days"] = matrix["month"].map(days).astype(np.int8)


'''
add month of each shop
'''
matrix["item_shop_first_sale"] = matrix["date_block_num"] - matrix.groupby(["item_id","shop_id"])["date_block_num"].transform('min')


'''
add item first sale
'''
matrix["item_first_sale"] = matrix["date_block_num"] - matrix.groupby(["item_id"])["date_block_num"].transform('min')





matrix = ft.add_lag_features(matrix, LAGS, ['shop_id', 'item_id'], lag_features)



if price_features:
    for i in [1,2,3]:
        matrix["delta_price_lag" + str(i) ] = (matrix["date_item_avg_item_price_lag" + str(i)]- matrix["item_avg_item_price"] )/ matrix["item_avg_item_price"]
    
    def select_trends(row) :
        for i in [1,2,3]:
            if row["delta_price_lag" + str(i)]:
                return row["delta_price_lag" + str(i)]
        return 0
    
    matrix["delta_price_lag"] = matrix.apply(select_trends, axis = 1)
    matrix["delta_price_lag"] = matrix.delta_price_lag.astype( np.float16 )
    matrix["delta_price_lag"].fillna( 0 ,inplace = True)








matrix.drop(columns=lag_features[1:] + removed_features, inplace = True)




'''
Training and Prediction
'''
label = 'item_cnt_month'

matrix[label] = matrix[label].clip(0, 20)

trainingX = matrix[matrix['date_block_num'] < 34]
trainingX.drop(columns=[label], inplace=True)
trainingY = matrix.loc[matrix['date_block_num'] < 34, label]

testingX = matrix[matrix['date_block_num'] == 34]
testingX.drop(columns=[label], inplace=True)

print(trainingX.head())
print(testingX.head())


model = XGBRegressor(max_depth=10,
                    n_estimators=1000,
                    min_child_weight=0.5, 
                    colsample_bytree=0.8, 
                    subsample=0.8, 
                    eta=0.1,
                    #     tree_method='gpu_hist',
                    seed=42)

model.fit(trainingX, trainingY);         

print(pd.Series(index=trainingX.columns, data=model.feature_importances_).sort_values(ascending=False))

testingX[label] = model.predict(testingX)


test = pd.merge(test, testingX[['shop_id', 'item_id', label]], on = ['shop_id', 'item_id'], how = "left")
test[label].fillna(0, inplace=True)
test[label] = test[label].clip(0, 20)
test[label] = test[label].astype('int16')

test[['ID', label]].to_csv('../data/prediction.csv', index = False)

print("TIME: ", time.time() - ts)


print("DONE")

'''
model = XGBRegressor(
    max_depth=10,
    n_estimators=1000,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1,
#     tree_method='gpu_hist',
    seed=42)
'''
'''
Collinearity
                          variables         VIF
0                    date_block_num    5.946952
1                           shop_id  170.171539
2                           item_id    5.001425
3                     shop_category    5.063728
4                         shop_city  181.471562
5                  item_category_id  172.425908
6                             name2    2.206011
7                             name3    1.899919
8                         item_type  172.434132
9                      item_subtype    3.276374
10                            month    3.606924
11                             days   41.721868
12             item_shop_first_sale   26.250171
13                  item_first_sale   26.773111
14                date_avg_cnt_lag1   22.382493
15           date_shop_avg_cnt_lag1   15.033166
16           date_item_avg_cnt_lag1    5.032364
17      date_shop_item_avg_cnt_lag1    4.221334
18   date_city_subtype_avg_cnt_lag1    5.276569
19   date_itemcat_shop_avg_cnt_lag1  184.895777
20  date_shopcat_name3_avg_cnt_lag1    7.892881
21   date_shop_subtype_avg_cnt_lag1  186.310781
22                date_avg_cnt_lag2   28.074756
23           date_shop_avg_cnt_lag2   24.111312
24           date_item_avg_cnt_lag2    6.286725
25      date_shop_item_avg_cnt_lag2    3.987554
26  date_name3_itemcat_avg_cnt_lag2   13.256601
27  date_shopcat_name3_avg_cnt_lag2   12.965292
28                date_avg_cnt_lag3   18.541992
29           date_shop_avg_cnt_lag3   15.998211
30           date_item_avg_cnt_lag3    4.310151
31      date_shop_item_avg_cnt_lag3    3.291685

---------------------
Skew Test

'date_avg_cnt_lag1': 'COXBOX',
'date_avg_cnt_lag2': 'COXBOX',
'date_avg_cnt_lag3': 'COXBOX',
'date_block_num': 'COXBOX',
'date_city_subtype_avg_cnt_lag1': 'COXBOX',
'date_item_avg_cnt_lag1': 'COXBOX',
'date_item_avg_cnt_lag2': 'COXBOX',
'date_item_avg_cnt_lag3': 'COXBOX',
'date_itemcat_shop_avg_cnt_lag1': 'COXBOX',
'date_name3_itemcat_avg_cnt_lag2': 'COXBOX',
'date_shop_avg_cnt_lag1': 'COXBOX',
'date_shop_avg_cnt_lag2': 'COXBOX',
'date_shop_avg_cnt_lag3': 'COXBOX',
'date_shop_item_avg_cnt_lag1': 'COXBOX',
'date_shop_item_avg_cnt_lag2': 'COXBOX',
'date_shop_item_avg_cnt_lag3': 'COXBOX',
'date_shop_subtype_avg_cnt_lag1': 'COXBOX',
'date_shopcat_name3_avg_cnt_lag1': 'COXBOX',
'date_shopcat_name3_avg_cnt_lag2': 'COXBOX',
'days': 'COXBOX',
'item_category_id': 'NORM',
'item_first_sale': 'COXBOX',
'item_id': 'NORM',
'item_shop_first_sale': 'COXBOX',
'item_subtype': 'COXBOX',
'item_type': 'COXBOX',
'month': 'COXBOX',
'name2': 'LOG1P',
'name3': 'LOG1P',
'shop_category': 'COXBOX',
'shop_city': 'NORM',
'shop_id': 'NORM'
'''