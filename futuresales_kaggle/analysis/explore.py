'''
Created on Jun. 27, 2021

@author: zollen
'''

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import calendar
import itertools
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')
sb.set_style('whitegrid')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)



categories = pd.read_csv('../data/item_categories.csv')
items = pd.read_csv('../data/items.csv')
shops = pd.read_csv('../data/shops.csv')
training = pd.read_csv('../data/sales_train.csv')
testing = pd.read_csv('../data/test.csv')


training['date'] = pd.to_datetime(training['date'], format="%d.%m.%Y")



if False:
    plt.figure(figsize=(10,4))
    plt.xlim(-100, 3000)
    flierprops = dict(marker='o', markerfacecolor='purple', markersize=6,
                      linestyle='none', markeredgecolor='black')
    sb.boxplot(x=training.item_cnt_day, flierprops=flierprops)
    
    plt.figure(figsize=(10,4))
    plt.xlim(training.item_price.min(), training.item_price.max()*1.1)
    sb.boxplot(x=training.item_price, flierprops=flierprops)
    

training = training[(training.item_price < 300000 ) & (training.item_cnt_day < 1000)]

training = training[training.item_price > 0].reset_index(drop = True)
training.loc[training.item_cnt_day < 1, "item_cnt_day"] = 0


# Якутск Орджоникидзе, 56
training.loc[training.shop_id == 0, 'shop_id'] = 57
testing.loc[testing.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
training.loc[training.shop_id == 1, 'shop_id'] = 58
testing.loc[testing.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
training.loc[training.shop_id == 10, 'shop_id'] = 11
testing.loc[testing.shop_id == 10, 'shop_id'] = 11

'''
I need baseline evaluation!!!!!!!!!!!!!!

2013-01-01 00:00:00
2015-10-31 00:00:00
'''


all_dates = [ '2012-12-31']
for yr in [2013, 2014, 2015]:
    for mn in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:        
        dd = calendar.monthrange(yr, mn)
        all_dates.append(f'{yr}-{mn:02d}-{dd[1]}')
        
all_dates.remove('2015-11-30')
all_dates.remove('2015-12-31')

'''
there are lot more combos don't show up in test data
Mismatch 307437  total:  418908
26% of training combo are in the test data

cnt = 0
done = {}
for _, trrow in training.iterrows():
    if (trrow['shop_id'], trrow['item_id']) not in done.keys():
        k = testing[(testing['shop_id'] == trrow['shop_id']) & (testing['item_id'] == trrow['item_id'])]
        if len(k) <= 0:
            cnt += 1
        done[(trrow['shop_id'], trrow['item_id'])] = 1
    
print(f"Mismatch {cnt}", " total: ", len(done))
exit()
'''

def buildData(data):

    index = 0    
    all_shops = training['shop_id'].unique()
    all_items = training['item_id'].unique()
    all_params = list(itertools.product(all_shops, all_items))
    
    print("TOTAL: ", len(all_params))
    
    for params in all_params[:200]:
        index, data = buildCombo(data, index, params[0], params[1])
    
       
def buildCombo(data, indx, shopId, itemId):
    
    k = training[(training['shop_id'] == shopId) & (training['item_id'] == itemId)]
   
    if len(k) <= 0:
        return indx, data
    
    recRef = k.iloc[0]

    for i in range(len(all_dates) - 1):
        
        targets = k[(k['date'] > all_dates[i]) & (k['date'] <= all_dates[i + 1])]   
              
        rec = {}
        rec['date'] = pd.to_datetime(all_dates[i + 1], format='%Y-%m-%d')
        rec['date_block_num'] = recRef['date_block_num']
        rec['shop_id'] = recRef['shop_id']
        rec['item_id'] = recRef['item_id']
        rec['item_price'] = recRef['item_price']
        
        
        if len(targets) <= 0:
            rec['item_cnt_day'] = 0
        else:
            cnt = 0
            for _, row in targets.iterrows():
                recRef = row
                cnt = cnt + row['item_cnt_day'] 
            rec['item_cnt_day'] = cnt
            
        data.loc[indx] = [ rec['date'], rec['date_block_num'],
                        rec['shop_id'], rec['item_id'], 
                        rec['item_price'], rec['item_cnt_day'] ] 

        indx += 1
        
        
    
    return indx, data
    

trainData = pd.DataFrame(columns=training.columns)
buildData(trainData)

trainData['date'] = trainData['date'].astype('datetime64')
trainData['date_block_num'] = trainData['date_block_num'].astype('int64')
trainData['shop_id'] = trainData['shop_id'].astype('int64')
trainData['item_id'] = trainData['item_id'].astype('int64')
trainData['item_price'] = trainData['item_price'].astype('float64')
trainData['item_cnt_day'] = trainData['item_cnt_day'].astype('int64')

trainData.to_csv('../data/train_data.csv', index = False)

print("Done")

plt.show()

