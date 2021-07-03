'''
Created on Jun. 27, 2021

@author: zollen
'''

import pandas as pd
import numpy as np
import calendar
import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)



categories = pd.read_csv('../data/item_categories.csv')
items = pd.read_csv('../data/items.csv')
shops = pd.read_csv('../data/shops.csv')
training = pd.read_csv('../data/sales_train.csv')
testing = pd.read_csv('../data/test.csv')


training.drop(columns='date', inplace = True)    

training = training[(training.item_price < 300000 ) & (training.item_cnt_day < 1000)]

training = training[training.item_price > 0].reset_index(drop = True)
#training.loc[training.item_cnt_day < 1, "item_cnt_day"] = 0


# Якутск Орджоникидзе, 56
training.loc[training.shop_id == 0, 'shop_id'] = 57
testing.loc[testing.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
training.loc[training.shop_id == 1, 'shop_id'] = 58
testing.loc[testing.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
training.loc[training.shop_id == 10, 'shop_id'] = 11
testing.loc[testing.shop_id == 10, 'shop_id'] = 11

training.loc[training.item_price < 0, 'item_price'] = 2499

testing['date_block_num'] = 34


'''
there are lot more combos don't show up in test data
Mismatch 307437  total:  418908
26% of training combo are in the test data
'''

all_dates = [ ]
for yr in  [ 2013, 2014, 2015 ]:
    for mn in [1,2,3,4,5,6,7,8,9,10,11,12]:
        dd = calendar.monthrange(yr, mn)
        all_dates.append(f'{yr}-{mn:02d}-{dd[1]}')
        
all_dates.remove('2015-11-30')
all_dates.remove('2015-12-31')

    
def buildCombo(data, prices, indx, rows):
     
    recRef = rows.iloc[0]

    for i in range(len(all_dates)):
        
        targets = rows[rows['date_block_num'] == i]   
              
        rec = {}
        rec['date_block_num'] = i
        rec['shop_id'] = recRef['shop_id']
        rec['item_id'] = recRef['item_id']
        rec['item_price'] = recRef['item_price']
        

        if len(targets) <= 0:
            rec['item_price'] = recRef['item_price']
            rec['item_cnt_day'] = 0
        else:
            cnt = 0
            for _, row in targets.iterrows():
                recRef = row
                rec['item_price'] = row['item_price']
                cnt = cnt + row['item_cnt_day'] 
            rec['item_cnt_day'] = cnt
            
        data[indx] = [ rec['date_block_num'],
                        rec['shop_id'], rec['item_id'], 
                        rec['item_price'], rec['item_cnt_day'] ] 
        
        if i == 33:
            prices[(rec['shop_id'], rec['item_id'])] = rec['item_price']

        indx += 1
         
    return indx


def get_price(rec):
    global prices
    return prices[(rec['shop_id'], rec['item_id'])]

def process(rows):
    global nn, index, prices
    index = buildCombo(nn, prices, index, rows)
    

print("Begin...")
        
nn = np.zeros((len(training['shop_id'].unique()) * 
               len(training['item_id'].unique()) * 
               len(all_dates), 5))
index = 0
prices = {}


training.groupby(['shop_id', 'item_id']).apply(process)

nn = nn[~np.all(nn == 0, axis = 1)]

trainData = pd.DataFrame(data=nn, columns=['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day'])
trainData['date_block_num'] = trainData['date_block_num'].astype('int64')
trainData['shop_id'] = trainData['shop_id'].astype('int64')
trainData['item_id'] = trainData['item_id'].astype('int64')
trainData['item_price'] = trainData['item_price'].astype('float64')
trainData['item_cnt_day'] = trainData['item_cnt_day'].astype('int64')

testing['item_price'] = testing.apply(get_price, axis=1)

print(len(trainData))


trainData.to_csv('../data/train_data.csv', index = False)
testing.to_csv('../data/test_data.csv', index = False)

print("Done")



