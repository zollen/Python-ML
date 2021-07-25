'''
Created on Jul. 23, 2021

@author: zollen
'''

import pandas as pd
import time
import futuresales_kaggle.lib.future_lib as ft
import warnings

warnings.filterwarnings('ignore')


train = pd.read_csv('../data/monthly_train.csv')
test = pd.read_csv('../data/monthly_test2.csv')

'''
start_time = time.time()
train, test = ft.add_sales_proximity(10, 
                        train, test)
end_time = time.time()
print(train.head())

print("DONE ", (end_time - start_time))
'''

def calculate_proximity(vals):   
        score = 0
        lgth = len(vals)
        total = lgth**2
        for idx, row in zip(range(1, lgth + 1), vals):
            score += row * idx**2 / total     
        return score

'''

'''    
kk = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(1, 21):
    print(calculate_proximity(kk[0:i]))
