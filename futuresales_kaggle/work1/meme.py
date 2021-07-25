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
     REAL   DEMO
0.0  1.23   0.0
0.0  0.88   0.0
0.0  0.59   0.0
1.0  1.36   1.0
0.0  1.00   0.64
0.0  0.72   0.444
0.0  0.51   0.326
0.0  0.36   0.25
0.0  0.25   0.197
0.0  0.16   0.16
0.0  0.09   0.132
0.0  0.04   0.111
0.0  0.01   0.094
0.0  0.00   0.081
0.0  0.00   0.071
0.0  0.00   0.625
0.0  0.00   0.055
0.0  0.00   0.049
0.0  0.00   0.044
0.0  0.00   0.04
'''    
kk = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(1, 21):
    print(calculate_proximity(kk[0:i]))
