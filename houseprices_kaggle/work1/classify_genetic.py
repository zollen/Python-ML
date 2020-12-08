'''
Created on Dec. 7, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
import warnings

SEED = 87

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
np.random.seed(SEED)

pp = pprint.PrettyPrinter(indent=3) 
'''
CAT:      0.11947
XGB:      0.11950
LASSO:    0.12150
ELEASTIC: 0.12016
HUBER:    0.13044
ADA:      0.12480
TWEET:    0.12504
SDG:      0.17630
PASSAGG:  0.19055
LINEAR:   0.12315
SVM:      0.12378
=================
BLENDING: 0.11582
'''
FILES = { 
            'cat.csv': 1.0761, 
            'xgb.csv': 0.026, 
            'lasso.csv': -0.0686, 
            'eleasticnet.csv': 0.0208, 
            'linear.csv': -0.0221,
            'svm.csv': -0.0298
        }

PROJECT_DIR=str(Path(__file__).parent.parent)  

result_df = pd.DataFrame()


y_intercept = 0.8881
   

for name in FILES:
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/models/', name))
    df = df.loc[df['Id'] >= 1461, ['Id', 'SalePrice']]
    
    if 'SalePrice' in result_df:
        result_df['SalePrice'] = result_df['SalePrice'] + (FILES[name] * df['SalePrice'])
    else:
        result_df['Id'] = df['Id']
        result_df['SalePrice'] = y_intercept + (FILES[name] * df['SalePrice'])
        

result_df.to_csv(os.path.join(PROJECT_DIR, 'data/results.csv'), index = False)

print("Done")