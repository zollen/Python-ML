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
SVM:      
'''
FILES = { 
            'cat.csv': [ 1.0 ], 
            'xgb.csv': [ 1.0 ], 
            'lasso.csv': [ 1.0 ], 
            'eleasticnet.csv': [ 1.0 ], 
            'huber.csv': [ 1.0 ], 
            'ada.csv': [ 1.0 ], 
            'pass_agg.csv': [ 1.0 ], 
            'sgd.csv': [ 1.0 ], 
            'tweedie.csv': [ 1.0 ], 
            'linear.csv': [ 1.0 ],
            'svm.csv': [1.0 ]
        }

PROJECT_DIR=str(Path(__file__).parent.parent)  


for name in FILES:
    FILES[name].append(pd.read_csv(os.path.join(PROJECT_DIR, 'data/models/', name)))
              
