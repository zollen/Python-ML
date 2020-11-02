'''
Created on Oct. 31, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import warnings


warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

pp = pprint.PrettyPrinter(indent=3) 

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

col_types = train_df.columns.to_series().groupby(train_df.dtypes)
numeric_columns = []
for col in col_types:
    if col[0] == 'object':
        categorical_columns = col[1].unique().tolist()
    else:
        numeric_columns += col[1].unique().tolist()


for name in categorical_columns:   
    keys = train_df[name].unique().tolist()
        
    if np.nan in keys:
        keys.remove(np.nan)
    
    vals = [ i  for i in range(0, len(keys))]
    labs = dict(zip(keys, vals))
    train_df[name] = train_df[name].map(labs)
    test_df[name] = test_df[name].map(labs)


numeric_columns.remove('Id')
numeric_columns.remove('SalePrice')

all_columns = numeric_columns + categorical_columns

imputer = SimpleImputer()
imputer.fit(train_df[all_columns])
train_df[all_columns] = imputer.transform(train_df[all_columns])
test_df[all_columns] = imputer.transform(test_df[all_columns])

model = LGBMRegressor()
model.fit(train_df[all_columns], train_df['SalePrice'])
train_df['Prediction'] = model.predict(train_df[all_columns]).astype('int64')
test_df['SalePrice'] = model.predict(test_df[all_columns]).astype('int64')

print(train_df[['Id', 'SalePrice', 'Prediction']])
print("======================================================")
print("RSME: ", np.sqrt(mean_squared_error(train_df['SalePrice'], train_df['Prediction'])))

test_df[['Id', 'SalePrice']].to_csv(os.path.join(PROJECT_DIR, 'data/results.csv'), index = False)
