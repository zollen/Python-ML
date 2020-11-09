'''
Created on Oct. 31, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sb
import warnings



warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

pp = pprint.PrettyPrinter(indent=3) 

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train_data.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test_data.csv'))





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

#corr = train_df[all_columns].corr()
#mask = np.triu(np.ones_like(corr, dtype=np.bool))    
#plt.figure(figsize=(16, 16))   
#sb.heatmap(corr, mask=mask, cmap='RdBu_r', annot=False, linewidths=0.5, fmt='0.2f')
#plt.show()



            
#for name in ['MiscVal', 'PoolArea', 'LotArea', '3SsnPorch', 'LowQualFinSF',
#             'KitchenAbvGr', 'BsmtFinSF2', 'ScreenPorch', 'BsmtHalfBath',
#             'EnclosedPorch', 'MasVnrArea', 'OpenPorchSF', 'LotFrontage',
#             'BsmtFinSF1', 'WoodDeckSF', 'MSSubClass', '1stFlrSF', 'GrLivArea',
#             '2ndFlrSF', 'OverallQual', 'TotRmsAbvGrd', 'HalfBath', 'Fireplaces',
#              'BsmtFullBath', ]:
#    train_df[name] = np.log1p(train_df[name])
#    test_df[name] = np.log1p(test_df[name])

all_df = pd.concat([ train_df, test_df ])     

all_df = pd.get_dummies(all_df, columns = categorical_columns)

categorical_columns = set(all_df.columns).symmetric_difference(numeric_columns + ['Id', 'SalePrice'])
categorical_columns = list(categorical_columns)

train_df[categorical_columns] = all_df.loc[all_df['Id'].isin(train_df['Id']), categorical_columns]
test_df[categorical_columns] = all_df.loc[all_df['Id'].isin(test_df['Id']), categorical_columns]

all_columns = numeric_columns + categorical_columns


scaler = MinMaxScaler()
train_df[numeric_columns] = scaler.fit_transform(train_df[numeric_columns])
test_df[numeric_columns] = scaler.transform(test_df[numeric_columns])    



model = LGBMRegressor()
model.fit(train_df[all_columns], train_df['SalePrice'])
train_df['Prediction'] = model.predict(train_df[all_columns]).astype('int64')
test_df['SalePrice'] = model.predict(test_df[all_columns]).astype('int64')


print("======================================================")
print("RSME: ", np.sqrt(mean_squared_error(train_df['SalePrice'], train_df['Prediction'])))

test_df[['Id', 'SalePrice']].to_csv(os.path.join(PROJECT_DIR, 'data/results.csv'), index = False)
