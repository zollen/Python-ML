'''
Created on Nov. 2, 2020

@author: zollen
'''
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sb
import warnings
from xgboost.sklearn import XGBRegressor



warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

pp = pprint.PrettyPrinter(indent=3) 

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))


train_df.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'], inplace = True)
test_df.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'], inplace = True)



all_df = pd.concat([ train_df, test_df ]) 


def fillValue(name, fid):
    
    global all_df
    
    col_types = all_df.columns.to_series().groupby(all_df.dtypes)
    categorical_columns = []
    numeric_columns = []
       
    for col in col_types:
        if col[0] == 'object':
            categorical_columns = col[1].unique().tolist()
        else:
            numeric_columns += col[1].unique().tolist()
            
    if name in numeric_columns:
        numeric_columns.remove(name)
        
    if name in categorical_columns:
        categorical_columns.remove(name)
        
    numeric_columns.remove('Id')
    numeric_columns.remove('SalePrice')
    

    damn_df = pd.get_dummies(all_df, columns = categorical_columns) 
    single_df = damn_df[damn_df['Id'] == fid]
    all_ddf = pd.get_dummies(all_df.dropna(), columns = categorical_columns) 
    

    categorical_columns = list(set(all_ddf.columns).symmetric_difference(numeric_columns + ['Id', 'SalePrice', name]))
    all_columns = numeric_columns + categorical_columns
    
    if all_df[name].dtypes == 'object':
        keys = all_ddf[name].unique().tolist()
        
        if np.nan in keys:
            keys.remove(np.nan)
        
        vals = [ i  for i in range(0, len(keys))]
        labs = dict(zip(keys, vals))
        rlabs = dict(zip(vals, keys))
        all_ddf[name] = all_ddf[name].map(labs)
        model = XGBClassifier()
    else:
        model = XGBRegressor()
        
 
    model.fit(all_ddf[all_columns], all_ddf[name])
    prediction = model.predict(single_df[all_columns])

    all_df.loc[all_df['Id'] == fid, name] = prediction[0]
    
    if all_df[name].dtypes == 'object':        
        print("%4d[%12s] ===> %s" %(fid, name, rlabs[prediction[0]]))
    else:
        print("%4d[%12s] ===> %d" %(fid, name, prediction[0]))
        








'''
Strong Correlation
TotalBsmtSF <- 1stFlrSF
SalePrices <- GrLiveArea, TotalBsmtSF, OverallQual
'''
