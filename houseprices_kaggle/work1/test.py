'''
Created on Nov. 2, 2020

@author: zollen
'''
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier
import warnings
from sklearn.impute import SimpleImputer
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
    
    all_ddf = all_df.copy()
    
    for colnam in categorical_columns:
        keys = all_df[colnam].unique()
        
        if np.nan in keys:
            keys.remove(np.nan)
            
        vals = [ i  for i in range(0, len(keys))]
        labs = dict(zip(keys, vals))
        all_ddf[colnam] = all_ddf[colnam].map(labs)
            
    
    imputer = SimpleImputer()
    all_ddf[categorical_columns + numeric_columns] = imputer.fit_transform(all_ddf[categorical_columns + numeric_columns])
    all_ddf[categorical_columns + numeric_columns] = all_ddf[categorical_columns + numeric_columns].round(0).astype('int64')
    

    all_ddf = pd.get_dummies(all_ddf, columns = categorical_columns) 
    single_df = all_ddf[all_ddf['Id'] == fid]  
    all_ddf = all_ddf[all_ddf[name].isna() == False]
    

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
        model = XGBClassifier(random_state = 87)
    else:
        model = XGBRegressor(random_state = 87)
        

    model.fit(all_ddf[all_columns], all_ddf[name])
    prediction = model.predict(single_df[all_columns])

    all_df.loc[all_df['Id'] == fid, name] = prediction[0]
    
    if all_df[name].dtypes == 'object':        
        print("%4d[%12s] ===> %s" %(fid, name, rlabs[prediction[0]]))
    else:
        print("%4d[%12s] ===> %d" %(fid, name, prediction[0]))
        


n_folds = 5

def rmsle_cv(model, data, label):
    kf = KFold(n_folds, shuffle=True, random_state=87).get_n_splits(data.values)
    rmse= np.sqrt(-cross_val_score(model, data.values, label, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


print(all_df.loc[all_df['Id'].isin([2041, 2186, 2525, 2218, 2219, 333, 949, 1488, 2349, 2121, 2189]), ['Id', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']])


'''
Id = 2121 may have no basement!!!
'''

'''
Strong Correlation
TotalBsmtSF <- 1stFlrSF
SalePrices <- GrLiveArea, TotalBsmtSF, OverallQual
'''
