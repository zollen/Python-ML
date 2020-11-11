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

kk = []
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
        all_df.loc[all_df['Id'] == fid, name] = rlabs[prediction[0]]
    else:
        all_df.loc[all_df['Id'] == fid, name] = prediction[0]
        print("%4d[%12s] ===> %d" %(fid, name, prediction[0]))
        kk.append(np.round(prediction[0], 0))
        


for fid in [18, 40, 91, 103, 157, 183, 260, 343, 363, 372, 393, 521, 533, 534, 554, 647, 706, 737, 750, 779, 869, 895, 898, 985, 1001, 1012, 1036, 1046, 1049, 1050, 1091, 1180, 1217, 1219, 1233, 1322, 1413, 1586, 1594, 1730, 1779, 1815, 1848, 1849, 1857, 1858, 1859, 1861, 1916, 2041, 2051, 2067, 2069, 2121, 2123, 2186, 2189, 2190, 2191, 2194, 2217, 2225, 2388, 2436, 2453, 2454, 2491, 2499, 2525, 2548, 2553, 2565, 2579, 2600, 2703, 2764, 2767, 2804, 2805, 2825, 2892, 2905]:
    fillValue('BsmtCond', fid)
    


print(kk)

'''
Strong Correlation
TotalBsmtSF <- 1stFlrSF
SalePrices <- GrLiveArea, TotalBsmtSF, OverallQual
'''
