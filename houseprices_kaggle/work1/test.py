'''
Created on Nov. 2, 2020

@author: zollen
'''
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
import math
import warnings
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import norm, skew, boxcox_normmax
from scipy.special import boxcox1p



warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
np.random.seed(0)

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
    categorical_columns.sort(reverse = True)
    all_columns = numeric_columns + categorical_columns
    
    if all_df[name].dtypes == 'object':
        keys = all_ddf[name].unique().tolist()

        if np.nan in keys:
            keys.remove(np.nan)
        
        vals = [ i  for i in range(0, len(keys))]
        labs = dict(zip(keys, vals))
        rlabs = dict(zip(vals, keys))
        all_ddf[name] = all_ddf[name].map(labs)
        model = CatBoostClassifier(random_state = 0, verbose = False)
    else:
        model = CatBoostRegressor(random_state = 0, verbose = False)
        

    model.fit(all_ddf[all_columns], all_ddf[name])
    prediction = model.predict(single_df[all_columns])
    
    if all_df[name].dtypes == 'object':  
        prediction = prediction[:, 0]
        print("%4d[%12s] ===> %s" % (fid, name, rlabs[prediction[0]]))
        all_df.loc[all_df['Id'] == fid, name] = rlabs[prediction[0]]
        kk.append(rlabs[prediction[0]])
    else:
        all_df.loc[all_df['Id'] == fid, name] = prediction[0]
        print("%4d[%12s] ===> %d" % (fid, name, prediction[0]))
        kk.append(np.round(prediction[0], 0))

        
col_types = all_df.columns.to_series().groupby(all_df.dtypes)
numeric_columns = []
       
for col in col_types:
    if col[0] != 'object':
        numeric_columns += col[1].unique().tolist()

numeric_columns.remove('Id')
numeric_columns.remove('SalePrice')

def deskew(df):
    
    for name in numeric_columns:
        col_df = pd.DataFrame()
    
        all_df_na = all_df.loc[all_df[name].isna() == False, name]
        df_na = df.loc[df[name].isna() == False, name]

        col_df['NORM'] = df_na.values
        col_df['LOG1P'] = df_na.apply(lambda x : np.log1p(x)).values
        col_df['CB'] = boxcox1p(df_na, boxcox_normmax(df_na + 1)).values
        col_df['MLE'] = boxcox1p(df_na, boxcox_normmax(df_na + 1, method = 'mle')).values
        col_df['CBA'] = boxcox1p(df_na, boxcox_normmax(all_df_na + 1)).values
    
        nums = []
    
        nums.append(np.abs(skew(col_df['NORM'])))
        nums.append(np.abs(skew(col_df['LOG1P'])))
        nums.append(np.abs(skew(col_df['CB'])))
        nums.append(np.abs(skew(col_df['MLE'])))
        nums.append(np.abs(skew(col_df['CBA'])))
    
        nums  = [999 if math.isnan(x) else x for x in nums]
        
    
        smallest = nums.index(min(nums))
        if smallest == 0:
            df.loc[df[name].isna() == False, name] = col_df['NORM']
        elif smallest == 1:
            df.loc[df[name].isna() == False, name] = col_df['LOG1P']
        elif smallest == 2:
            df.loc[df[name].isna() == False, name] = col_df['CB']
        elif smallest == 3:
            df.loc[df[name].isna() == False, name] = col_df['MLE']
        else:
            df.loc[df[name].isna() == False, name] = col_df['CBA']
            
        print("[%s] [%d]:  %0.4f, %0.4f, %0.4f, %0.4f, %0.4f" % 
              (name, smallest, nums[0], nums[1], nums[2], nums[3], nums[4]))
            



'''
Strong Correlation
TotalBsmtSF <- 1stFlrSF
SalePrices <- GrLiveArea, TotalBsmtSF, OverallQual
'''
