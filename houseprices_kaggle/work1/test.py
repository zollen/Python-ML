'''
Created on Nov. 2, 2020

@author: zollen
'''
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
import warnings
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
from missingpy import MissForest
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor




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

col_types = all_df.columns.to_series().groupby(all_df.dtypes)
categorical_columns = []
numeric_columns = []
       
for col in col_types:
    if col[0] == 'object':
        categorical_columns = col[1].unique().tolist()
    else:
        numeric_columns += col[1].unique().tolist()


numeric_columns.remove('Id')
numeric_columns.remove('SalePrice')




kk = []
def fillValue(name, fid):
    
    global all_df
    
    num_columns = numeric_columns.copy()
    cat_columns = categorical_columns.copy()
            
    if name in num_columns:
        num_columns.remove(name)
        
    if name in cat_columns:
        cat_columns.remove(name)
    
    all_ddf = all_df.copy()
    
    for colnam in cat_columns:
        keys = all_df[colnam].unique()
        
        if np.nan in keys:
            keys.remove(np.nan)
            
        vals = [ i  for i in range(0, len(keys))]
        labs = dict(zip(keys, vals))
        all_ddf[colnam] = all_ddf[colnam].map(labs)
        
    all_columns = num_columns + cat_columns
            
    '''
    https://pypi.org/project/missingpy/
    '''
    imputer = MissForest()
    all_ddf[all_columns] = imputer.fit_transform(all_ddf[all_columns])
    all_ddf[all_columns] = all_ddf[all_columns].round(0).astype('int64')
    

    all_ddf = pd.get_dummies(all_ddf, columns = cat_columns) 
    single_df = all_ddf[all_ddf['Id'] == fid]  
    all_ddf = all_ddf[all_ddf[name].isna() == False]
    

    cat_columns = list(set(all_ddf.columns).symmetric_difference(num_columns + ['Id', 'SalePrice', name]))
    cat_columns.sort(reverse = True)
    all_columns = num_columns + cat_columns
    
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

        
def fillValue2(df, name):
    work_df = df.copy()
    cat_columns = categorical_columns.copy()
    num_columns = numeric_columns.copy()
    catEncoder = None
        
    for col in cat_columns:
        encoder = LabelEncoder()
        if col == name:
            catEncoder = encoder 
        work_df.loc[work_df[col].isna() == False, col] = encoder.fit_transform(work_df.loc[work_df[col].isna() == False, col])

    if catEncoder != None:
        cat_columns.remove(name)
    else:
        num_columns.remove(name)
    
    columns = num_columns + cat_columns
    
    for col in columns:
        scaler = RobustScaler()
        work_df.loc[work_df[col].isna() == False, col] = scaler.fit_transform(work_df.loc[work_df[col].isna() == False, col].values.reshape(-1, 1))
    
    
    imputer = KNNImputer()
    work_df[columns] = imputer.fit_transform(work_df[columns])

    learn_df = work_df[work_df[name].isna() == False]
    predict_df = work_df[work_df[name].isna() == True]

    if catEncoder != None:
        model = LGBMClassifier()
    else:
        model = LGBMRegressor()
    
    learn_df[name] = learn_df[name].astype('int64')
    model.fit(learn_df[columns], learn_df[name])
    predict_df['Prediction'] = model.predict(predict_df[columns])
    
    if catEncoder == None:
        df.loc[df['Id'].isin(predict_df['Id']), name] = predict_df['Prediction']
    else:
        df.loc[df['Id'].isin(predict_df['Id']), name] = catEncoder.inverse_transform(predict_df['Prediction'])
    



fillValue('GarageArea', 2577)

fillValue('Electrical', 1380)
    
fillValue('GarageFinish', 2127)
fillValue('GarageYrBlt', 2127)
fillValue('GarageFinish', 2577)
fillValue('GarageYrBlt', 2577)

    
'''
Strong Correlation
TotalBsmtSF <- 1stFlrSF
SalePrices <- GrLiveArea, TotalBsmtSF, OverallQual
'''
