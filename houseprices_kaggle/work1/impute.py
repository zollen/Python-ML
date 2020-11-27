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
from missingpy import MissForest
from lightgbm import LGBMClassifier, LGBMRegressor





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





        
def fillValue(df, name):
    work_df = df.copy()
    cat_columns = categorical_columns.copy()
    num_columns = numeric_columns.copy()
    catEncoder = None
        
    for col in cat_columns:
        encoder = LabelEncoder()
        if col == name:
            catEncoder = encoder 
        work_df.loc[work_df[col].isna() == False, col] = encoder.fit_transform(
            work_df.loc[work_df[col].isna() == False, col])

    
    print(name)
    if catEncoder != None:
        cat_columns.remove(name)
    else:
        num_columns.remove(name)
    
    columns = num_columns + cat_columns
   
    
    
    for col in columns:
        scaler = RobustScaler()
        work_df.loc[work_df[col].isna() == False, col] = scaler.fit_transform(
            work_df.loc[work_df[col].isna() == False, col].values.reshape(-1, 1)).astype('float64')
    
   
    '''
    https://pypi.org/project/missingpy/
    '''
    imputer = MissForest()
    work_df[columns] = imputer.fit_transform(work_df[columns])
    
    

    learn_df = work_df[work_df[name].isna() == False]
    predict_df = work_df[work_df[name].isna() == True]

    if catEncoder != None:
        model = LGBMClassifier()
    else:
        model = LGBMRegressor()
    
    
    model.fit(learn_df[columns], learn_df[name].astype('int64'))
    predict_df['Prediction'] = model.predict(predict_df[columns])
    
    if catEncoder == None:
        df.loc[df['Id'].isin(predict_df['Id']), name] = predict_df['Prediction']
    else:
        df.loc[df['Id'].isin(predict_df['Id']), name] = catEncoder.inverse_transform(predict_df['Prediction'])
    


kk  = all_df.isnull().sum()
names = []
sums = []
for name, val in kk.items():
    names.append(name)
    sums.append(val)

damn_df = pd.DataFrame()
damn_df['Name'] = names
damn_df['Nulls'] = sums
damn_df = damn_df[damn_df['Nulls'] > 0]
damn_df.sort_values('Nulls', ascending = True, inplace = True)

labels = damn_df['Name'].tolist()
labels.remove('SalePrice')


# features sorted by the least number of missing values to the largest number of missing values    
for name in labels:
    fillValue(all_df, name)
    
print(all_df.isnull().sum())

all_df.to_csv(os.path.join(PROJECT_DIR, 'data/all.csv'), index = False)


    
'''
Strong Correlation
TotalBsmtSF <- 1stFlrSF
SalePrices <- GrLiveArea, TotalBsmtSF, OverallQual
'''
