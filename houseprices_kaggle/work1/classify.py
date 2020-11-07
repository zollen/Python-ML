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
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))


train_df.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'], inplace = True)
test_df.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'], inplace = True)



last = 0
for size in range(0, 1200, 100):
    train_df.loc[(train_df['LowQualFinSF'] >= last) & (train_df['LowQualFinSF'] < size), 'LowQualFinSFP'] = size
    test_df.loc[(test_df['LowQualFinSF'] >= last) & (test_df['LowQualFinSF'] < size), 'LowQualFinSFP'] = size
    last = size

last = 0
for size in range(0, 1200, 100):
    train_df.loc[(train_df['WoodDeckSF'] >= last) & (train_df['WoodDeckSF'] < size), 'WoodDeckSFP'] = size
    test_df.loc[(test_df['WoodDeckSF'] >= last) & (test_df['WoodDeckSF'] < size), 'WoodDeckSFP'] = size
    last = size

last = 0
for size in range(0, 2000, 100):
    train_df.loc[(train_df['1stFlrSF'] >= last) & (train_df['1stFlrSF'] < size), '1stFlrSFP'] = size
    test_df.loc[(test_df['1stFlrSF'] >= last) & (test_df['1stFlrSF'] < size), '1stFlrSFP'] = size
    last = size


all_df = pd.concat([ train_df, test_df ]) 

print(test_df[test_df['SaleType'].isna() == True])

grps = all_df.groupby(['YrSold', 'SaleCondition', 'SaleType'])['SaleType'].count()
print(grps)


