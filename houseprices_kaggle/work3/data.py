'''
Created on Dec. 8, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import houseprices_kaggle.lib.house_lib as hb
import warnings

SEED = 23

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
np.random.seed(SEED)

ID_FIELD = ['Id']
LABEL_FIELD = ['SalePrice'] 

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))


train_df.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'], inplace = True)
test_df.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'], inplace = True)            

all_df = pd.concat([ train_df, test_df ], ignore_index = True)

all_df.loc[(all_df['BsmtCond'].isna() == True) &
            (all_df['TotalBsmtSF'] == 0), 'BsmtCond'] = 'None'
all_df.loc[(all_df['BsmtQual'].isna() == True) &
            (all_df['TotalBsmtSF'] == 0), 'BsmtQual'] = 'None'
all_df.loc[(all_df['BsmtFinType1'].isna() == True) &
            (all_df['TotalBsmtSF'] == 0), 'BsmtFinType1'] = 'None'
all_df.loc[(all_df['BsmtFinType2'].isna() == True) &
            (all_df['TotalBsmtSF'] == 0), 'BsmtFinType2'] = 'None'
all_df.loc[(all_df['BsmtExposure'].isna() == True) &
            (all_df['TotalBsmtSF'] == 0), 'BsmtExposure'] = 'None'
            
all_df.loc[(all_df['GarageFinish'].isna() == True) & 
             (all_df['GarageArea'] == 0), 'GarageFinish'] = 'None'
all_df.loc[(all_df['GarageType'].isna() == True) & 
             (all_df['GarageArea'] == 0), 'GarageType'] = 'None'
all_df.loc[(all_df['GarageQual'].isna() == True) & 
             (all_df['GarageArea'] == 0), 'GarageQual'] = 'None'    
all_df.loc[(all_df['GarageCond'].isna() == True) & 
             (all_df['GarageArea'] == 0), 'GarageCond'] = 'None'   
all_df.loc[(all_df['GarageYrBlt'].isna() == True) & 
             (all_df['GarageArea'] == 0), 'GarageYrBlt'] = 0

for iid in [2121, 2189]:            
    all_df.loc[all_df['Id'] == iid, 'BsmtExposure'] = 'None'
    all_df.loc[all_df['Id'] == iid, 'BsmtQual'] = 'None'
    all_df.loc[all_df['Id'] == iid, 'BsmtCond'] = 'None'
    all_df.loc[all_df['Id'] == iid, 'BsmtFinType1'] = 'None'
    all_df.loc[all_df['Id'] == iid, 'BsmtFinType2'] = 'None'
    all_df.loc[all_df['Id'] == iid, 'BsmtFinSF1'] = 0
    all_df.loc[all_df['Id'] == iid, 'BsmtFinSF2'] = 0
    all_df.loc[all_df['Id'] == iid, 'BsmtUnfSF'] = 0
    all_df.loc[all_df['Id'] == iid, 'TotalBsmtSF'] = 0
    all_df.loc[all_df['Id'] == iid, 'BsmtFullBath'] = 0
    all_df.loc[all_df['Id'] == iid, 'BsmtHalfBath'] = 0




imputer = hb.MultStageImputer(ID_FIELD + LABEL_FIELD)
all_df = imputer.fit_transform(all_df)

train_df = all_df[all_df['Id'] < 1461]
test_df = all_df[all_df['Id'] >= 1461]

train_df.to_csv(os.path.join(PROJECT_DIR, 'data/train_data.csv'), index = False)
test_df.to_csv(os.path.join(PROJECT_DIR, 'data/test_data.csv'), index = False)

print("DONE")