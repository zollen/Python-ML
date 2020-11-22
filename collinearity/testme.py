'''
Created on Nov. 21, 2020

@author: zollen
'''
import os
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor

'''
Calculating VIF (Variable Inflation Factors)
VIF = 1 / ( 1 - R^2)
So, the closer the R^2 value to 1, the higher the value of VIF and the higher the 
    multicollinearity with the particular independent variable.

1. VIF starts at 1 and has no upper limit
2. VIF = 1, no correlation between the independent variable and the other variables
3. VIF exceeding 5 or 10 indicates high multicollinearity between this independent variable and 
    the others
'''


warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.2f}'.format)

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'houseprices_kaggle/data/train.csv'))

last = 0
for val in range(0, 222000, 1000):
    train_df.loc[(train_df['LotArea'] >= last) & (train_df['LotArea'] < val), 'LotAreaP'] = val
    last = val
    
all_df = pd.concat([ train_df ]) 


'''
Fill BsmtQual, BsmtCond, BsmtFinType2, BsmtExposure
'''
train_df.loc[(train_df['BsmtCond'].isna() == True) &
            (train_df['TotalBsmtSF'] == 0), 'BsmtCond'] = 'None'
train_df.loc[(train_df['BsmtQual'].isna() == True) &
            (train_df['TotalBsmtSF'] == 0), 'BsmtQual'] = 'None'
train_df.loc[(train_df['BsmtFinType1'].isna() == True) &
            (train_df['TotalBsmtSF'] == 0), 'BsmtFinType1'] = 'None'
train_df.loc[(train_df['BsmtFinType2'].isna() == True) &
            (train_df['TotalBsmtSF'] == 0), 'BsmtFinType2'] = 'None'
train_df.loc[(train_df['BsmtExposure'].isna() == True) &
            (train_df['TotalBsmtSF'] == 0), 'BsmtExposure'] = 'None'



# OverallQual(8), OverallCond(5), TotalBsmtSF(3206), BsmtQual(Gd), BsmtCond(TA), BsmtExposure(No), BsmtFinType2(NaN)
train_df.loc[train_df['Id'] == 333, 'BsmtFinType2'] = 'BLQ'

# OverallQual(7), OverallCond(5), TotalBsmtSF(936), BsmtQual(Gd)  BsmtCond(TA) BsmtExposure(NaN)
train_df.loc[train_df['Id'] == 949, 'BsmtExposure'] = 'No'



train_df.loc[(train_df['GarageFinish'].isna() == True) & 
             (train_df['GarageArea'] == 0), 'GarageFinish'] = 'None'
train_df.loc[(train_df['GarageType'].isna() == True) & 
             (train_df['GarageArea'] == 0), 'GarageType'] = 'None'
train_df.loc[(train_df['GarageQual'].isna() == True) & 
             (train_df['GarageArea'] == 0), 'GarageQual'] = 'None'    
train_df.loc[(train_df['GarageCond'].isna() == True) & 
             (train_df['GarageArea'] == 0), 'GarageCond'] = 'None'   
train_df.loc[(train_df['GarageYrBlt'].isna() == True) & 
             (train_df['GarageArea'] == 0), 'GarageYrBlt'] = 0


             




'''
Fill Electrical
'''
# MSSubClass(80) MSZoing(RL) OverallQual(5) YearBuilt(2006) MasVnrArea(Sbrkr)
train_df.loc[train_df['Electrical'].isna() == True, 'Electrical'] = 'Mix'



'''
Fill MasVnrType
'''
train_df.loc[train_df['Id'] == 235, 'MasVnrType'] = 'BrkFace'
train_df.loc[train_df['Id'] == 530, 'MasVnrType'] = 'BrkFace'
train_df.loc[train_df['Id'] == 651, 'MasVnrType'] = 'Stone'
train_df.loc[train_df['Id'] == 937, 'MasVnrType'] = 'BrkFace'
train_df.loc[train_df['Id'] == 974, 'MasVnrType'] = 'Stone'
train_df.loc[train_df['Id'] == 978, 'MasVnrType'] = 'Stone'
train_df.loc[train_df['Id'] == 1244, 'MasVnrType'] = 'Stone'
train_df.loc[train_df['Id'] == 1279, 'MasVnrType'] = 'BrkFace'


             
'''
Fill MasVnrArea
'''
train_df.loc[train_df['Id'] == 235, 'MasVnrArea'] = 168
train_df.loc[train_df['Id'] == 530, 'MasVnrArea'] = 568
train_df.loc[train_df['Id'] == 651, 'MasVnrArea'] = 127
train_df.loc[train_df['Id'] == 937, 'MasVnrArea'] = 119
train_df.loc[train_df['Id'] == 974, 'MasVnrArea'] = 124
train_df.loc[train_df['Id'] == 978, 'MasVnrArea'] = 140
train_df.loc[train_df['Id'] == 1244, 'MasVnrArea'] = 333
train_df.loc[train_df['Id'] == 1279, 'MasVnrArea'] = 220

       
       
'''
Fill LotFrontage
'''          
def fillLotFrontage(df1):
    
    grps = all_df.groupby(['OverallQual', 'LotAreaP', 
                           'LotConfig'])['LotFrontage'].median()
    for index, _ in grps.items():
        for df in  [ df1 ]:
            df.loc[(df['LotFrontage'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['LotAreaP'] == index[1]) &
                 (df['LotConfig'] == index[2]), 'LotFrontage'] = grps[index[0], index[1], index[2]]
      
    grps = all_df.groupby(['OverallQual', 'LotAreaP'])['LotFrontage'].median()
    for index, _ in grps.items():
        for df in  [ df1 ]:
            df.loc[(df['LotFrontage'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['LotAreaP'] == index[1]), 'LotFrontage'] = grps[index[0], index[1]]
      
    grps = all_df.groupby(['OverallQual'])['LotFrontage'].median()
    for index, _ in grps.items():
        for df in  [ df1 ]:
            df.loc[(df['LotFrontage'].isna() == True) &
                 (df['OverallQual'] == index), 'LotFrontage'] = grps[index]
                 
    
                 
fillLotFrontage(train_df)  

       
       


'''
Fill FireplaceQu
'''
train_df.loc[train_df['Fireplaces'] == 0, 'FireplaceQu'] = 'None'









'''
Fill Utilities
'''
train_df.loc[train_df['Utilities'].isna() == True, 'Utilities'] = 'AllPub'


    
   







'''
Add TotalSF
'''
train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']




'''
Merge YrSold and MoSold
'''
def mergeSold(rec):
    yrSold = rec['YrSold']
    moSold = rec['MoSold']
    
    years = {2006: 0, 2007: 1, 2008: 2, 2009: 3, 2010: 4}
    
    return round(years[yrSold] + (moSold / 12), 2)
      
train_df['YrSold'] = train_df.apply(mergeSold, axis = 1)





'''
Remove support features
'''
train_df.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MoSold', 'LotAreaP'], inplace = True)





col_types = train_df.columns.to_series().groupby(train_df.dtypes)
numeric_columns = []
       
for col in col_types:
    if col[0] != 'object':
        numeric_columns += col[1].unique().tolist()

numeric_columns.remove('Id')
numeric_columns.remove('SalePrice')


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

numeric_columns.remove('Id')
numeric_columns.remove('SalePrice')



all_columns = numeric_columns + categorical_columns
print(all_columns)
print(train_df[all_columns].head())

print(calc_vif(train_df[all_columns]))