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
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))



train_df.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'], inplace = True)
test_df.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'], inplace = True)


last = 0
for val in range(1000, 2020, 10):
    train_df.loc[(train_df['YearBuilt'] >= last) & (train_df['YearBuilt'] < val), 'YearBuiltP'] = val
    test_df.loc[(test_df['YearBuilt'] >= last) & (test_df['YearBuilt'] < val), 'YearBuiltP'] = val
    last = val

last = 0
for val in range(0, 2000, 200):
    train_df.loc[(train_df['MasVnrArea'] >= last) & (train_df['MasVnrArea'] < val), 'MasVnrAreaP'] = val
    test_df.loc[(test_df['MasVnrArea'] >= last) & (test_df['MasVnrArea'] < val), 'MasVnrAreaP'] = val
    last = val    
    
last = 0
for val in range(0, 3000, 200):
    train_df.loc[(train_df['1stFlrSF'] >= last) & (train_df['1stFlrSF'] < val), '1stFlrSFP'] = val
    test_df.loc[(test_df['1stFlrSF'] >= last) & (test_df['1stFlrSF'] < val), '1stFlrSFP'] = val
    last = val    
    
last = 0
for val in range(0, 2000, 200):
    train_df.loc[(train_df['WoodDeckSF'] >= last) & (train_df['WoodDeckSF'] < val), 'WoodDeckSFP'] = val
    test_df.loc[(test_df['WoodDeckSF'] >= last) & (test_df['WoodDeckSF'] < val), 'WoodDeckSFP'] = val
    last = val       
    
last = 0
for val in range(0, 1200, 200):
    train_df.loc[(train_df['LowQualFinSF'] >= last) & (train_df['LowQualFinSF'] < val), 'LowQualFinSFP'] = val
    test_df.loc[(test_df['LowQualFinSF'] >= last) & (test_df['LowQualFinSF'] < val), 'LowQualFinSFP'] = val
    last = val

last = 0
for val in range(0, 2000, 200):
    train_df.loc[(train_df['OpenPorchSF'] >= last) & (train_df['OpenPorchSF'] < val), 'OpenPorchSFP'] = val
    test_df.loc[(test_df['OpenPorchSF'] >= last) & (test_df['OpenPorchSF'] < val), 'OpenPorchSFP'] = val
    last = val


    
all_df = pd.concat([ train_df, test_df ]) 


'''
Fill BsmtQual, BsmtCond, BsmtFinType2, BsmtExposure
'''
train_df.loc[(train_df['BsmtCond'].isna() == True) &
            (train_df['TotalBsmtSF'] == 0), 'BsmtCond'] = 'NA'
train_df.loc[(train_df['BsmtQual'].isna() == True) &
            (train_df['TotalBsmtSF'] == 0), 'BsmtQual'] = 'NA'
train_df.loc[(train_df['BsmtFinType1'].isna() == True) &
            (train_df['TotalBsmtSF'] == 0), 'BsmtFinType1'] = 'NA'
train_df.loc[(train_df['BsmtFinType2'].isna() == True) &
            (train_df['TotalBsmtSF'] == 0), 'BsmtFinType2'] = 'NA'
train_df.loc[(train_df['BsmtExposure'].isna() == True) &
            (train_df['TotalBsmtSF'] == 0), 'BsmtExposure'] = 'NA'

test_df.loc[(test_df['BsmtCond'].isna() == True) &
            (test_df['TotalBsmtSF'] == 0), 'BsmtCond'] = 'NA'
test_df.loc[(test_df['BsmtQual'].isna() == True) &
            (test_df['TotalBsmtSF'] == 0), 'BsmtQual'] = 'NA'
test_df.loc[(test_df['BsmtFinType1'].isna() == True) &
            (test_df['TotalBsmtSF'] == 0), 'BsmtFinType1'] = 'NA'
test_df.loc[(test_df['BsmtFinType2'].isna() == True) &
            (test_df['TotalBsmtSF'] == 0), 'BsmtFinType2'] = 'NA'
test_df.loc[(test_df['BsmtExposure'].isna() == True) &
            (test_df['TotalBsmtSF'] == 0), 'BsmtExposure'] = 'NA'
               
all_df.loc[(all_df['BsmtCond'].isna() == True) &
            (all_df['TotalBsmtSF'] == 0), 'BsmtCond'] = 'NA'
all_df.loc[(all_df['BsmtQual'].isna() == True) &
            (all_df['TotalBsmtSF'] == 0), 'BsmtQual'] = 'NA'
all_df.loc[(all_df['BsmtFinType1'].isna() == True) &
            (all_df['TotalBsmtSF'] == 0), 'BsmtFinType1'] = 'NA'
all_df.loc[(all_df['BsmtFinType2'].isna() == True) &
            (all_df['TotalBsmtSF'] == 0), 'BsmtFinType2'] = 'NA'
all_df.loc[(all_df['BsmtExposure'].isna() == True) &
            (all_df['TotalBsmtSF'] == 0), 'BsmtExposure'] = 'NA'
             
# OverallQual(8), OverallCond(9), BsmtQual(Gd), TotalBsmtSF(1426), BsmtExposure(Mn), BsmtCond(Nan)
test_df.loc[test_df['Id'] == 2041, 'BsmtCond'] = 'Gd'

# OverallQual(6), OverallCond(6), BsmtQual(TA), TotalBsmtSF(1127), BsmtExposure(No), BsmtCond(Nan)
test_df.loc[test_df['Id'] == 2186, 'BsmtCond'] = 'TA'

# OverallQual(5), OverallCond(7), BsmtQual(TA), TotalBsmtSF(995), BsmtExposure(Av), BsmtQual(NaN)
test_df.loc[test_df['Id'] == 2525, 'BsmtCond'] = 'TA'

# OverallQual(4), OverallCond(7), BsmtCond(Fa), TotalBsmtSF(173), BsmtExposure(No), BsmtQual(NaN)
test_df.loc[test_df['Id'] == 2218, 'BsmtQual'] = 'Fa'

# OverallQual(4), OverallCond(7), BsmtCond(TA), TotalBsmtSF(356), BsmtExposure(No), BsmtQual(NaN)
test_df.loc[test_df['Id'] == 2219, 'BsmtQual'] = 'TA'

# OverallQual(8), OverallCond(5), TotalBsmtSF(3206), BsmtQual(Gd), BsmtCond(TA), BsmtExposure(No), BsmtFinType2(NaN)
train_df.loc[train_df['Id'] == 333, 'BsmtFinType2'] = 'Unf'

# OverallQual(7), OverallCond(5), TotalBsmtSF(936), BsmtQual(Gd)  BsmtCond(TA) BsmtExposure(NaN)
train_df.loc[train_df['Id'] == 949, 'BsmtExposure'] = 'No'

# OverallQual(8), OverallCond(5), TotalBsmtSF(1595), BsmtQual(Gd)  BsmtCond(TA) BsmtExposure(NaN)
test_df.loc[test_df['Id'] == 1488, 'BsmtExposure'] = 'No'

# OverallQual(5), OverallCond(5), TotalBsmtSF(725), BsmtQual(Gd)  BsmtCond(TA) BsmtExposure(NaN)
test_df.loc[test_df['Id'] == 2349, 'BsmtExposure'] = 'No'

# No basement
test_df.loc[(test_df['Id'] == 2121) | (test_df['Id'] == 2189), ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']] = 'NA'
test_df.loc[(test_df['Id'] == 2121) | (test_df['Id'] == 2189), ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = 0           

'''
Fill GarageFinish, GarageType, GarageQual, GarageCond
'''
test_df.loc[test_df['Id'] == 2127, ['GarageCond', 'GarageQual']] = 'TA'

# MSSubClass(70) MSZoning(RM) LotArea(9060) YearBuilt(1923) GarageType(Detchd)
test_df.loc[test_df['Id'] == 2577, 'GarageQual'] = 'Fa'
test_df.loc[test_df['Id'] == 2577, 'GarageCond'] = 'TA'
test_df.loc[test_df['Id'] == 2577, 'GarageArea'] = 546
test_df.loc[test_df['Id'] == 2577, 'GarageYrBlt'] = 1925
test_df.loc[test_df['Id'] == 2577, 'GarageFinish'] = 'Unf'
test_df.loc[test_df['Id'] == 2577, 'GarageCars'] = 2

train_df.loc[(train_df['GarageFinish'].isna() == True) & 
             (train_df['GarageArea'] == 0), 'GarageFinish'] = 'NA'
train_df.loc[(train_df['GarageType'].isna() == True) & 
             (train_df['GarageArea'] == 0), 'GarageType'] = 'NA'
train_df.loc[(train_df['GarageQual'].isna() == True) & 
             (train_df['GarageArea'] == 0), 'GarageQual'] = 'NA'    
train_df.loc[(train_df['GarageCond'].isna() == True) & 
             (train_df['GarageArea'] == 0), 'GarageCond'] = 'NA'   
train_df.loc[(train_df['GarageYrBlt'].isna() == True) & 
             (train_df['GarageArea'] == 0), 'GarageYrBlt'] = 0

test_df.loc[(test_df['GarageFinish'].isna() == True) & 
             (test_df['GarageArea'] == 0), 'GarageFinish'] = 'NA'
test_df.loc[(test_df['GarageType'].isna() == True) & 
             (test_df['GarageArea'] == 0), 'GarageType'] = 'NA'
test_df.loc[(test_df['GarageQual'].isna() == True) & 
             (test_df['GarageArea'] == 0), 'GarageQual'] = 'NA'    
test_df.loc[(test_df['GarageCond'].isna() == True) & 
             (test_df['GarageArea'] == 0), 'GarageCond'] = 'NA' 
test_df.loc[(test_df['GarageYrBlt'].isna() == True) & 
             (test_df['GarageArea'] == 0), 'GarageYrBlt'] = 0
             
all_df.loc[(all_df['GarageFinish'].isna() == True) & 
             (all_df['GarageArea'] == 0), 'GarageFinish'] = 'NA'
all_df.loc[(all_df['GarageType'].isna() == True) & 
             (all_df['GarageArea'] == 0), 'GarageType'] = 'NA'
all_df.loc[(all_df['GarageQual'].isna() == True) & 
             (all_df['GarageArea'] == 0), 'GarageQual'] = 'NA'    
all_df.loc[(all_df['GarageCond'].isna() == True) & 
             (all_df['GarageArea'] == 0), 'GarageCond'] = 'NA' 
all_df.loc[(all_df['GarageYrBlt'].isna() == True) & 
             (all_df['GarageArea'] == 0), 'GarageYrBlt'] = 0       



    



'''
Fill GarageType
'''
def fillGarageType(df1, df2):
       
    grps = all_df.groupby(['OverallQual', 'YearBuiltP', 'MasVnrAreaP', 'GarageType'])['GarageType'].count()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['GarageType'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['YearBuiltP'] == index[1]) &
                 (df['MasVnrAreaP'] == index[2]), 'GarageType'] = grps[index[0], index[1], index[2]].idxmax()
                 
    grps = all_df.groupby(['OverallQual', 'YearBuiltP', 'GarageType'])['GarageType'].count()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['GarageType'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['YearBuiltP'] == index[1]), 'GarageType'] = grps[index[0], index[1]].idxmax()
                 
    grps = all_df.groupby(['OverallQual', 'GarageType'])['GarageType'].count()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['GarageType'].isna() == True) &
                 (df['OverallQual'] == index[0]), 'GarageType'] = grps[index[0]].idxmax()
                 
    
fillGarageType(train_df, test_df)




'''
Fill GarageFinish
'''
def fillGarageFinish(df1, df2):
       
    grps = all_df.groupby(['OverallQual', 'YearBuiltP', 'MasVnrAreaP', 'GarageFinish'])['GarageFinish'].count()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['GarageFinish'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['YearBuiltP'] == index[1]) &
                 (df['MasVnrAreaP'] == index[2]), 'GarageFinish'] = grps[index[0], index[1], index[2]].idxmax()
                 
    
    grps = all_df.groupby(['OverallQual', 'YearBuiltP', 'GarageFinish'])['GarageFinish'].count()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['GarageFinish'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['YearBuiltP'] == index[1]), 'GarageFinish'] = grps[index[0], index[1]].idxmax()
                 
    
    grps = all_df.groupby(['OverallQual', 'GarageFinish'])['GarageFinish'].count()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['GarageFinish'].isna() == True) &
                 (df['OverallQual'] == index[0]), 'GarageFinish'] = grps[index[0]].idxmax()
                 
    
    
fillGarageFinish(train_df, test_df)


'''
Fill GarageYrBlt
'''
def fillGarageYrBlt(df1, df2):
    
    grps = all_df.groupby(['OverallQual', 'YearBuiltP'])['GarageYrBlt'].median()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['GarageYrBlt'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['YearBuiltP'] == index[1]), 'GarageYrBlt'] = grps[index[0], index[1]]
     
    grps = all_df.groupby(['OverallQual'])['GarageYrBlt'].median()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['GarageYrBlt'].isna() == True) &
                   (df['OverallQual'] == index), 'GarageYrBlt'] = grps[index]
                   
    df1['GarageYrBlt'] = df1['GarageYrBlt'].round(0)
    df2['GarageYrBlt'] = df2['GarageYrBlt'].round(0)
    

fillGarageYrBlt(train_df, test_df)




'''
Fill Electrical
'''
# MSSubClass(80) MSZoing(RL) OverallQual(5) YearBuilt(2006) MasVnrArea(Sbrkr)
train_df.loc[train_df['Electrical'].isna() == True, 'Electrical'] = 1.0



'''
Fill MasVnrType
'''
def fillMasVnrType(df1, df2):
    
    grps = all_df.groupby(['OverallQual', 'MSZoning', 'Exterior1st', 'Exterior2nd', 'MasVnrType'])['MasVnrType'].count()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['MasVnrType'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['MSZoning'] == index[1]) & 
                 (df['Exterior1st'] == index[2]) &
                 (df['Exterior2nd'] == index[3]), 'MasVnrType'] = grps[index[0], index[1], index[2], index[3]].idxmax()
                 
    grps = all_df.groupby(['OverallQual', 'MSZoning', 'Exterior2nd', 'MasVnrType'])['MasVnrType'].count()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['MasVnrType'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['MSZoning'] == index[1]) & 
                 (df['Exterior2nd'] == index[2]), 'MasVnrType'] = grps[index[0], index[1], index[2]].idxmax()
                 
    grps = all_df.groupby(['OverallQual', 'MSZoning', 'MasVnrType'])['MasVnrType'].count()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['MasVnrType'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['MSZoning'] == index[1]), 'MasVnrType'] = grps[index[0], index[1]].idxmax()
    
fillMasVnrType(train_df, test_df)

             
             
'''
Fill MasVnrArea
'''
def fillMasVnrArea(df1, df2):
     
    grps = all_df.groupby(['OverallQual', '1stFlrSFP', 'TotRmsAbvGrd', 
                           'GarageCars', 'WoodDeckSFP', 'MasVnrArea'])['MasVnrArea'].count()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['MasVnrArea'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['1stFlrSFP'] == index[1]) & 
                 (df['TotRmsAbvGrd'] == index[2]) &
                 (df['GarageCars'] == index[3]) &
                 (df['WoodDeckSFP'] == index[4]), 'MasVnrArea'] = grps[index[0], index[1], index[2], index[3], index[4]].idxmax()
    
    grps = all_df.groupby(['OverallQual', 'MasVnrArea'])['MasVnrArea'].count()             
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['MasVnrArea'].isna() == True) &
                 (df['OverallQual'] == index[0]), 'MasVnrArea'] = grps[index[0]].idxmax()
                    
fillMasVnrArea(train_df, test_df)             
          

       
       
'''
Fill LotFrontage
'''       
def fillLotFrontage(df1, df2):
    
    grps = all_df.groupby(['OverallQual', 'MSSubClass', '1stFlrSFP', 'LowQualFinSFP',
                           'OpenPorchSFP', 'GarageCars'])['LotFrontage'].median()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['LotFrontage'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['MSSubClass'] == index[1]) & 
                 (df['1stFlrSFP'] == index[2]) &
                 (df['LowQualFinSFP'] == index[3]) &
                 (df['OpenPorchSFP'] == index[4]) &
                 (df['GarageCars'] == index[5]), 'LotFrontage'] = grps[index[0], index[1], index[2], index[3], index[4], index[5]]
      
    grps = all_df.groupby(['OverallQual', 'MSSubClass', '1stFlrSFP', 'LowQualFinSFP',
                           'OpenPorchSFP'])['LotFrontage'].median()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['LotFrontage'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['MSSubClass'] == index[1]) & 
                 (df['1stFlrSFP'] == index[2]) &
                 (df['LowQualFinSFP'] == index[3]) &
                 (df['OpenPorchSFP'] == index[4]), 'LotFrontage'] = grps[index[0], index[1], index[2], index[3], index[4]]
                 
    grps = all_df.groupby(['OverallQual', 'MSSubClass', '1stFlrSFP', 'LowQualFinSFP'
                           ])['LotFrontage'].median()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['LotFrontage'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['MSSubClass'] == index[1]) & 
                 (df['1stFlrSFP'] == index[2]) &
                 (df['LowQualFinSFP'] == index[3]), 'LotFrontage'] = grps[index[0], index[1], index[2], index[3]]
                 
    grps = all_df.groupby(['OverallQual', 'MSSubClass', '1stFlrSFP'])['LotFrontage'].median()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['LotFrontage'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['MSSubClass'] == index[1]) & 
                 (df['1stFlrSFP'] == index[2]), 'LotFrontage'] = grps[index[0], index[1], index[2]]
                 
    grps = all_df.groupby(['OverallQual', 'MSSubClass'])['LotFrontage'].median()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['LotFrontage'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['MSSubClass'] == index[1]), 'LotFrontage'] = grps[index[0], index[1]]
                 
    grps = all_df.groupby(['OverallQual'])['LotFrontage'].median()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['LotFrontage'].isna() == True) &
                 (df['OverallQual'] == index), 'LotFrontage'] = grps[index]
                 
fillLotFrontage(train_df, test_df)  

       
       


'''
Fill FireplaceQu
'''
train_df.loc[train_df['Fireplaces'] == 0, 'FireplaceQu'] = 'NA'
test_df.loc[test_df['Fireplaces'] == 0, 'FireplaceQu'] = 'NA'
all_df.loc[all_df['Fireplaces'] == 0, 'FireplaceQu'] = 'NA'



'''
Fill MSZoning   
'''
def fillMSZoning(df1, df2):
      
    grps = all_df.groupby(['MSSubClass', 'Neighborhood', 'Condition1', 'Condition2',
                           'MSZoning'])['MSZoning'].count()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['MSZoning'].isna() == True) &
                 (df['MSSubClass'] == index[0]) &
                 (df['Neighborhood'] == index[1]) & 
                 (df['Condition1'] == index[2]) &
                 (df['Condition2'] == index[3]), 'MSZoning'] = grps[index[0], index[1], index[2], index[3]].idxmax()

fillMSZoning(train_df, test_df)



'''
Fill Utilities
'''
train_df.loc[train_df['Utilities'].isna() == True, 'Utilities'] = 'AllPub'
test_df.loc[test_df['Utilities'].isna() == True, 'Utilities'] = 'AllPub'
all_df.loc[all_df['Utilities'].isna() == True, 'Utilities'] = 'AllPub'
    
    

'''
Fill Exterior1st & Exterior2nd
'''
test_df.loc[(test_df['Exterior1st'].isna() == True) &
            (test_df['Exterior2nd'].isna() == True), ['Exterior1st', 'Exterior2nd']] = 'Plywood'



'''
Fill KitchenQual
'''
test_df.loc[test_df['KitchenQual'].isna() == True, 'KitchenQual'] = 'TA'




'''
Fill Functional
'''
test_df.loc[test_df['Functional'].isna() == True, 'Functional'] = 'Typ'



'''
Fill SaleType
'''
test_df.loc[test_df['SaleType'].isna() == True, 'SaleType'] = 'WD'



train_df.drop(columns = ['YearBuiltP', 'MasVnrAreaP', '1stFlrSFP', 'WoodDeckSFP', 'OpenPorchSFP', 'LowQualFinSFP'], inplace = True)
test_df.drop(columns = ['YearBuiltP', 'MasVnrAreaP', '1stFlrSFP', 'WoodDeckSFP', 'OpenPorchSFP', 'LowQualFinSFP'], inplace = True)


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
