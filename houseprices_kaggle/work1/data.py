'''
Created on Oct. 31, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
import math
import pprint
import warnings
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p



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


#train_df.drop(columns = ['Utilities', 'Street'], inplace = True)
#test_df.drop(columns = ['Utilities', 'Street'], inplace = True)

last = 0
for val in range(0, 222000, 1000):
    train_df.loc[(train_df['LotArea'] >= last) & (train_df['LotArea'] < val), 'LotAreaP'] = val
    test_df.loc[(test_df['LotArea'] >= last) & (test_df['LotArea'] < val), 'LotAreaP'] = val
    last = val
    
all_df = pd.concat([ train_df, test_df ]) 


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

test_df.loc[(test_df['BsmtCond'].isna() == True) &
            (test_df['TotalBsmtSF'] == 0), 'BsmtCond'] = 'None'
test_df.loc[(test_df['BsmtQual'].isna() == True) &
            (test_df['TotalBsmtSF'] == 0), 'BsmtQual'] = 'None'
test_df.loc[(test_df['BsmtFinType1'].isna() == True) &
            (test_df['TotalBsmtSF'] == 0), 'BsmtFinType1'] = 'None'
test_df.loc[(test_df['BsmtFinType2'].isna() == True) &
            (test_df['TotalBsmtSF'] == 0), 'BsmtFinType2'] = 'None'
test_df.loc[(test_df['BsmtExposure'].isna() == True) &
            (test_df['TotalBsmtSF'] == 0), 'BsmtExposure'] = 'None'
               

             
# OverallQual(8), OverallCond(9), BsmtQual(Gd), TotalBsmtSF(1426), BsmtExposure(Mn), BsmtCond(Nan)
test_df.loc[test_df['Id'] == 2041, 'BsmtCond'] = 'TA'

# OverallQual(6), OverallCond(6), BsmtQual(TA), TotalBsmtSF(1127), BsmtExposure(No), BsmtCond(Nan)
test_df.loc[test_df['Id'] == 2186, 'BsmtCond'] = 'TA'

# OverallQual(5), OverallCond(7), BsmtQual(TA), TotalBsmtSF(995), BsmtExposure(Av), BsmtQual(NaN)
test_df.loc[test_df['Id'] == 2525, 'BsmtCond'] = 'TA'

# OverallQual(4), OverallCond(7), BsmtCond(Fa), TotalBsmtSF(173), BsmtExposure(No), BsmtQual(NaN)
test_df.loc[test_df['Id'] == 2218, 'BsmtQual'] = 'TA'

# OverallQual(4), OverallCond(7), BsmtCond(TA), TotalBsmtSF(356), BsmtExposure(No), BsmtQual(NaN)
test_df.loc[test_df['Id'] == 2219, 'BsmtQual'] = 'TA'

# OverallQual(8), OverallCond(5), TotalBsmtSF(3206), BsmtQual(Gd), BsmtCond(TA), BsmtExposure(No), BsmtFinType2(NaN)
train_df.loc[train_df['Id'] == 333, 'BsmtFinType2'] = 'BLQ'

# OverallQual(7), OverallCond(5), TotalBsmtSF(936), BsmtQual(Gd)  BsmtCond(TA) BsmtExposure(NaN)
train_df.loc[train_df['Id'] == 949, 'BsmtExposure'] = 'No'

# OverallQual(8), OverallCond(5), TotalBsmtSF(1595), BsmtQual(Gd)  BsmtCond(TA) BsmtExposure(NaN)
test_df.loc[test_df['Id'] == 1488, 'BsmtExposure'] = 'Av'

# OverallQual(5), OverallCond(5), TotalBsmtSF(725), BsmtQual(Gd)  BsmtCond(TA) BsmtExposure(NaN)
test_df.loc[test_df['Id'] == 2349, 'BsmtExposure'] = 'No'

## 2121 may have no basement
test_df.loc[test_df['Id'] == 2121, 'BsmtExposure'] = 'None'
test_df.loc[test_df['Id'] == 2121, 'BsmtQual'] = 'None'
test_df.loc[test_df['Id'] == 2121, 'BsmtCond'] = 'None'
test_df.loc[test_df['Id'] == 2121, 'BsmtFinType1'] = 'None'
test_df.loc[test_df['Id'] == 2121, 'BsmtFinType2'] = 'None'
test_df.loc[test_df['Id'] == 2121, 'BsmtFinSF1'] = 0
test_df.loc[test_df['Id'] == 2121, 'BsmtFinSF2'] = 0
test_df.loc[test_df['Id'] == 2121, 'BsmtUnfSF'] = 0
test_df.loc[test_df['Id'] == 2121, 'TotalBsmtSF'] = 0
test_df.loc[test_df['Id'] == 2121, 'BsmtFullBath'] = 0
test_df.loc[test_df['Id'] == 2121, 'BsmtHalfBath'] = 0

test_df.loc[test_df['Id'] == 2189, 'BsmtExposure'] = 'None'
test_df.loc[test_df['Id'] == 2189, 'BsmtQual'] = 'None'
test_df.loc[test_df['Id'] == 2189, 'BsmtCond'] = 'None'
test_df.loc[test_df['Id'] == 2189, 'BsmtFinType1'] = 'None'
test_df.loc[test_df['Id'] == 2189, 'BsmtFinType2'] = 'None'
test_df.loc[test_df['Id'] == 2189, 'BsmtFinSF1'] = 0
test_df.loc[test_df['Id'] == 2189, 'BsmtFinSF2'] = 0
test_df.loc[test_df['Id'] == 2189, 'BsmtUnfSF'] = 0
test_df.loc[test_df['Id'] == 2189, 'TotalBsmtSF'] = 0
test_df.loc[test_df['Id'] == 2189, 'BsmtFullBath'] = 0
test_df.loc[test_df['Id'] == 2189, 'BsmtHalfBath'] = 0

'''
Fill GarageFinish, GarageType, GarageQual, GarageCond
'''
test_df.loc[test_df['Id'] == 2127, 'GarageCond'] = 'TA'
test_df.loc[test_df['Id'] == 2127, 'GarageQual'] = 'TA'

# MSSubClass(70) MSZoning(RM) LotArea(9060) YearBuilt(1923) GarageType(Detchd)
test_df.loc[test_df['Id'] == 2577, 'GarageQual'] = 'TA'
test_df.loc[test_df['Id'] == 2577, 'GarageCond'] = 'TA'
test_df.loc[test_df['Id'] == 2577, 'GarageArea'] = 568
test_df.loc[test_df['Id'] == 2577, 'GarageYrBlt'] = 1967
test_df.loc[test_df['Id'] == 2577, 'GarageFinish'] = 'Unf'
test_df.loc[test_df['Id'] == 2577, 'GarageCars'] = 1

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

test_df.loc[(test_df['GarageFinish'].isna() == True) & 
             (test_df['GarageArea'] == 0), 'GarageFinish'] = 'None'
test_df.loc[(test_df['GarageType'].isna() == True) & 
             (test_df['GarageArea'] == 0), 'GarageType'] = 'None'
test_df.loc[(test_df['GarageQual'].isna() == True) & 
             (test_df['GarageArea'] == 0), 'GarageQual'] = 'None'    
test_df.loc[(test_df['GarageCond'].isna() == True) & 
             (test_df['GarageArea'] == 0), 'GarageCond'] = 'None' 
test_df.loc[(test_df['GarageYrBlt'].isna() == True) & 
             (test_df['GarageArea'] == 0), 'GarageYrBlt'] = 0
             




'''
Fill GarageFinish
'''
test_df.loc[test_df['Id'] == 2127, 'GarageFinish'] = 'Unf'
test_df.loc[test_df['Id'] == 2127, 'GarageYrBlt'] = 1925
test_df.loc[test_df['Id'] == 2577, 'GarageFinish'] = 'Unf'
test_df.loc[test_df['Id'] == 2577, 'GarageYrBlt'] = 1950



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
test_df.loc[test_df['Id'] == 1692, 'MasVnrType'] = 'BrkFace'
test_df.loc[test_df['Id'] == 1707, 'MasVnrType'] = 'Stone'
test_df.loc[test_df['Id'] == 1883, 'MasVnrType'] = 'BrkFace'
test_df.loc[test_df['Id'] == 1993, 'MasVnrType'] = 'BrkFace'
test_df.loc[test_df['Id'] == 2005, 'MasVnrType'] = 'BrkFace'
test_df.loc[test_df['Id'] == 2042, 'MasVnrType'] = 'BrkFace'
test_df.loc[test_df['Id'] == 2312, 'MasVnrType'] = 'Stone'
test_df.loc[test_df['Id'] == 2326, 'MasVnrType'] = 'BrkFace'
test_df.loc[test_df['Id'] == 2341, 'MasVnrType'] = 'Stone'
test_df.loc[test_df['Id'] == 2350, 'MasVnrType'] = 'Stone'
test_df.loc[test_df['Id'] == 2369, 'MasVnrType'] = 'Stone'
test_df.loc[test_df['Id'] == 2593, 'MasVnrType'] = 'BrkFace'
test_df.loc[test_df['Id'] == 2611, 'MasVnrType'] = 'BrkFace'
test_df.loc[test_df['Id'] == 2658, 'MasVnrType'] = 'Stone'
test_df.loc[test_df['Id'] == 2687, 'MasVnrType'] = 'Stone'
test_df.loc[test_df['Id'] == 2863, 'MasVnrType'] = 'BrkFace'

             
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
test_df.loc[test_df['Id'] == 1692, 'MasVnrArea'] = 107
test_df.loc[test_df['Id'] == 1707, 'MasVnrArea'] = 151
test_df.loc[test_df['Id'] == 1883, 'MasVnrArea'] = 142
test_df.loc[test_df['Id'] == 1993, 'MasVnrArea'] = 128
test_df.loc[test_df['Id'] == 2005, 'MasVnrArea'] = 184
test_df.loc[test_df['Id'] == 2042, 'MasVnrArea'] = 212
test_df.loc[test_df['Id'] == 2312, 'MasVnrArea'] = 66
test_df.loc[test_df['Id'] == 2326, 'MasVnrArea'] = 87
test_df.loc[test_df['Id'] == 2341, 'MasVnrArea'] = 250
test_df.loc[test_df['Id'] == 2350, 'MasVnrArea'] = 201
test_df.loc[test_df['Id'] == 2369, 'MasVnrArea'] = 166
test_df.loc[test_df['Id'] == 2593, 'MasVnrArea'] = 90
test_df.loc[test_df['Id'] == 2658, 'MasVnrArea'] = 402
test_df.loc[test_df['Id'] == 2687, 'MasVnrArea'] = 298
test_df.loc[test_df['Id'] == 2863, 'MasVnrArea'] = 99
       
       
'''
Fill LotFrontage
'''          
def fillLotFrontage(df1, df2):
    
    grps = all_df.groupby(['OverallQual', 'LotAreaP', 
                           'LotConfig'])['LotFrontage'].median()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['LotFrontage'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['LotAreaP'] == index[1]) &
                 (df['LotConfig'] == index[2]), 'LotFrontage'] = grps[index[0], index[1], index[2]]
      
    grps = all_df.groupby(['OverallQual', 'LotAreaP'])['LotFrontage'].median()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['LotFrontage'].isna() == True) &
                 (df['OverallQual'] == index[0]) &
                 (df['LotAreaP'] == index[1]), 'LotFrontage'] = grps[index[0], index[1]]
      
    grps = all_df.groupby(['OverallQual'])['LotFrontage'].median()
    for index, _ in grps.items():
        for df in  [ df1, df2 ]:
            df.loc[(df['LotFrontage'].isna() == True) &
                 (df['OverallQual'] == index), 'LotFrontage'] = grps[index]
                 
    
                 
fillLotFrontage(train_df, test_df)  

       
       


'''
Fill FireplaceQu
'''
train_df.loc[train_df['Fireplaces'] == 0, 'FireplaceQu'] = 'None'
test_df.loc[test_df['Fireplaces'] == 0, 'FireplaceQu'] = 'None'




'''
Fill MSZoning   
'''
test_df.loc[test_df['Id'] == 1916, 'MSZoning'] = 'C (all)'
test_df.loc[test_df['Id'] == 2217, 'MSZoning'] = 'C (all)'
test_df.loc[test_df['Id'] == 2251, 'MSZoning'] = 'RM'
test_df.loc[test_df['Id'] == 2905, 'MSZoning'] = 'RL'



'''
Fill Utilities
'''
train_df.loc[train_df['Utilities'].isna() == True, 'Utilities'] = 'AllPub'
test_df.loc[test_df['Utilities'].isna() == True, 'Utilities'] = 'AllPub'

    
    

'''
Fill Exterior1st & Exterior2nd
'''
test_df.loc[(test_df['Exterior1st'].isna() == True), 'Exterior1st'] = 'Wd Sdng'
test_df.loc[(test_df['Exterior2nd'].isna() == True), 'Exterior2nd'] = 'Plywood'



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



'''
Add TotalSF
'''
train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']




'''
Add Potiental features
'''
#train_df['YrBltAndRemod'] = train_df['YearBuilt']+train_df['YearRemodAdd']
#test_df['YrBltAndRemod'] = test_df['YearBuilt']+test_df['YearRemodAdd']
#train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
#test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']
#train_df['TotalSqrfootage'] = (train_df['BsmtFinSF1'] + train_df['BsmtFinSF2'] + train_df['1stFlrSF'] + train_df['2ndFlrSF'])
#test_df['TotalSqrfootage'] = (test_df['BsmtFinSF1'] + test_df['BsmtFinSF2'] + test_df['1stFlrSF'] + train_df['2ndFlrSF'])
#train_df['TotalBathrooms'] = (train_df['FullBath'] + (0.5 * train_df['HalfBath']) + train_df['BsmtFullBath'] + (0.5 * train_df['BsmtHalfBath']))
#test_df['TotalBathrooms'] = (test_df['FullBath'] + (0.5 * test_df['HalfBath']) + test_df['BsmtFullBath'] + (0.5 * test_df['BsmtHalfBath']))
#train_df['TotalPorchSF'] = (train_df['OpenPorchSF'] + train_df['3SsnPorch'] + train_df['EnclosedPorch'] + train_df['ScreenPorch'] + train_df['WoodDeckSF'])
#test_df['TotalPorchSF'] = (test_df['OpenPorchSF'] + test_df['3SsnPorch'] + test_df['EnclosedPorch'] + test_df['ScreenPorch'] + test_df['WoodDeckSF'])

#train_df['TotalHomeQuality'] = train_df['OverallQual'] + train_df['OverallCond']
#test_df['TotalHomeQuality'] = test_df['OverallQual'] + test_df['OverallCond']

#train_df['HasPool'] = train_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
#test_df['HasPool'] = test_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
#train_df['Has2ndFlr'] = train_df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
#test_df['Has2ndFlr'] = test_df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
#train_df['HasGarage'] = train_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
#test_df['HasGarage'] = test_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
#train_df['HasBsmt'] = train_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
#test_df['HasBsmt'] = test_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
#train_df['HasFireplace'] = train_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
#test_df['HasFireplace'] = test_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

#train_df['OtherRoom'] = train_df["TotRmsAbvGrd"] - train_df['KitchenAbvGr'] - train_df['Bedroom']
#test_df['OtherRoom'] = test_df["TotRmsAbvGrd"] - test_df['KitchenAbvGr'] - test_df['Bedroom']


#train_df["SqFtPerRoom"] = train_df["GrLivArea"] / (
#    train_df["TotRmsAbvGrd"]
#    + train_df["FullBath"]
#    + train_df["HalfBath"]
#    + train_df["KitchenAbvGr"]
#)

#test_df["SqFtPerRoom"] = test_df["GrLivArea"] / (
#    test_df["TotRmsAbvGrd"]
#    + test_df["FullBath"]
#    + test_df["HalfBath"]
#    + test_df["KitchenAbvGr"]
#)

#train_df["BuiltAge"] = train_df["YrSold"] - train_df["YearBuilt"]
#train_df["RemodAge"] = train_df["YrSold"] - train_df["YearRemodAdd"]
#train_df["Remodeled"] = train_df["YearBuilt"] != train_df["YearRemodAdd"]
#train_df["BuiltAge"] = train_df["BuiltAge"].apply(lambda x: 0 if x < 0 else x)
#train_df["RemodAge"] = train_df["RemodAge"].apply(lambda x: 0 if x < 0 else x)

#test_df["BuiltAge"] = test_df["YrSold"] - test_df["YearBuilt"]
#test_df["RemodAge"] = test_df["YrSold"] - test_df["YearRemodAdd"]
#test_df["Remodeled"] = test_df["YearBuilt"] != test_df["YearRemodAdd"]
#test_df["BuiltAge"] = test_df["BuiltAge"].apply(lambda x: 0 if x < 0 else x)
#test_df["RemodAge"] = test_df["RemodAge"].apply(lambda x: 0 if x < 0 else x)




'''
Merge YrSold and MoSold
RMSE   : 7639.0560
CV RMSE: 20208.7261
Site   : 0.12014
'''
def mergeSold(rec):
    yrSold = rec['YrSold']
    moSold = rec['MoSold']
    
    years = {2006: 0, 2007: 1, 2008: 2, 2009: 3, 2010: 4}
    
    return round(years[yrSold] + (moSold / 12), 2)
   
    
train_df['YrSold'] = train_df.apply(mergeSold, axis = 1)
test_df['YrSold'] = test_df.apply(mergeSold, axis = 1)

train_df.drop(columns = ['MoSold'], inplace = True)
test_df.drop(columns = ['MoSold'], inplace = True)


'''
Remove support features
'''
train_df.drop(columns = ['LotAreaP'], inplace = True)
test_df.drop(columns = ['LotAreaP'], inplace = True)



'''
DeSkew numerical features
'''
col_types = train_df.columns.to_series().groupby(train_df.dtypes)
numeric_columns = []
       
for col in col_types:
    if col[0] != 'object':
        numeric_columns += col[1].unique().tolist()

numeric_columns.remove('Id')
numeric_columns.remove('SalePrice')


   
for name in numeric_columns:
    col_df = pd.DataFrame()
    
    col_df['NORM'] = train_df[name].values
    col_df['LOG1P'] = train_df[name].apply(lambda x : np.log1p(x)).values
    cb_lambda = boxcox_normmax(train_df[name] + 1)
    col_df['COXBOX'] = boxcox1p(train_df[name], cb_lambda).values
    
    nums = []
    
    nums.append(np.abs(skew(col_df['NORM'])))
    nums.append(np.abs(skew(col_df['LOG1P'])))
    nums.append(np.abs(skew(col_df['COXBOX'])))
    
    nums  = [999 if math.isnan(x) else x for x in nums]
        
    
    smallest = nums.index(min(nums))
    if smallest == 1:
        train_df[name] = col_df['LOG1P']
        test_df[name] = test_df[name].apply(lambda x : np.log1p(x)).values
    elif smallest == 2:
        train_df[name] = col_df['COXBOX']
        test_df[name] = boxcox1p(test_df[name], cb_lambda).values
            
  
train_df['SalePrice'] = train_df['SalePrice'].apply(lambda x : np.log1p(x))  
  

train_df.to_csv(os.path.join(PROJECT_DIR, 'data/train_data.csv'), index = False)
test_df.to_csv(os.path.join(PROJECT_DIR, 'data/test_data.csv'), index = False)

print("DONE")