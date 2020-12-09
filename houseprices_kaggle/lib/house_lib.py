'''
Created on Nov. 30, 2020

@author: zollen
'''

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p

def rmse_cv(model, data, label, n_folds):
    kf = KFold(n_folds, shuffle=True, random_state=87).get_n_splits(data.values)
    rmse = np.sqrt(-1 * cross_val_score(model, 
                                  data.values, label, scoring="neg_mean_squared_error", cv = kf))
    return np.mean(rmse)


def feature_engineering1(df1, df2):
    
    df1['OverallQual'] = df1['OverallQual'] * df1['OverallCond']
    df2['OverallQual'] = df2['OverallQual'] * df2['OverallCond']

    df1.drop(columns = ['OverallCond'], inplace = True)
    df2.drop(columns = ['OverallCond'], inplace = True)

    def mergeSold(rec):
        yrSold = rec['YrSold']
        moSold = rec['MoSold']
    
        years = {2006: 0, 2007: 1, 2008: 2, 2009: 3, 2010: 4}
    
        return round(years[yrSold] + (moSold / 12), 2)
   
    
    df1['YrSold'] = df1.apply(mergeSold, axis = 1)
    df2['YrSold'] = df2.apply(mergeSold, axis = 1)

    df1.drop(columns = ['MoSold'], inplace = True)
    df2.drop(columns = ['MoSold'], inplace = True)

    df1['TotalSF'] = df1['TotalBsmtSF'] + df1['1stFlrSF'] + df1['2ndFlrSF'] + df1['OpenPorchSF']
    df2['TotalSF'] = df2['TotalBsmtSF'] + df2['1stFlrSF'] + df2['2ndFlrSF'] + df2['OpenPorchSF']

    df1['ExterQual'] = df1['ExterQual'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    df2['ExterQual'] = df2['ExterQual'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    df1['ExterCond'] = df1['ExterCond'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    df2['ExterCond'] = df2['ExterCond'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    df1['ExterQual'] = df1['ExterQual'] * df1['ExterCond']
    df2['ExterQual'] = df2['ExterQual'] * df2['ExterCond']

    df1.drop(columns = ['ExterCond'], inplace = True)
    df2.drop(columns = ['ExterCond'], inplace = True)

def feature_engineering2(df1, df2):

    df1["BuiltAge"] = df1["YrSold"] - df1["YearBuilt"]
    df1["RemodAge"] = df1["YrSold"] - df1["YearRemodAdd"]
    df1["Remodeled"] = df1["YearBuilt"] != df1["YearRemodAdd"]
    df1["BuiltAge"] = df1["BuiltAge"].apply(lambda x: 0 if x < 0 else x)
    df1["RemodAge"] = df1["RemodAge"].apply(lambda x: 0 if x < 0 else x)

    df2["BuiltAge"] = df2["YrSold"] - df2["YearBuilt"]
    df2["RemodAge"] = df2["YrSold"] - df2["YearRemodAdd"]
    df2["Remodeled"] = df2["YearBuilt"] != df2["YearRemodAdd"]
    df2["BuiltAge"] = df2["BuiltAge"].apply(lambda x: 0 if x < 0 else x)
    df2["RemodAge"] = df2["RemodAge"].apply(lambda x: 0 if x < 0 else x)

    df1['TotalSF'] = df1['TotalBsmtSF'] + df1['1stFlrSF'] + df1['2ndFlrSF']
    df2['TotalSF'] = df2['TotalBsmtSF'] + df2['1stFlrSF'] + df2['2ndFlrSF']

    df1["SqFtPerRoom"] = df1["GrLivArea"] / (
                                df1["TotRmsAbvGrd"]
                                + df1["FullBath"]
                                + df1["HalfBath"]
                                + df1["KitchenAbvGr"]
                                )

    df2["SqFtPerRoom"] = df2["GrLivArea"] / (
                            df2["TotRmsAbvGrd"]
                            + df2["FullBath"]
                            + df2["HalfBath"]
                            + df2["KitchenAbvGr"]
                            )

    df1['HasPool'] = df1['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df2['HasPool'] = df2['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df1['Has2ndFlr'] = df1['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    df2['Has2ndFlr'] = df2['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    df1['HasGarage'] = df1['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    df2['HasGarage'] = df2['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    df1['HasBsmt'] = df1['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    df2['HasBsmt'] = df2['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    df1['HasFireplace'] = df1['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    df2['HasFireplace'] = df2['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    df1['OtherRoom'] = df1["TotRmsAbvGrd"] - df1['KitchenAbvGr'] - df1['BedroomAbvGr']
    df2['OtherRoom'] = df2["TotRmsAbvGrd"] - df2['KitchenAbvGr'] - df2['BedroomAbvGr']

def deSkew(df1, df2, numeric_columns):
    for name in numeric_columns:
        col_df = pd.DataFrame()
    
        col_df['NORM'] = df1[name].values
        col_df['LOG1P'] = df1[name].apply(lambda x : np.log1p(x)).values
        cb_lambda = boxcox_normmax(df1[name] + 1)
        col_df['COXBOX'] = boxcox1p(df1[name], cb_lambda).values
    
        nums = []
    
        nums.append(np.abs(skew(col_df['NORM'])))
        nums.append(np.abs(skew(col_df['LOG1P'])))
        nums.append(np.abs(skew(col_df['COXBOX'])))
    
        nums  = [999 if math.isnan(x) else x for x in nums]
        
    
        smallest = nums.index(min(nums))
        if smallest == 1:
            df1[name] = col_df['LOG1P']
            df2[name] = df2[name].apply(lambda x : np.log1p(x)).values
        elif smallest == 2:
            df1[name] = col_df['COXBOX']
            df2[name] = boxcox1p(df2[name], cb_lambda).values
            
    if 'SalePrice' in df1:
            df1['SalePrice'] = df1['SalePrice'].apply(lambda x : np.log1p(x))  
            
    if 'SalePrice' in df2:
            df2['SalePrice'] = df2['SalePrice'].apply(lambda x : np.log1p(x))  
            
def write_result(name, df1, df2):
    
    df1['SalePrice'] = df1['Prediction']
    all_df = pd.concat([df1[['Id', 'SalePrice']], df2[['Id', 'SalePrice']]], ignore_index=True)
    
    all_df.to_csv(name, index = False)

def auto_encode(df):
    
    col_types = df.columns.to_series().groupby(df.dtypes)
    categorical_columns = []
    
    for col in col_types:
        if col[0] == 'object':
            categorical_columns = col[1].unique().tolist()

    for name in categorical_columns:
        res_df = pd.DataFrame()
        keys = []
        vals = []
        grps = df.groupby([name])['SalePrice'].median()
        for index, _ in grps.items():
            keys.append(index)
            vals.append(grps[index])
        
        res_df['Key'] = keys
        res_df['Val'] = vals
        
        res_df.sort_values('Val', ascending = True, inplace = True)
                
        manifest = {}
        labels = res_df['Key'].tolist()
        for index, label in zip(range(0, len(labels)), labels):
            manifest[label] = index
            
        df[name] = df[name].map(manifest)
    
    