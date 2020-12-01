'''
Created on Nov. 30, 2020

@author: zollen
'''

import pandas as pd
import numpy as np
import math
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p


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
