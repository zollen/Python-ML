'''
Created on Apr. 7, 2021

@author: zollen
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
import seaborn as sb

sb.set_style('whitegrid')

df_ice_cream = pd.read_csv('ice_cream.csv')

df_ice_cream.rename(columns={'DATE': 'date', 'IPN31152N':'production'}, inplace = True)

df_ice_cream['date'] = pd.to_datetime(df_ice_cream.date)
df_ice_cream.set_index('date', inplace=True)

start_date = pd.to_datetime('2010-01-01')
df_ice_cream = df_ice_cream[start_date:]

if False:
    plt.figure(figsize=(10,4))
    plt.plot(df_ice_cream.production)
    plt.title("Ice Cream Production Over Time", fontsize = 20)
    plt.ylabel('Production', fontsize = 16)
    for year in range(2011, 2021):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--')

if False:
    # Based on "decaying pattern over time" ACF, we are likely dealing with an Auto Regressive process
    acf_plot = plot_acf(df_ice_cream.production, lags=100)
    
if True:
    # A PACF pattern would expect few standout PACFs, and the rest of close to zero
    # Based on PACF, we should start with an Auto Regressive models with lags 1, 2, 3, 4, 8, 11, 14
    pacf_plot = plot_pacf(df_ice_cream.production)
    
plt.show()