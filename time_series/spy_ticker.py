'''
Created on Apr. 7, 2021

@author: zollen
'''

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
import seaborn as sb

sb.set_style('whitegrid')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

tickerSymbol = "SPY"

tickerData = yf.Ticker(tickerSymbol)

ticketDf = tickerData.history(period='1d', start = '2015-1-1', end='2020-1-1')

ticketDf = ticketDf[['Close']]

print(ticketDf.head())

if False:
    # This graph shows that the time series is not stationary.
    plt.figure(figsize=(10,4))
    plt.plot(ticketDf.Close)
    plt.title("Stock Price over Time (%s)" % tickerSymbol, fontsize = 20)
    plt.ylabel('Price', fontsize = 16)
    for year in range(2015, 2021):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--')
        
        
        
first_diffs = ticketDf.Close.values[1:] - ticketDf.Close.values[:-1]
first_diffs = np.concatenate([first_diffs, [0]])

ticketDf['FirstDifference'] = first_diffs        

if True:
    # This first difference graph solves the stationary issue.
    plt.figure(figsize=(10,4))
    plt.plot(ticketDf.FirstDifference)
    plt.title("First Difference over Time (%s)" % tickerSymbol, fontsize = 20)
    plt.ylabel('Price', fontsize = 16)
    for year in range(2015, 2021):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--')
        
if False:
    # ACF isn't that informative
    acf_plot = plot_acf(ticketDf.FirstDifference)

if False:
    # PACF also doesn't tell us much either
    pacf_plot = plot_pacf(ticketDf.FirstDifference)
    
## Lesson, prediction stock prices is not easy as ACF and PACF reveval no useful information
    
plt.show()