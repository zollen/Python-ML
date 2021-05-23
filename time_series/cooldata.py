'''
Created on May 22, 2021

@author: zollen
'''

import datetime
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

day = 24 * 60 * 60
year = 365.2425 * day

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC', regression="c")
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.") 

def generate_data(start, end) -> pd.DataFrame:
    """ Create a time series x sin wave dataframe. """
    df = pd.DataFrame(columns=['Date', 'Price'])
    df['Date'] = pd.date_range(start=start, end=end, freq='D')
    df['Price'] = 1 + np.sin(df['Date'].astype('int64') // 1e9 * (4 * np.pi / year))
    df['Price'] = (df['Price'] * 100).round(2)
    df['Price'] = df['Price'] + [ d for d in range(len(df['Date']))]
    df['Price'] = df['Price'].astype('float64')
    df['Date'] = df['Date'].apply(lambda d: d.strftime('%Y-%m-%d'))
    df = df.set_index('Date')
   
    return df


train_df = generate_data('2018-01-01', '2021-01-01')

adfuller_test(train_df)

train_df.plot(figsize=(12, 8))
plt.show()


stl = STL(train_df)
result = stl.fit()

seasonal, trend, resid = result.seasonal, result.trend, result.resid

plt.figure(figsize=(8,6))

plt.subplot(4,1,1)
plt.plot(train_df)
plt.title('Original Series', fontsize=16)

plt.subplot(4,1,2)
plt.plot(trend)
plt.title('Trend', fontsize=16)

plt.subplot(4,1,3)
plt.plot(seasonal)
plt.title('Seasonal', fontsize=16)

plt.subplot(4,1,4)
plt.plot(resid)
plt.title('Residual', fontsize=16)

plt.tight_layout()

plt.show()