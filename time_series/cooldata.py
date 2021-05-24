'''
Created on May 22, 2021

@author: zollen
'''

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

def adfuller_test(series, signif=0.05, reg='c', name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC', regression=reg)
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    print("P value of reg(%s): %0.4f" % (reg, p_value))
    
    
def simulate_seasonal_term(periodicity, total_cycles, noise_std=1.,
                           harmonics=None):
    duration = periodicity * total_cycles
    assert duration == int(duration)
    duration = int(duration)
    harmonics = harmonics if harmonics else int(np.floor(periodicity / 2))

    lambda_p = 2 * np.pi / float(periodicity)

    gamma_jt = noise_std * np.random.randn((harmonics))
    gamma_star_jt = noise_std * np.random.randn((harmonics))

    total_timesteps = 100 * duration # Pad for burn in
    series = np.zeros(total_timesteps)
    for t in range(total_timesteps):
        gamma_jtp1 = np.zeros_like(gamma_jt)
        gamma_star_jtp1 = np.zeros_like(gamma_star_jt)
        for j in range(1, harmonics + 1):
            cos_j = np.cos(lambda_p * j)
            sin_j = np.sin(lambda_p * j)
            gamma_jtp1[j - 1] = (gamma_jt[j - 1] * cos_j
                                 + gamma_star_jt[j - 1] * sin_j
                                 + noise_std * np.random.randn())
            gamma_star_jtp1[j - 1] = (- gamma_jt[j - 1] * sin_j
                                      + gamma_star_jt[j - 1] * cos_j
                                      + noise_std * np.random.randn())
        series[t] = np.sum(gamma_jtp1)
        gamma_jt = gamma_jtp1
        gamma_star_jt = gamma_star_jtp1
    wanted_series = series[-duration:] # Discard burn in

    return wanted_series

def myfunc(df):
    return 1 + np.sin(df.astype('int64') // 1e9 * (4 * np.pi / year))

def generate_data(start, end) -> pd.DataFrame:
    """ Create a time series x sin wave dataframe. """
    df = pd.DataFrame(columns=['Date', 'Price'])
    df['Date'] = pd.date_range(start=start, end=end, freq='D')
    df['Price'] = myfunc(df['Date'])
    df['Price'] = (df['Price'] * 100).round(2)
    df['Price'] = df['Price'] + np.arange(len(df['Date'])) + np.random.standard_normal(len(df['Date']))
    df['Price'] = df['Price'].astype('float64')
    df['Date'] = df['Date'].apply(lambda d: d.strftime('%Y-%m-%d'))
    df = df.set_index('Date')
    df = df.asfreq(pd.infer_freq(df.index))

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