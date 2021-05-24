'''
Created on May 22, 2021

@author: zollen
'''

from datetime import datetime, timedelta
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
    '''
    Consider the problem of modeling time series data with multiple seasonal components 
    with different periodicities. Let us take the time series y(t) and decompose it 
    explicitly to have a level component and two seasonal components.
    
    y(t)=μ(t) + γ1(t) + γ2(t)
    μ - lambda
    γ - gamma
    '''
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

if True:
    duration = 100 * 3
    periodicities = [10, 100]
    num_harmonics = [3, 2]
    std = np.array([2, 3])
    np.random.seed(8678309)
    
    terms = []
    for ix, _ in enumerate(periodicities):
        s = simulate_seasonal_term(
            periodicities[ix],
            duration / periodicities[ix],
            harmonics=num_harmonics[ix],
            noise_std=std[ix])
        terms.append(s)
    terms.append(np.ones_like(terms[0]) * 10.)
    series = pd.Series(np.sum(terms, axis=0))
    df = pd.DataFrame(data={'Price': series,
                            '10(3)': terms[0],
                            '100(2)': terms[1],
                            'level':terms[2]})

    if True:    
        h1, = plt.plot(df['Price'])
        h2, = plt.plot(df['10(3)'])
        h3, = plt.plot(df['100(2)'])
        h4, = plt.plot(df['level'])
        plt.legend(['total','10(3)','100(2)', 'level'])

    start_date = datetime.now()
    end_date = start_date + timedelta(days=len(series) - 1)
    train_df = pd.DataFrame({
                    'Date': pd.date_range(start=start_date, end=end_date, freq='D'),
                    'Price': series
                })
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    train_df = train_df.set_index('Date')
    train_df = train_df.asfreq(pd.infer_freq(train_df.index), method="pad")
else:
    train_df = generate_data('2018-01-01', '2021-01-01')

adfuller_test(train_df)

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