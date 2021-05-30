'''
Created on May 30, 2021

@author: zollen
@url: https://towardsdatascience.com/time-based-cross-validation-d259b13d42b8
'''

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sb
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')  
    
def generate_seasonal_data(periodicity, total_cycles, noise_std=1.,
                           harmonics=None):
    '''
    Consider the problem of modeling time series data with multiple seasonal components 
    with different periodicities. Let us take the time series y(t) and decompose it 
    explicitly to have a level component and two seasonal components.
    
    y(t)=μ(t) + γ1(t) + γ2(t)
    μ - lambda
    γ - gamma
    
    where μ(t) represents the trend or level, γ1(t) represents a seasonal component 
    with a relatively short period, and γ2(t) represents another seasonal component 
    of longer period. We will have a fixed intercept term for our level and consider 
    both γ2(t) and γ2(t) to be stochastic so that the seasonal patterns can vary 
    over time.
    
    We will simulate 300 periods and two seasonal terms parametrized in the frequency 
    domain having periods 10 and 100, respectively, and 3 and 2 number of harmonics, 
    respectively. Further, the variances of their stochastic parts are 4 and 9, 
    respectively.
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


duration = 100 * 3
periodicities = [10, 100]
num_harmonics = [3, 2]
std = np.array([2, 3])
np.random.seed(8678309)

terms = []
for ix, _ in enumerate(periodicities):
    s = generate_seasonal_data(
            periodicities[ix],
            duration / periodicities[ix],
            harmonics=num_harmonics[ix],
            noise_std=std[ix])
    terms.append(s)
    

terms.append(np.ones_like(terms[0]) * 10.)
series = pd.Series(np.sum(terms, axis=0))

start_date = datetime(2020, 1, 1)
end_date = start_date + timedelta(days=len(series) - 1)

date_rng = pd.date_range(start=start_date, end=end_date, freq='D')
print("Start Date: ", start_date, " END Date: ", end_date)

df = pd.DataFrame(data={
                'Date': date_rng,
                'Price': series})

df.set_index('Date', inplace=True)
df = df.asfreq(pd.infer_freq(df.index))


if True:
    plt.figure(figsize=(10,4))
    plt.plot(df)
    plt.title('Generated TimeSeries Data', fontsize=20)
    plt.ylabel('Prices', fontsize=16)
    plt.show()
    
tscv = TimeSeriesSplit(n_splits=10, max_train_size=120)
print(tscv)
for train_index, test_index in tscv.split(df):
    X_train, X_test = df.iloc[train_index], df.iloc[test_index]
    print(len(X_train), len(X_test))
    print("TRAIN: ", X_train.index[0], " ==> ", X_train.index[-1],
          "TEST: ", X_test.index[0], " ===> ", X_test.index[-1])

    
