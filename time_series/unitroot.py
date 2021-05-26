'''
Created on May 26, 2021

@author: zollen
'''
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')


def generate_ar_process(lags, coefs, length):
    
    '''
    lags = 1
    AR(t): y(t) = const + γ1(t-1) + e(t) <-- adfuller(reg='c')
    γ - coefs
    '''
    
    #cast coefs to np array
    coefs = np.array(coefs)
    
    #initial values
    series = [np.random.normal() for _ in range(lags)]

    
    for _ in range(length):
        #get previous values of the series, reversed
        prev_vals = series[-lags:][::-1]
        
        #get new value of time series
        new_val = np.sum(np.array(prev_vals) * coefs) + np.random.normal()
        
        series.append(new_val)
        
    return np.array(series)


def perform_adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    


data = generate_ar_process(1, [0.5], 100)
plt.figure(figsize=(10,4))
plt.plot(data)
plt.title('Stationary AR(1) Process', fontsize=18)

perform_adf_test(data)


data = generate_ar_process(1, [1], 100)
plt.figure(figsize=(10,4))
plt.plot(data)
plt.title('Non-stationary AR(1) Process', fontsize=18)

perform_adf_test(data)


data = generate_ar_process(2, [0.5, 0.3], 100)
plt.figure(figsize=(10,4))
plt.plot(data)
plt.title('Stationary AR(2) Process', fontsize=18)

perform_adf_test(data)


data = generate_ar_process(2, [0.7, 0.3], 100)  # coefs sum up to 1
plt.figure(figsize=(10,4))
plt.plot(data)
plt.title('Non-stationary AR(2) Process', fontsize=18)

perform_adf_test(data)



plt.show()