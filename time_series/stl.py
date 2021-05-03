'''
Created on May 3, 2021

@author: zollen
@url: https://www.youtube.com/watch?v=1NXryMoU7Ho&list=PLvcbYUQ5t0UHOLnBzl46_Q6QKtFgfMGc3&index=39&ab_channel=ritvikmath
@title: Seasonal-Trend Decomposition using LOESS (STL)
'''


import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

ice_cream_interest = pd.read_csv('ice_cream_interest.csv')
ice_cream_interest.set_index('month', inplace=True)
ice_cream_interest = ice_cream_interest.asfreq(pd.infer_freq(ice_cream_interest.index))

print(ice_cream_interest.head())

if False:
    plt.figure(figsize=(10,4))
    plt.plot(ice_cream_interest)
    for year in range(2004,2021):
        plt.axvline(datetime(year,1,1), color='k', linestyle='--', alpha=0.5)



stl = STL(ice_cream_interest)
result = stl.fit()

seasonal, trend, resid = result.seasonal, result.trend, result.resid

if True:
    plt.figure(figsize=(8,6))
    
    plt.subplot(4,1,1)
    plt.plot(ice_cream_interest)
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


if True:
    estimated = trend + seasonal
    plt.figure(figsize=(12,4))
    plt.plot(ice_cream_interest)
    plt.plot(estimated)
    plt.legend(['Ice Cream Interests', 'Trend + Seasonal'], fontsize=16)
    plt.title('Original Data vs Trend + Seasonal')


'''
Anomaly Detection
'''
resid_mu = resid.mean()
resid_dev = resid.std()

lower = resid_mu - 3*resid_dev
upper = resid_mu + 3*resid_dev

if True:
    plt.figure(figsize=(10,4))
    plt.plot(resid)
    
    plt.fill_between([datetime(2003,1,1), datetime(2021,8,1)], lower, upper, color='g', alpha=0.25, linestyle='--', linewidth=2)
    plt.xlim(datetime(2003,9,1), datetime(2020,12,1))
    
anomalies = ice_cream_interest[(resid < lower) | (resid > upper)]

if True:
    plt.figure(figsize=(10,4))
    plt.plot(ice_cream_interest)
    for year in range(2004,2021):
        plt.axvline(datetime(year,1,1), color='k', linestyle='--', alpha=0.5)
        
    plt.scatter(anomalies.index, anomalies.interest, color='r', marker='D')

plt.show()