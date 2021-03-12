'''
Created on Mar. 8, 2021

@author: zollen
'''

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams
import statsmodels.api as sm
import matplotlib

warnings.filterwarnings("ignore")
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
plt.style.use('fivethirtyeight')

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


df = pd.read_csv('superstores.csv', encoding='unicode_escape')
furniture = df.loc[df['Category'] == 'Furniture']

cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Segment', 'City', 'State', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit', 'Product Base Margin']
furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')

furniture['Order Date'] = pd.to_datetime(furniture['Order Date'], format="%m/%d/%Y")

furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
furniture = furniture.set_index('Order Date')
y = furniture['Sales'].resample('MS').mean()

if False:
    y.plot(figsize=(15, 6))
    plt.show()

if False:
    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
    plt.show()

if False:
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))

    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
        
'''
The optimal model with the lowest AIC: 369.31400816630793 - ARIMA(1, 1, 1)x(0, 1, 1, 12)
'''
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print('ARIMA{}x{}12 - AIC:{}'.format((1, 1, 1), (0, 1, 1, 12), results.aic))    

pred = results.get_prediction(start=pd.to_datetime('2012-01-13'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()