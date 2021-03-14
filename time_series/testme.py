'''
Created on Mar. 8, 2021

@author: zollen
'''

import warnings
import itertools
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
'''
url: https://towardsdatascience.com/using-the-pandas-resample-function-a231144194c4
MS - Month Start Frequency 
'''
y = furniture['Sales'].resample(rule='MS').mean()

if False:
    y.plot(figsize=(15, 6))
    plt.show()
    exit()

if False:
    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
    plt.show()
    exit()

if False:
    best_score = 999999
    best_param = None
    best_sparam = None
    p = d = q = range(0, 4)
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
                if results.aic < best_score:
                    best_score = results.aic
                    best_param = param
                    best_sparam = param_seasonal
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    print()
    print()
    print("Best AIC: {} Best Order {} Best Seasonal Order {}".format(best_score, best_param, best_sparam))
    exit()    
'''
The optimal model with the lowest AIC: 4.0 - ARIMA(0, 0, 0)x(0, 3, 1, 12)
'''
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 0, 0),
                                seasonal_order=(0, 3, 1, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=False)
results = mod.fit()
print('ARIMA{}x{}12 - AIC:{}'.format((0, 0, 0), (0, 3, 1, 12), results.aic))    

print(results.summary())


if False:
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()
    exit()



pred = results.get_prediction(start=36, end=47, dynamic=False)
pred_ci = pred.conf_int()
ax = y['2009':].plot(label='Observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()