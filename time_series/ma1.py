'''
Created on Apr. 11, 2021

@author: zollen
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime, timedelta
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)


'''
y(t) = 50 + 0.4e(t-1) + 0.3e(t-2) + e(t)
e(t) ~ N(0,1)  <- normal distribution mean = 0, standard deviation = 1
'''

errors = np.random.normal(0, 1, 400)
date_index = pd.date_range(start='9/1/2019', end='1/1/2020')

mu = 50
series = []
for t in range(1, len(date_index) +1):
    series.append(mu + 0.4 * errors[t-1] + 0.3* errors[t-2] + errors[t])
    
series = pd.Series(series, date_index)
series = series.asfreq(pd.infer_freq(series.index))

if False:
    plt.figure(figsize=(10,4))
    plt.plot(series)
    plt.axhline(mu, linestyle='--', color='grey')
    plt.show()
    
#def calc_corr(series, lag):
#    return pearsonr(series[:-lag], series[lag:])[0]

'''
ACF
'''
acf_vals = acf(series)
num_lags = 10
if False:
    plt.bar(range(num_lags), acf_vals[:num_lags])
    plt.show()
    
'''
PACF
'''   
pacf_vals = pacf(series)
num_lags = 25 
if False:
    plt.bar(range(num_lags), pacf_vals[:num_lags])
    plt.show()
    
train_end = datetime(2019, 12, 30)
test_end = datetime(2020, 1, 1)  # we predict only two periods (2019-12-31 -> 2020-1-1)

train_data = series[:train_end]
test_data = series[train_end + timedelta(days=1):test_end]

'''
ARIMA (0, 0, 2) with 2 lags (MA(2))
It predicts the const 49.8942 which is very close to 50
It predicts the L1 0.3330 which is somewhat close to 0.4
It predicts the L2 0.3026 which is very close to 0.3
The predicted values are pretty close to the original model
All P values are very low (0.00, 0.00, 0.002), which are good
Our predicted model looks like this:
y(t) = 49.9842 + 0.3330 e(t-1) + 0.3026 e(t-2)
'''
model = ARIMA(train_data, order=(0, 0, 2))
model_fit = model.fit()

print(model_fit.summary())


pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]

predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
residuals = test_data - predictions

if False:
    plt.figure(figsize=(10,4))
    plt.plot(series[-14:])
    plt.plot(predictions)
    plt.legend(('Data', 'Predictions'), fontsize=16)
    plt.show()

print('Mean Absolute Percent Error: ', round(np.mean(abs(residuals/test_data)), 4))
print('Root Mean Squared Error: ', np.sqrt(np.mean(residuals**2)))