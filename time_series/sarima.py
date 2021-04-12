'''
Created on Apr. 11, 2021

@author: zollen
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
from time import time
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)


def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')

catfish_sales = pd.read_csv('catfish.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

catfish_sales = catfish_sales.asfreq(pd.infer_freq(catfish_sales.index))

start_date = datetime(1996, 1, 1)
end_date = datetime(2000, 1, 1)
lim_catfish_sales = catfish_sales[start_date:end_date]

if False:
    plt.figure(figsize=(10,4))
    plt.plot(lim_catfish_sales)
    plt.title('CatFish Sales in 1000s of Pounds', fontsize=20)
    plt.ylabel('Sales', fontsize=16)
    for year in range(start_date.year, end_date.year):
        plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
    plt.show()

'''    
Let's remove the trend
'''
first_diff = lim_catfish_sales.diff()[1:]
if False:
    plt.figure(figsize=(10,4))
    plt.plot(first_diff)
    plt.title('CatFish Sales in 1000s of Pounds', fontsize=20)
    plt.ylabel('Sales', fontsize=16)
    for year in range(start_date.year, end_date.year):
        plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
    plt.show()

'''   
Making the first difference seems to make it stationary. We are going to use intergration of 1
Based on ACF, we should start with a seasonal MA process with lag of one year(or 12 months)
'''
if False:
    acf_vals = acf(first_diff)
    num_lags = 20
    plt.bar(range(num_lags), acf_vals[:num_lags])
    plt.show()


'''
Based on PACF, we should start with a seaonsal AR process as there are sigificant coeffs at 11, 
    12, 13 and 14 lags
'''   
if False:
    pacf_vals = pacf(first_diff)
    plt.bar(range(num_lags), pacf_vals[:num_lags])
    plt.show()


train_end = datetime(1999, 7, 1)
test_end = datetime(2000, 1, 1)

train_data = lim_catfish_sales[:train_end]
test_data = lim_catfish_sales[train_end + timedelta(days = 1):test_end]

'''
Fit the SARIMA model
There is pure seasonal behavior so AR(0) and MA(0)
'''
my_order = (0, 1, 0)  # Non-Seasonal AR(0) and MA(0) with integration of 1 
my_seasonal_order = (1, 0, 1, 12) # Seasonal AR(1) and MA(1) and freqeuncy of 12
model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)

start = time()
model_fit = model.fit()
end = time()
print('Model Fitting Time: ', end - start)
print(model_fit.summary())

'''
The coeff of seasonal AR(12) is 0.825 which means it has positive influence to the prediction
The coeff of seasonal MA(12) is -0.5187 which means it has negative influence to the prediction
All P scores are close to 0 means they are sigificant
'''
predictions = model_fit.forecast(len(test_data))
predictions = pd.Series(predictions, index=test_data.index)
residuals = test_data - predictions
if False:
    plt.figure(figsize=(10,4))
    plt.plot(residuals)
    plt.title('Residuals from SARIMA Model', fontsize=20)
    plt.ylabel('Error', fontsize=16)
    plt.show()

'''
Above diagram shows the errors are way below 0, it means there is a systematic bias we have not 
accounted for
'''
if False:
    plt.figure(figsize=(10,4))
    plt.plot(lim_catfish_sales)
    plt.plot(predictions)
    plt.legend(('Data', 'Predictions'), fontsize=16)
    plt.title('Production', fontsize=20)
    plt.ylabel('Sales', fontsize=16)
    for year in range(start_date.year, end_date.year):
        plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
    plt.show()

'''
Above diagram shows that we are always over-predicting 
'''
    
print('Mean Absolute Percent Error: ', round(np.mean(abs(residuals/test_data)), 4))
print('Root Mean Squared Error: ', np.sqrt(np.mean(residuals**2)))

'''
Let's try a different technique. Rolling Forecast Origin
'''