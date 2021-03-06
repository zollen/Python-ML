'''
Created on Apr. 19, 2021

@author: zollen
@url: https://www.youtube.com/watch?v=XPwCo4cqqt0&list=PLvcbYUQ5t0UHOLnBzl46_Q6QKtFgfMGc3&index=43
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


train_end = datetime(1999, 7, 1)
test_end = datetime(2000, 1, 1)

train_data = lim_catfish_sales[:train_end]
test_data = lim_catfish_sales[train_end + timedelta(days = 1):test_end]

'''
Let's try a different technique. Rolling Forecast Origin
It use one month to predict the next, it returns much better prediction
The error is now centered around 0
'''
rolling_predictions = test_data.copy()

my_order = (0, 1, 0)  # Non-Seasonal AR(0) and MA(0) with integration of 1 
my_seasonal_order = (1, 0, 1, 12) # Seasonal AR(1) and MA(1) and freqeuncy of 12
for train_end in test_data.index:
    train_data = lim_catfish_sales[:train_end - timedelta(days = 1)]
    model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
    model_fit = model.fit()
    pred = model_fit.forecast()
    rolling_predictions[train_end] = pred

rolling_residuals = test_data - rolling_predictions

if False:
    plt.figure(figsize=(10,4))
    plt.plot(rolling_residuals)
    plt.title('Rolling Forecase Residuals from SARIMA Model', fontsize=20)
    plt.ylabel('Error', fontsize=16)
    for month in ['08', '09', '10', '11', '12']:
        plt.axvline(pd.to_datetime('1999-' + month), color='k', linestyle='--', alpha=0.2)
    plt.axvline(pd.to_datetime('2000-01'), color='k', linestyle='--', alpha=0.2)
    plt.show()
    
if False:
    plt.figure(figsize=(10,4))
    plt.plot(lim_catfish_sales)
    plt.plot(rolling_predictions)
    plt.legend(('Data', 'Predictions'), fontsize=16)
    plt.title('Production', fontsize=20)
    plt.ylabel('Sales', fontsize=16)
    for year in range(start_date.year, end_date.year):
        plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
    plt.show()
    
'''
The rolling prediction is much better
'''
print('Mean Absolute Percent Error: ', round(np.mean(abs(rolling_residuals/test_data)), 4))
print('Root Mean Squared Error: ', np.sqrt(np.mean(rolling_residuals**2)))

'''
Let's introduce an anomaly
At December 1 1998
lim_catfish_sales[datetime(1998,12,1)] = 10000    
'''
lim_catfish_sales[datetime(1998,12,1)] = 10000
if False:
    plt.figure(figsize=(10,4))
    plt.plot(lim_catfish_sales)
    plt.title('CatFish Sales in 1000s of Pounds', fontsize=20)
    plt.ylabel('Sales', fontsize=16)
    for year in range(start_date.year, end_date.year):
        plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
    plt.show()

train_end = datetime(1999, 7, 1)
test_end = datetime(2000, 1, 1)

train_data = lim_catfish_sales[:train_end]
test_data = lim_catfish_sales[train_end + timedelta(days = 1):test_end]

rolling_predictions = test_data.copy()

my_order = (0, 1, 0)  # Non-Seasonal AR(0) and MA(0) with integration of 1 
my_seasonal_order = (1, 0, 1, 12) # Seasonal AR(1) and MA(1) and freqeuncy of 12
for train_end in test_data.index:
    train_data = lim_catfish_sales[:train_end - timedelta(days = 1)]
    model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
    model_fit = model.fit()
    pred = model_fit.forecast()
    rolling_predictions[train_end] = pred

rolling_residuals = test_data - rolling_predictions

if False:
    plt.figure(figsize=(10,4))
    plt.plot(rolling_residuals)
    plt.title('Rolling Forecase Residuals from SARIMA Model', fontsize=20)
    plt.ylabel('Error', fontsize=16)
    plt.axhline(0, color='k', linestyle='--', alpha=0.2)
    plt.show()
    
if False:
    plt.figure(figsize=(10,4))
    plt.plot(lim_catfish_sales)
    plt.plot(rolling_predictions)
    plt.legend(('Data', 'Predictions'), fontsize=16)
    plt.title('Production', fontsize=20)
    plt.ylabel('Sales', fontsize=16)
    for year in range(start_date.year, end_date.year):
        plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
    plt.show()
    
'''
The rolling prediction is much better
'''
print('Mean Absolute Percent Error: ', round(np.mean(abs(rolling_residuals/test_data)), 4))
print('Root Mean Squared Error: ', np.sqrt(np.mean(rolling_residuals**2)))

'''
Detcting the anomaly    
Attempt #1: Deviation Method
Calculating a standard deviation in each group of data points and look for any unusual
'''
rolling_deviations = pd.Series(dtype=float, index = lim_catfish_sales.index)
for date in rolling_deviations.index:
    # get the window ending at this data point
    window = lim_catfish_sales.loc[:date]
    # get the deviation within this window
    rolling_deviations.loc[date] = window.std()
    
# get the first difference in devation between one time point and the next
diff_rolling_deviations = rolling_deviations.diff()
diff_rolling_deviations = diff_rolling_deviations.dropna()

if False:
    # we ignore the first highest point because initial standard deviation calculation using only a few points
    # This method is not good because if the anomaly happen early on, it would be hard to know for sure.
    # another drawback is that anomaly not necessary has extreme values.
    plt.figure(figsize=(10,4))
    plt.plot(diff_rolling_deviations)
    plt.title('Deviation First Differences', fontsize=20)
    plt.ylabel('Sales', fontsize=16)
    for year in range(start_date.year, end_date.year):
        plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
    plt.show()
    
'''
Attempt #2: Seasonal Method
better method
We know there is a sessonal pattern in the data, so we are going to use that
Calculating the standard deviation using the same date of each month.
'''
month_deviations = lim_catfish_sales.groupby(lambda d: d.month).std()

if False:
    # december has a large spike
    plt.figure(figsize=(10,4))
    plt.plot(month_deviations)
    plt.title('Deviation by Month', fontsize=20)
    plt.ylabel('Sales', fontsize=16)
    plt.show()
    
december_data = lim_catfish_sales[lim_catfish_sales.index.month == 12]
print(december_data)

min_dev = 9999999
curr_anomaly = None
for date in december_data.index:
    other_data = december_data[december_data.index != date]
    curr_dev = other_data.std()
    if curr_dev < min_dev:
        min_dev = curr_dev
        curr_anomaly = date
        
print("ANOMALY: ", curr_anomaly)

'''
There are many more much better anomaly detection methods
What to do with the anomaly??
Simple Idea: use the mean of other months
'''
adjusted_data = lim_catfish_sales.copy()
adjusted_data.loc[curr_anomaly] = december_data[(december_data.index != curr_anomaly) & (december_data.index < test_data.index[0])].mean()
lim_catfish_sales = adjusted_data

if False:
    plt.figure(figsize=(10,4))
    plt.plot(lim_catfish_sales, color='firebrick', alpha=0.4)
    plt.plot(adjusted_data)
    plt.title('Catfish Sales in 1000s of Pounds', fontsize=20)
    plt.ylabel('Sales', fontsize=16)
    for year in range(start_date.year,end_date.year):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
#    plt.axvline(curr_anomaly, color='k', alpha=0.7)
    plt.show()

for train_end in test_data.index:
    train_data = lim_catfish_sales[:train_end - timedelta(days = 1)]
    model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
    model_fit = model.fit()
    pred = model_fit.forecast()
    rolling_predictions[train_end] = pred

rolling_residuals = test_data - rolling_predictions

if False:
    plt.figure(figsize=(10,4))
    plt.plot(rolling_residuals)
    plt.title('Rolling Forecase Residuals from SARIMA Model', fontsize=20)
    plt.ylabel('Error', fontsize=16)
    plt.axhline(0, color='k', linestyle='--', alpha=0.2)
    plt.show()
    
if False:
    plt.figure(figsize=(10,4))
    plt.plot(lim_catfish_sales)
    plt.plot(rolling_predictions)
    plt.legend(('Data', 'Predictions'), fontsize=16)
    plt.title('Production', fontsize=20)
    plt.ylabel('Sales', fontsize=16)
    for year in range(start_date.year, end_date.year):
        plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
    plt.show()
    
print('Mean Absolute Percent Error: ', round(np.mean(abs(rolling_residuals/test_data)), 4))
print('Root Mean Squared Error: ', np.sqrt(np.mean(rolling_residuals**2)))