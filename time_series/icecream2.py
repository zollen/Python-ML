'''
Created on Apr. 7, 2021

@author: zollen
'''

from time import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.arima_model import ARMA
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)


def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')

df_ice_cream = pd.read_csv('ice_cream.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

                           
df_ice_cream.rename('production', inplace = True)


df_ice_cream = df_ice_cream.asfreq(pd.infer_freq(df_ice_cream.index))

start_date = pd.to_datetime('2010-01-01')
df_ice_cream = df_ice_cream[start_date:]



if False:
    plt.figure(figsize=(10,4))
    plt.plot(df_ice_cream)
    plt.title("Ice Cream Production Over Time", fontsize = 20)
    plt.ylabel('Production', fontsize = 16)
    for year in range(2011, 2021):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--')
    plt.show()
        
if False:
    acf_plot = plot_acf(df_ice_cream, lags = 100)
    plt.show()
    
if False:
    # Base on PACF, we should start with Auto Regressive model with lags 1, 2, 3
    pacf_plot = plot_pacf(df_ice_cream)
    plt.show()
     
     
train_end = datetime(2018, 12, 1)
test_end = datetime(2019, 12, 1)    

train_data = df_ice_cream[:train_end]
test_data = df_ice_cream[train_end + timedelta(days = 1):test_end] 
     
model = ARMA(train_data, order=(3, 0))
start = time()
model_fit = model.fit()
end = time()     
print('Model Fitting Time: ', end - start)   
print(model_fit.summary())

'''
The lag 2 shows that P>|z| is 0.695, which is above 0.05 sigificant. For AR(3) model, we should keep
const, Lag1 and lag3

Let's use this model and predict 
'''

pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]
predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
residuals = test_data - predictions

if False:
    # The residuals shows that there may be a pattern and there is something else we did not quite capture
    plt.figure(figsize=(10,4))
    plt.plot(residuals)
    plt.title("Residulas from AR Model", fontsize = 20)
    plt.ylabel('Error', fontsize = 16)
    for year in range(2019, 2021):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--')
    plt.show()

if False:
    plt.figure(figsize=(10,4))
    plt.plot(test_data)
    plt.plot(predictions)
    plt.legend(('Data', 'Predictions'), fontsize = 16)
    plt.title('Ice Cream Production over Time', fontsize = 20)
    plt.ylabel('Production', fontsize = 16)
    for year in range(2019, 2021):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--')
    plt.show()


print('Mean Absolute Percent Error: ', round(np.mean(abs(residuals/test_data)), 4))
print('Root Mean Squared Error: ', np.sqrt(np.mean(residuals**2)))