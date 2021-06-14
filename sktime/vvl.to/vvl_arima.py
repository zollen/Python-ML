'''
Created on Jun. 2, 2021

@author: zollen
@url: https://github.com/alan-turing-institute/sktime/blob/main/examples/01_forecasting.ipynb
'''

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.arima import ARIMA

from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from datetime import datetime, timedelta
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

end_date = datetime(2021, 6, 4)
start_date = end_date - timedelta(weeks=64)
test_size=36

def get_stock(TICKER):
   
    vvl = web.DataReader(TICKER, 'yahoo', start=start_date, end=end_date).Close
    vvl.index = [d.date() for d in vvl.index]
    
    prices = pd.DataFrame({
                            'Date' : vvl.index, 
                            'Prices' : vvl.values, 
                           })
    
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = prices.set_index('Date')
    prices = prices.asfreq(pd.infer_freq(prices.index), method="pad")
    prices['Prices'] = prices['Prices'].astype('float64')
    
    return prices
  

        
           
y = get_stock('VVL.TO')

#plot_series(y);
y_to_train, y_to_test = temporal_train_test_split(y, test_size=test_size)
#plot_series(y_to_train, y_to_test, labels=["y_train", "y_test"])
#plt.show()

if False:
    stl = STL(y)
    result = stl.fit()
    
    seasonal, trend, resid = result.seasonal, result.trend, result.resid
    plt.figure(figsize=(8,6))
    
    plt.subplot(4,1,1)
    plt.plot(y)
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
    


fh = ForecastingHorizon(y_to_test.index, is_relative=False)

'''
(0,0,0)(1,1,0,12): 0.0125
(2,1,2)(0,1,0,8) : 0.0091
'''
model = ARIMA(order=(2, 1, 2), seasonal_order=(0, 1, 0, 8), suppress_warnings=True)
kk = model.fit(y_to_train['Prices'])
print(model.summary())

y_forecast, y_forecast_int = model.predict(fh, return_pred_int=True, alpha=0.05)

rmse = mean_absolute_percentage_error(y_to_test, y_forecast)
rmse = round(rmse, 6)
print("RMSE: %0.4f" % rmse)

figure, ax = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=1)

ax[0].plot(y.iloc[-128:])
ax[0].plot(y_to_test.index, y_forecast)
ax[0].legend(('Data', 'Predictions'), fontsize=16)
ax[0].set_title(F"ARIMA(VVL.TO) {rmse}", fontsize=20)
#plt.xticks(rotation=90)
ax[0].set_ylabel('Price', fontsize=16) 
ax[0].set_ylim(32, 40)
ax[0].fill_between(
            y_to_test.index,
            y_forecast_int["lower"],
            y_forecast_int["upper"],
            alpha=0.2,
            color='b'
        )





def exog_create():
    
    exog = pd.DataFrame({'Date': y.index})
    exog = exog.set_index(exog['Date'])
    exog.drop(columns=['Date'], inplace=True)
    
    freqs = [20, 81, 86, 94, 101, 102, 107, 146, 524, 147]
    coeffs = [1.5870266419444485, 0.3186192943552314, 0.6017395178798809, 1.5687491458100933, 1.8466539435276494, 0.36883709046035296, 1.929456217767924, 1.841477121316948, 0.874471206109826, 1.005621522861652]
    cols = ['sin20', 'cos20', 'sin81', 'cos81', 'sin86', 'cos86',
             'sin94', 'cos94', 'sin101', 'cos101', 'sin102', 'cos102', 
             'sin107', 'cos107', 'sin146', 'cos146', 'sin524', 'cos524',
             'sin147', 'cos147']
    
    for freq, coeff in zip(freqs, coeffs):
        exog['sin' + str(freq)] = coeff * np.sin(freq * np.pi * exog.index.dayofyear / 365.25)
        exog['cos' + str(freq)] = coeff * np.cos(freq * np.pi * exog.index.dayofyear / 365.25)     
    
    exog = exog[cols]
    
    exog_train = exog.loc[y_to_train.index]
    exog_test = exog.loc[y_to_test.index]
    
    return exog_train, exog_test
    

exog_train, exog_test = exog_create()

model = ARIMA(order=(2, 1, 2), seasonal_order=(0, 1, 0, 8), suppress_warnings=True)
kk = model.fit(y_to_train['Prices'], X=exog_train)
print(model.summary())

y_forecast, y_forecast_int = model.predict(fh, X=exog_test, return_pred_int=True, alpha=0.05)

rmse = mean_absolute_percentage_error(y_to_test, y_forecast)
rmse = round(rmse, 6)
print("RMSE: %0.4f" % rmse)

ax[1].plot(y.iloc[-128:])
ax[1].plot(y_to_test.index, y_forecast)
ax[1].legend(('Data', 'Predictions'), fontsize=16)
ax[1].set_title(F"ARIMA(VVL.TO) {rmse}", fontsize=20)
#plt.xticks(rotation=90)
ax[1].set_ylabel('Price', fontsize=16) 
ax[1].set_ylim(32, 40)
ax[1].fill_between(
            y_to_test.index,
            y_forecast_int["lower"],
            y_forecast_int["upper"],
            alpha=0.2,
            color='b'
        )


plt.show()
