'''
Created on Jun. 2, 2021

@author: zollen
@url: https://github.com/alan-turing-institute/sktime/blob/main/examples/01_forecasting.ipynb
'''

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ets import AutoETS

from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from datetime import datetime, timedelta
import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

end_date = datetime(2021, 6, 4)
start_date = end_date - timedelta(weeks=72)
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
0.0084
'''
model = AutoETS(auto=True, trend="add", sp=8, seasonal="additive", n_jobs=-1)
model.fit(y_to_train['Prices'])

y_forecast = model.predict(fh)

print("RMSE: %0.4f" % mean_absolute_percentage_error(y_to_test, y_forecast))
plt.figure(figsize=(10,4))
plt.plot(y.iloc[-128:])
plt.plot(y_to_test.index, y_forecast)
plt.legend(('Data', 'Predictions'), fontsize=16)
plt.title("ExponentialSmoothing(VVL.TO)", fontsize=20)
plt.ylabel('Price', fontsize=16) 
plt.ylim(32, 40)

plt.show()
