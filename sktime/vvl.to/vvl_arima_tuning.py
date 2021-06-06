'''
Created on Jun. 2, 2021

@author: zollen
@url: https://tanzu.vmware.com/content/blog/forecasting-time-series-data-with-multiple-seasonal-periods
@url: https://medium.com/intive-developers/forecasting-time-series-with-multiple-seasonalities-using-tbats-in-python-398a00ac0e8a
'''

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.arima import ARIMA

from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from datetime import datetime, timedelta
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import itertools
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
    
    '''
    Both acf & pacf shows that the residuals still has some seasonal patterns.
    We need Fourier regressors to handle these remaining seasonal patterns.
    '''
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.set_ylabel("ACF(Residual)")
    plot_acf(result.resid, title="", ax=ax1)
    ax2.set_ylabel("PACF(Residual)")
    plot_pacf(result.resid, title="", ax=ax2)
    
    plt.show()
    


fh = ForecastingHorizon(y_to_test.index, is_relative=False)

'''
MAPE: 0.00874232

20  : 0.00868552
81  : 0.00868402
86  : 0.00864950
94  : 0.00857343
102 : 0.00836136
103 : 0.00858858
106 : 0.00838859
107 : 0.00834268
147 : 0.00846231
(20, 86, 102, 107, 147): 0.00769097
(86, 94, 102, 103, 106): 0.00740850
'''

all_coeffs = [20, 81, 86, 94, 102, 103, 106, 107, 147]
params = []
models = []
aics = []
mapes = []

for length in range(1, len(all_coeffs) + 1):
     
    for coeffs in list(itertools.combinations(all_coeffs, length)):
        
        exog = pd.DataFrame({'Date': y.index})
        exog = exog.set_index(exog['Date'])
        exog.drop(columns=['Date'], inplace=True)
        
        cols = []
        for coeff in coeffs:
            exog['sin' + str(coeff)] = np.sin(coeff * np.pi * exog.index.dayofyear / 365.25)
            exog['cos' + str(coeff)] = np.cos(coeff * np.pi * exog.index.dayofyear / 365.25)
            cols.append('sin' + str(coeff))
            cols.append('cos' + str(coeff))
            
        exog_train = exog.loc[y_to_train.index]
        exog_test = exog.loc[y_to_test.index]
            
        model = ARIMA(order=(2, 1, 2), seasonal_order=(0, 1, 0, 8), suppress_warnings=True)
        model.fit(y_to_train['Prices'], X=exog_train[cols])
        y_forecast = model.predict(fh, X=exog_test[cols])
        
        params.append(coeffs)
        models.append(model)
        mapes.append(mean_absolute_percentage_error(y_to_test, y_forecast))



min_ind = mapes.index(min(mapes)) 
bestparam = params[min_ind]
bestmodel = models[min_ind]
print(bestmodel.summary())
print('MAPE: %0.8f' % min(mapes), 'best_param: ', bestparam)






'''
print("RMSE: %0.4f" % mean_absolute_percentage_error(y_to_test, y_forecast))
plt.figure(figsize=(10,4))
plt.plot(y.iloc[-128:])
plt.plot(y_to_test.index, y_forecast)
plt.legend(('Data', 'Predictions'), fontsize=16)
plt.title("ARIMA(VVL.TO)", fontsize=20)
plt.ylabel('Price', fontsize=16) 
plt.ylim(32, 40)
plt.fill_between(
            y_to_test.index,
            y_forecast_int["lower"],
            y_forecast_int["upper"],
            alpha=0.2,
            color='b'
        )


plt.show()
'''