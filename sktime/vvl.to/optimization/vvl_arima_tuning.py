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
import threading
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

fh = ForecastingHorizon(y_to_test.index, is_relative=False)
    
class Worker(threading.Thread):     
    
    def __init__(self, threadId):
        threading.Thread.__init__(self)
        self.threadID = threadId
        
    def run(self):
        
        global karts
        
        try:       
            while True:
                self.evaludate(karts.pop(0))   
   
        except:
            pass
            
        
    def evaludate(self, params):
        exog = pd.DataFrame({'Date': y.index})
        exog = exog.set_index(exog['Date'])
        exog.drop(columns=['Date'], inplace=True)
        
        cols = []
        for coeff in params:
            exog['sin' + str(coeff)] = np.sin(coeff * np.pi * exog.index.dayofyear / 365.25)
            exog['cos' + str(coeff)] = np.cos(coeff * np.pi * exog.index.dayofyear / 365.25)
            cols.append('sin' + str(coeff))
            cols.append('cos' + str(coeff))
            
        exog_train = exog.loc[y_to_train.index]
        exog_test = exog.loc[y_to_test.index]
            
        model = ARIMA(order=(2, 1, 2), seasonal_order=(0, 1, 0, 8), suppress_warnings=True)
        model.fit(y_to_train['Prices'], X=exog_train[cols])
        y_forecast = model.predict(fh, X=exog_test[cols])
        score = mean_absolute_percentage_error(y_to_test, y_forecast)
        
        '''
        For using AIC for optimization
        AutoETS: forecaster._fitted_forecaster.bic relevant class is ETSResults from statsmodels
        ExponentialSmoothing: forecaster._fitted_forecaster.bic relevant class is HoltWintersResults from statsmodels
        AutoARIMA: forecaster._forecaster.model_.arima_res_.bic relevant class is SARIMAXResults from statsmodels
        ARIMA: forecaster._forecaster.arima_res_.bicrelevant class is SARIMAXResults from statsmodels
        '''
        
        global best_params, best_score, wlock
        
        wlock.acquire()

        if best_params == None or best_score > score:
            best_params = params
            best_score = score
            print("InProgress[%10d][%3d][%3s] => Score: %0.8f" % (len(karts), threading.activeCount(), str(self.threadID), best_score), " params: ", best_params)
        
        wlock.release()


best_params = None
best_score = 99999999
wlock = threading.Lock()

karts = []
all_coeffs = [20, 81, 86, 94, 101, 102, 107, 148, 206, 296, 297,
              306, 307, 349, 352, 367, 379, 392, 419, 420, 425, 467, 472 ]

start_param = 0
count = 0
for coeffs in list(itertools.combinations(all_coeffs, 10)):
    if count >= start_param:
        karts.append(coeffs)
    count = count + 1
  
orig_size = len(karts) 
print("total: ", orig_size)  
print()


threads = []
for id in range(0, 150):
    threads.append(Worker(id))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print("=========================================================")
print("Orig: ", orig_size, " Remaining: ", len(karts))
print("FINAL RESULT: %0.8f" % best_score, " params: ", best_params)


'''
MAPE: 0.00874232

Searching for all possible regressors that can lower the mape without exog
20 : 0.00868552
81 : 0.00868402
86 : 0.00864950
94 : 0.00857343
101 : 0.00865395
102 : 0.00836136
103 : 0.00858858
106 : 0.00838859
107 : 0.00834268
120 : 0.00861365
122 : 0.00871660
146 : 0.00858785
147 : 0.00846231
148 : 0.00862946
151 : 0.00870307
154 : 0.00869671
160 : 0.00873079
166 : 0.00872009
182 : 0.00868611
172 : 0.00870681
173 : 0.00870667
176 : 0.00872720
177 : 0.00872313
178 : 0.00870347
193 : 0.00870975
202 : 0.00859713
206 : 0.00872939
207 : 0.00870188
211 : 0.00854945
265 : 0.00867982
252 : 0.00872257
262 : 0.00871665
260 : 0.00872238
269 : 0.00870891
271 : 0.00869399
261 : 0.00874062
263 : 0.00873903
264 : 0.00871780
267 : 0.00867140
227 : 0.00871475
296 : 0.00870132
297 : 0.00870784
305 : 0.00873459
306 : 0.00867852
307 : 0.00861823
309 : 0.00872773
348 : 0.00872624
349 : 0.00871261
352 : 0.00871002
365 : 0.00874193
366 : 0.00871805
367 : 0.00869459 
392 : 0.00868421
351 : 0.00874119
379 : 0.00867866
419 : 0.00865721
420 : 0.00855733
425 : 0.00863723
434 : 0.00867162
459 : 0.00872998
466 : 0.00869470
467 : 0.00868690
468 : 0.00870607
469 : 0.00872755
472 : 0.00866974
473 : 0.00870345 
478 : 0.00872459
507 : 0.00851959
516 : 0.00872634
519 : 0.00861583
523 : 0.00873664
524 : 0.00869169
531 : 0.00869656
541 : 0.00873622
543 : 0.00869285
544 : 0.00873271
547 : 0.00869536

(20, 86, 102, 107, 147): 0.00769097
(86, 94, 102, 103, 106): 0.00740850
(20, 101, 103, 147)    : 0.00740587
(81, 94, 106, 148)     : 0.00737365
(20, 86, 102, 120, 146): 0.00732646
(20, 94, 147, 148, 151): 0.00709723
(20, 81, 86, 94, 101, 102, 106, 148, 207, 271) : 0.00706450
(20, 81, 86, 94, 101, 102, 107, 148, 206, 296) : 0.00691188  (44352165:44347560)
(20, 81, 86, 94, 101, 102, 107, 211, 352, 211) : 0.00685742 
'''

