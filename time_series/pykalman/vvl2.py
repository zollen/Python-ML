'''
Created on Jun. 21, 2021

@author: zollen
'''

from datetime import datetime, timedelta
import pandas_datareader.data as web
import pandas as pd
import seaborn as sb
import numpy as np
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
sb.set_style('whitegrid')

end_date = datetime(2021, 6, 4)
start_date = end_date - timedelta(weeks=64)

STOCK="VVL.TO"

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

data = get_stock(STOCK)



r = 0.01      # measurement error
q = 0.15      # process noise variance
    

#initialization: guess (will be removed later)
estims = [ 22.0 ]
estims_uncertainty = [ 100.0 * 100.0  + q ]

for rnd in range(0, len(data)):
    # state update
    K = estims_uncertainty[-1] / (estims_uncertainty[-1] + r)
    next_estims = estims[-1] + K * (data['Prices'].iloc[rnd] - estims[-1])
    next_estims_uncertainty = (1 - K) * estims_uncertainty[-1]
    
    # predict
    estims.append(next_estims)  # constant dynamic model
    estims_uncertainty.append(next_estims_uncertainty + q)
    
# remove initial guesses
estims = estims[1:]
estims_uncertainty = estims_uncertainty[1:]
    

np.set_printoptions(formatter={"float_kind": lambda x: "%0.4f" % x})
print(np.array(estims))   
print(np.array(estims_uncertainty))
print("RMSE: %0.4f" % mean_absolute_percentage_error(data['Prices'], estims))
 


plt.figure(figsize=(16, 10))
plt.plot(data.index, data['Prices'], marker='o')
plt.plot(data.index, estims, marker='x', color='r', alpha=0.4)
plt.title('VVL.TO')
plt.legend(['Prices', 'Estimates'])
plt.xlabel('Date')
plt.ylabel('Prices ($)')
plt.show()    
