'''
Created on Jun. 15, 2021

@author: zollen
@title: Kalman Filter for Time Series Data
@url: https://medium.com/dataman-in-ai/kalman-filter-explained-4d65b47916bf
'''

from datetime import datetime, timedelta
import pandas_datareader.data as web
import pandas as pd
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from pykalman import KalmanFilter
import numpy as np
import seaborn as sb
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

# Construct a Kalman filter
kf = KalmanFilter(transition_matrices = [1],    # The value for At. It is a random walk so is set to 1.0
                  observation_matrices = [1],   # The value for Ht.
                  initial_state_mean = 0,       # Any initial value. It will converge to the true state value.
                  initial_state_covariance = 1, # Sigma value for the Qt in Equation (1) the Gaussian distribution
                  observation_covariance=1,     # Sigma value for the Rt in Equation (2) the Gaussian distribution
                  transition_covariance=2.2)    # A small turbulence in the random walk parameter 1.0
# Get the Kalman smoothing
state_means, _ = kf.filter(data.values)


# Call it KF_mean
data['KF_mean'] = np.array(state_means)
print(data.head())

if True:
    data[['Prices','KF_mean']].plot()
    plt.title(F'Kalman Filter estimates for [{STOCK}]')
    plt.legend([STOCK,'Kalman Estimate'])
    plt.xlabel('Day')
    plt.ylabel('Price')
    
    
print("RMSE: %0.4f" % mean_absolute_percentage_error(data['Prices'], data['KF_mean']))
    
    
    
    
plt.show()