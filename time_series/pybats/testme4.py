'''
Created on Jun. 25, 2021

@author: zollen
@url: https://lavinei.github.io/pybats//
@url: https://lavinei.github.io/pybats/analysis.html#analysis
'''

from sktime.datasets import load_airline
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.forecasting.model_selection import temporal_train_test_split
from datetime import timedelta
import pandas as pd
import numpy as np
from pybats.analysis import *
from pybats.point_forecast import *
from pybats.plot import *
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

TRAINING_SIZE=36
TEST_SIZE = 36

y = load_airline()

y_to_train, y_to_test = temporal_train_test_split(y, test_size=TEST_SIZE)


forecast_start = y_to_train.index[-TRAINING_SIZE] 
forecast_end = y_to_train.index[-1] 


prior_length = 5    # Number of days of data used to set prior
k = 36              # Forecast horizon
rho = 0.5           # Random effect discount factor to increase variance of forecast distribution
seasPeriods=[12]
seasHarmComponents = [[1, 5, 6]]

mod, samples = analysis(y_to_train.values, 
            k=k, 
            forecast_start=forecast_start, 
            forecast_end=forecast_end,
            family='poisson',
            seasPeriods=seasPeriods, 
            seasHarmComponents=seasHarmComponents,
            ntrend=2,  
            prior_length=prior_length, 
            dates=y_to_train.index,
            rho=rho,
            deltrend=0.95,      # Discount factor on the trend component (the intercept)
            delregn=0.98,       # Discount factor on the regression component
            delseas=0.98,       # Discount factor on the seasonal component
            ret = ['model', 'forecast'])

print(mod.get_coef())

# Take the median as the point forecast
'''
The samples are stored in a 3-dimensional array, with axes nsamps * forecast length * k
 - nsamps is the number of samples drawn from the forecast distribution
 - forecast length is the number of time steps between forecast_start and forecast_end
 - k is the forecast horizon, or the number of steps that were forecast ahead
'''
forecast = median(samples)  


forecast = forecast.flatten()[-TEST_SIZE:]

print("Shape(samples): ", samples.shape)
print("Shape(forecast): ", forecast.shape)


print("RMSE: %0.4f" % mean_absolute_percentage_error(y_to_test.values, forecast))

plt.plot(y_to_test.values)
plt.plot(forecast)

plt.legend(('Airline', 'Predictions'))


plt.show()