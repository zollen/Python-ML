'''
Created on Jun. 25, 2021

@author: zollen
@url: https://lavinei.github.io/pybats//
@url: https://lavinei.github.io/pybats/analysis.html#analysis
'''

from sktime.datasets import load_airline
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
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


y = load_airline()

prior_length = 5    # Number of days of data used to set prior
k = 1               # Forecast horizon
rho = 0.8           # Random effect discount factor to increase variance of forecast distribution

forecast_start = y.index[-36] 
forecast_end = y.index[-1] 
Y = y.values

exog = pd.DataFrame()
for i in range(1, 6):
    exog['sin' + str(i)] = np.sin(i * np.pi * y.index.dayofyear / 365.25)
    exog['cos' + str(i)] = np.cos(i * np.pi * y.index.dayofyear / 365.25)
exog = exog.values

mod, samples = analysis(Y, exog,
            k=k, 
            forecast_start=forecast_start, 
            forecast_end=forecast_end,
            family='poisson',
            ntrend=2,  
            prior_length=prior_length, 
            dates=y.index,
            rho=rho,
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


plot_length = 36
data_steps = Y[-plot_length:]
forecast = forecast.flatten()

print("Shape(samples): ", samples.shape)
print("Shape(forecast): ", forecast.shape)


print("RMSE: %0.4f" % mean_absolute_percentage_error(data_steps, forecast))

plt.plot(data_steps)
plt.plot(forecast)

plt.legend(('Airline', 'Predictions'))


plt.show()