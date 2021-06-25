'''
Created on Jun. 25, 2021

@author: zollen
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

forecast_start = y.index[-31] 
forecast_end = y.index[-1] 
Y = y.values

exog = pd.DataFrame()
for i in range(1, 8):
    exog['sin' + str(i)] = np.sin(i * np.pi * y.index.dayofyear / 365.25)
    exog['cos' + str(i)] = np.cos(i * np.pi * y.index.dayofyear / 365.25)
exog = exog.values

mod, samples = analysis(Y, exog,
            k, forecast_start, forecast_end,
            family='poisson',
            ntrend=2,  
            prior_length=prior_length, 
            dates=y.index,
            rho=rho,
            ret = ['model', 'forecast'])

print(mod.get_coef())

forecast = median(samples)  


plot_length = 30
data_1step = Y[-plot_length:]
samples_1step = samples[:,-plot_length-1:-1,0]
print("Shape(samples): ", samples.shape)
print("Shape(sample_1step): ", samples_1step.shape)

preds = []
for i in range(plot_length):
    preds.append(np.mean(samples_1step[:,i]))

print("RMSE: %0.4f" % mean_absolute_percentage_error(data_1step, preds))

plt.plot(data_1step)
plt.plot(preds)

plt.legend(('Airline', 'Predictions'))


plt.show()