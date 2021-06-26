'''
Created on Jun. 24, 2021

@author: zollen
@url: https://lavinei.github.io/pybats//
@url: https://lavinei.github.io/pybats/analysis.html#analysis

'''


# Import the necessary libraries
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from pybats.shared import load_sales_example
from pybats.analysis import *
from pybats.point_forecast import *
from pybats.plot import *
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

# Load example sales and advertising data. Source: Abraham & Ledolter (1983)
data = load_sales_example()             
print(data.head())

'''
The sales are integer valued counts, which we model with a Poisson Dynamic Generalized 
Linear Model (DGLM). Second, we extract the outcome (Y) and covariates (X) from this 
dataset. We'll set the forecast horizon k=1 for this example. We could look at multiple 
forecast horizons by setting k to a larger value. analysis, a core PyBATS function, will 
automatically:
'''

Y = data['Sales'].values
X = data['Advertising'].values.reshape(-1,1)

k = 1                                               # Forecast 1 step ahead
forecast_start = 15                                 # Start forecast at time step 15
forecast_end = 35                                   # End forecast at time step 35 (final time step)

mod, samples = analysis(Y, X, family="poisson",
    ntrend=2,                           # Use an intercept and local slope
    forecast_start=forecast_start,      # First time step to forecast on
    forecast_end=forecast_end,          # Final time step to forecast on
    k=k,                                # Forecast horizon. If k>1, default is to forecast 1:k steps ahead, marginally
    prior_length=6,                     # How many data point to use in defining prior
    rho=.5,                             # Random effect extension, which increases the forecast variance (see Berry and West, 2019)
    deltrend=0.95,                      # Discount factor on the trend component (the intercept)
    delregn=0.95                        # Discount factor on the regression component
    )

print(mod.get_coef())

# Take the median as the point forecast
'''
The samples are stored in a 3-dimensional array, with axes nsamps * forecast length * k
 - nsamps is the number of samples drawn from the forecast distribution
 - forecast length is the number of time steps between forecast_start and forecast_end
 - k is the forecast horizon, or the number of steps that were forecast ahead
'''
forecast = median(samples)   

print("RMSE: %0.4f" % mean_absolute_percentage_error(
                        Y[forecast_start:forecast_end + k], 
                        forecast.flatten()))
                            

# Plot the 1-step ahead point forecast plus the 95% credible interval
fig, ax = plt.subplots(1, 1, figsize=(8, 6))   
ax = plot_data_forecast(fig, ax, Y[forecast_start:forecast_end + k], forecast, samples,
                        dates=np.arange(forecast_start, forecast_end+1, dtype='int'))
ax = ax_style(ax, ylabel='Sales', xlabel='Time', xlim=[forecast_start, forecast_end],
              legend=['Forecast', 'Sales', 'Credible Interval'])



plt.show()