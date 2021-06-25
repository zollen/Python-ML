'''
Created on Jun. 24, 2021

@author: zollen
'''


# Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from pybats.shared import load_sales_example2
from pybats.analysis import *
from pybats.point_forecast import *
from pybats.plot import *
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

# Load example sales and advertising data. Source: Abraham & Ledolter (1983)
data = load_sales_example2()             
print(data.head())

'''
The sales are integer valued counts, which we model with a Poisson Dynamic Generalized 
Linear Model (DGLM). Second, we extract the outcome (Y) and covariates (X) from this 
dataset. We'll set the forecast horizon k=1 for this example. We could look at multiple 
forecast horizons by setting k to a larger value. analysis, a core PyBATS function, will 
automatically:
'''

prior_length = 21   # Number of days of data used to set prior
k = 1               # Forecast horizon
rho = 0.3           # Random effect discount factor to increase variance of forecast distribution
deltrend = 0.98     # Discount factor on the trend component
delregn = 0.98      # Discount factor on the regression component
delseas = 0.98      # Discount factor on the seasonal component
delhol = 1          # Discount factor on the holiday component

forecast_samps = 1000  # Number of forecast samples to draw
forecast_start = pd.to_datetime('2018-01-01') # Date to start forecasting
forecast_end = pd.to_datetime('2018-06-01')   # Date to stop forecasting
Y = data['Sales'].values
X = data[['Price', 'Promotion']].values       # End forecast at time step 35 (final time step)

seasPeriods=[7]
seasHarmComponents = [[1,2,3]]

mod, samples = analysis(Y, X,
            k, forecast_start, forecast_end, nsamps=forecast_samps,
            family='poisson',
            seasPeriods=seasPeriods, 
            seasHarmComponents=seasHarmComponents,
            prior_length=prior_length, 
            dates=data.index,
            rho=rho,
            deltrend = deltrend,
            delregn=delregn,
            delseas=delseas,
            delhol=delhol,
            ret = ['model', 'forecast'])

print(mod.get_coef())

# Take the median as the point forecast
forecast = median(samples)
                           

plot_length = 30
data_1step = data.loc[forecast_end-pd.DateOffset(30):forecast_end]
samples_1step = samples[:,-31:,0]
fig, ax = plt.subplots(1,1, figsize=(8, 6))
ax = plot_data_forecast(fig, ax,
                        data_1step.Sales,
                        median(samples_1step),
                        samples_1step,
                        data_1step.index,
                        credible_interval=75)


preds = []
for i in range(31):
    preds.append(np.median(samples_1step[:,i]))

print("RMSE: %0.4f" % mean_absolute_percentage_error(
                        data_1step['Sales'],
                        preds
                        ))

plt.show()