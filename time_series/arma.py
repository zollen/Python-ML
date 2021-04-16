'''
Created on Apr. 16, 2021

@author: zollen
@url: https://www.youtube.com/watch?v=xg2-9DhE5vc&list=PLvcbYUQ5t0UHOLnBzl46_Q6QKtFgfMGc3&index=30&ab_channel=ritvikmath
'''

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA
from time import time
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

register_matplotlib_converters()

def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')

catfish_sales = pd.read_csv('catfish.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

catfish_sales = catfish_sales.asfreq(pd.infer_freq(catfish_sales.index))

start_date = datetime(2000, 1, 1)
end_date = datetime(2004, 1, 1)
lim_catfish_sales = catfish_sales[start_date:end_date]

if False:
    plt.figure(figsize=(10,4))
    plt.plot(lim_catfish_sales)
    plt.title('CatFish Sales in 1000s of Pounds', fontsize=20)
    plt.ylabel('Sales', fontsize=16)
    for year in range(start_date.year, end_date.year):
        plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
    plt.axhline(lim_catfish_sales.mean(), color='r', alpha=0.2, linestyle='--')
    plt.show()
    
first_diff = lim_catfish_sales.diff()[1:]

if False:
    plt.figure(figsize=(10,4))
    plt.plot(first_diff)
    plt.title('First Difference of Catfish Sales', fontsize=20)
    plt.ylabel('Sales', fontsize=16)
    for year in range(start_date.year, end_date.year):
        plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
    plt.axhline(first_diff.mean(), color='r', alpha=0.2, linestyle='--')
    plt.show()

'''
Based on ACF, we should start with a MA(1) process for now
'''    
if False:
    acf_vals = acf(first_diff)
    num_lags = 20
    plt.bar(range(num_lags), acf_vals[:num_lags])
    plt.show()

'''
Based on PACF, we should start with a AR(4) process
'''
if False:
    num_lags = 15
    pacf_vals = pacf(first_diff, nlags=num_lags)
    plt.bar(range(num_lags), pacf_vals[:num_lags])
    plt.show()
    
train_end = datetime(2003, 7, 1)
test_end = datetime(2004, 1, 1)
train_data = first_diff[:train_end]
test_data = first_diff[train_end + timedelta(days=1):test_end]

model = ARMA(train_data, order=(4, 1))
start = time()
model_fit = model.fit()
end = time()
print("Model Fitting Time: ", end - start)
print(model_fit.summary())

'''
                 coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------
const          37.3376    129.751      0.288      0.774    -216.970     291.645
ar.L1.Total    -0.8666      0.185     -4.692      0.000      -1.229      -0.505
ar.L2.Total    -0.4236      0.166     -2.547      0.011      -0.750      -0.098
ar.L3.Total    -0.5584      0.156     -3.579      0.000      -0.864      -0.253
ar.L4.Total    -0.6144      0.126     -4.894      0.000      -0.861      -0.368
ma.L1.Total     0.5197      0.219      2.370      0.018       0.090       0.950
The summary shows:
1. The 4 coeffs of the AR(4) are all negatively correlates with the prediction
2. The coeff of the MA(1) is postively correlates with the prediction
3. The constant has a P value way above 0.05, so we should exclude the constant

Here is the final model
y(t) = -0.8666 y(t-1) - 0.4236 y(t-2) - 0.5584 y(t-3) - 0.6144 y(t-4) + 0.5197 e(t-1)
'''

pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]

predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
residuals = test_data - predictions

if False:
    # The residuals shows that there may be a pattern and there is something else we did not quite capture
    plt.figure(figsize=(10,4))
    plt.plot(residuals)
    plt.title("Residulas from AR Model", fontsize = 20)
    plt.ylabel('Error', fontsize = 16)
    plt.axhline(0, color='r', linestyle='--', alpha=0.2)
    plt.show()

if False:
    plt.figure(figsize=(10,4))
    plt.plot(test_data)
    plt.plot(predictions)
    plt.legend(('Data', 'Predictions'), fontsize = 16)
    plt.title('First Difference of Catfish Sales', fontsize = 20)
    plt.ylabel('Sales', fontsize = 16)
    plt.show()


print('Mean Absolute Percent Error: ', round(np.mean(abs(residuals/test_data)), 4))
print('Root Mean Squared Error: ', np.sqrt(np.mean(residuals**2)))
