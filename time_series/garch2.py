'''
Created on Apr. 28, 2021

@author: zollen
@desc GARCH Stock Volatity Forcasting
@url: https://www.youtube.com/watch?v=NKHQiN-08S8&list=PLvcbYUQ5t0UHOLnBzl46_Q6QKtFgfMGc3&index=34&ab_channel=ritvikmath
'''

import pandas_datareader.data as web
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

'''
Disney Stock
'''
start = datetime(2015, 1, 1)
end = datetime(2020, 6, 10)

dis = web.DataReader('DIS', 'yahoo', start=start, end=end)

returns = 100 * dis.Close.pct_change().dropna()

if False:
    plt.figure(figsize=(10,4))
    plt.plot(returns)
    plt.ylabel('Pct Return', fontsize=16)
    plt.title('DIS Returns', fontsize=20)
    plt.show()
    
'''
PACF
Partial Autocorrelation Function
'''
if False:
    # PACF graph shows lag 1, 2, 3 are strong, then they start to shutdown
    plot_pacf(returns**2)
    plt.show()    
    
'''
GARCH(3, 3)
                 coef    std err          t      P>|t|     95.0% Conf. Int.
---------------------------------------------------------------------------
omega          0.5417      0.189      2.860  4.231e-03    [  0.171,  0.913]
alpha[1]       0.0684  3.947e-02      1.733  8.314e-02 [-8.968e-03,  0.146]
alpha[2]       0.2032  9.867e-02      2.060  3.943e-02  [9.836e-03,  0.397]
alpha[3]       0.3177      0.152      2.096  3.604e-02  [2.068e-02,  0.615]
beta[1]        0.0000      0.162      0.000      1.000    [ -0.318,  0.318]
beta[2]    5.0201e-15  9.634e-02  5.211e-14      1.000    [ -0.189,  0.189]
beta[3]        0.2296      0.168      1.370      0.171 [-9.880e-02,  0.558]
===========================================================================
'''
model = arch_model(returns, p=3, q=3)
model_fit = model.fit()
print(model_fit.summary())

'''
Above summary shows that the volitity lags aren't that sigificant. So let's remove them

Try GARCH(3,0) = ARCH(3)
                 coef    std err          t      P>|t|     95.0% Conf. Int.
---------------------------------------------------------------------------
omega          0.8619      0.138      6.230  4.678e-10    [  0.591,  1.133]
alpha[1]       0.0886  4.530e-02      1.955  5.052e-02 [-2.022e-04,  0.177]
alpha[2]       0.2621  9.123e-02      2.873  4.060e-03  [8.334e-02,  0.441]
alpha[3]       0.3558      0.169      2.102  3.559e-02  [2.397e-02,  0.688]
===========================================================================
'''
model = arch_model(returns, p=3, q=0)
model_fit = model.fit()
print(model_fit.summary())

rolling_predictions = []
test_size = 365

for i in range(test_size):
    train = returns[:-(test_size-i)]
    model = arch_model(train, p=3, q=0)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
    
rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-365:])

if False:
    plt.figure(figsize=(10,4))
    true, = plt.plot(returns[-365:])
    preds, = plt.plot(rolling_predictions)
    plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
    plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)
    plt.show()
    
'''
S&P 500
'''
start = datetime(2000, 1, 1)
end = datetime(2020, 6, 10)

spy = web.DataReader('SPY', 'yahoo', start=start, end=end)

returns = 100 * spy.Close.pct_change().dropna()

if False:
    plt.figure(figsize=(10,4))
    plt.plot(returns)
    plt.ylabel('Pct Return', fontsize=16)
    plt.title('SPY Returns', fontsize=20)
    plt.show()
    
if False:
    # Lag 1, 2 are significant
    plot_pacf(returns**2)
    plt.show()
    
'''
GARCH(2,2)    
'''
model = arch_model(returns, p=2, q=2)
model_fit = model.fit()
print(model_fit.summary())

'''
Rolling Forecast
'''
rolling_predictions = []
test_size = 365*5

for i in range(test_size):
    train = returns[:-(test_size-i)]
    model = arch_model(train, p=2, q=2)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
    
rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-365*5:])

if False:
    plt.figure(figsize=(10,4))
    true, = plt.plot(returns[-365*5:])
    preds, = plt.plot(rolling_predictions)
    plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
    plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)
    plt.show()
    
'''
How to use the model
'''
train = returns
model = arch_model(train, p=2, q=2)
model_fit = model.fit(disp='off')

pred = model_fit.forecast(horizon=7)
future_dates = [returns.index[-1] + timedelta(days=i) for i in range(1,8)]
pred = pd.Series(np.sqrt(pred.variance.values[-1,:]), index=future_dates)

if False:
    plt.figure(figsize=(10,4))
    plt.plot(pred)
    plt.title('Volatility Prediction - Next 7 Days', fontsize=20)
    plt.show()