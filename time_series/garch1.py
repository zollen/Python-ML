'''
Created on Apr. 26, 2021

@author: zollen
'''

from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')


'''
GARCH(2,2) Model 
GARCH(lags, violatity)
----------------
a(t) = e(t) sqrt( w + φ1 a(t-1)^2 + φ2 a(t-2)^2 + γ1 b(t-1)^2 + γ2 b(t-2)^2 )
a(0), a(1) ~ N(0,1)
b0 = 1, b1 = 1
e(t) ~ N(0, 1)
'''

n = 1000
omega = 0.5
alpha_1 = 0.1
alpha_2 = 0.2

beta_1 = 0.3
beta_2 = 0.4

test_size = int(n*0.1)

series = [gauss(0,1), gauss(0,1)]
vols = [1, 1]

for _ in range(n):
    # above formula
    new_vol = np.sqrt(omega + alpha_1*series[-1]**2 + alpha_2*series[-2]**2 + beta_1*vols[-1]**2 + beta_2*vols[-2]**2)
    new_val = gauss(0,1) * new_vol
    
    vols.append(new_vol)
    series.append(new_val)
    
if False:
    plt.figure(figsize=(10,4))
    plt.plot(series)
    plt.plot(vols, color='red')
    plt.title("Data and Volatility", fontsize=20)
    plt.show()
    
'''
PACF plot
Because of the above formula, expect the lag 1 and 2 are high, then begin to drop off
'''
if False:
    plot_pacf(np.array(series)**2)
    plt.show()
    
train, test = series[:-test_size], series[-test_size:]
model = arch_model(train, p=2, q=2)
model_fit = model.fit()
'''
                 coef    std err          t      P>|t|     95.0% Conf. Int.
---------------------------------------------------------------------------
omega          0.8095      0.280      2.895  3.786e-03    [  0.262,  1.357]
alpha[1]       0.0780  3.648e-02      2.140  3.239e-02  [6.552e-03,  0.150]
alpha[2]       0.3136  5.464e-02      5.739  9.524e-09    [  0.206,  0.421]
beta[1]        0.1158  8.519e-02      1.359      0.174 [-5.119e-02,  0.283]
beta[2]        0.4926  8.072e-02      6.103  1.044e-09    [  0.334,  0.651]

all P values are less than 0.05. It makes sense they are all used to generate the data

omega is suppose to be 0.5, but it is 0.8095, much off.
alpha1 is suppose to be 0.1, but it is 0.078, close enough.
alpha2 is suppose to be 0.2, but it is 0.3136, little off, but not by much.
beta1 is suppose to be0.3, but it is 0.1158, much off.
beta2 is suppose to be 0.4, but it is 0.4926, close enough

'''
print(model_fit.summary())



predictions = model_fit.forecast(horizon=test_size)

if False:
    plt.figure(figsize=(10,4))
    true, = plt.plot(vols[-test_size:])
    preds, = plt.plot(np.sqrt(predictions.variance.values[-1, :]))
    plt.title('Volatility Prediction', fontsize=20)
    plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
    plt.show()
    

predictions_long_term = model_fit.forecast(horizon=1000)

if False:
    plt.figure(figsize=(10,4))
    true, = plt.plot(vols[-test_size:])
    preds, = plt.plot(np.sqrt(predictions_long_term.variance.values[-1, :]))
    plt.title('Long Term Volatility Prediction', fontsize=20)
    plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
    plt.show()
    
'''
Rolling Forecast Origin
'''

rolling_predictions = []
for i in range(test_size):
    train = series[:-(test_size-i)]
    model = arch_model(train, p=2, q=2)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
    
if True:
    # Extermely accurate for prediction one period at a time
    plt.figure(figsize=(10,4))
    true, = plt.plot(vols[-test_size:])
    preds, = plt.plot(rolling_predictions)
    plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
    plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
    plt.show()