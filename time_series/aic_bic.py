'''
Created on May 26, 2021

@author: zollen
'''

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')



def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')

def perform_adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    
series = pd.read_csv('catfish.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series = series.asfreq(pd.infer_freq(series.index))
series = series.loc[datetime(2004,1,1):]
series = series.diff().diff().dropna()

perform_adf_test(series)

if False:
    plt.plot(series)
    plot_pacf(series, lags=10)


ar_orders = [1, 4, 6, 10]  
if True:
    plt.figure(figsize=(12,12))
    
    # let's test our best candidates from PACF plot
    # AR(1), AR(4), AR(6), AR(10)
    fitted_model_dict = {}
    
    for idx, ar_order in enumerate(ar_orders):
        
        #create AR(p) model
        ar_model = ARMA(series, order=(ar_order,0))
        ar_model_fit = ar_model.fit()
        fitted_model_dict[ar_order] = ar_model_fit
        plt.subplot(4,1,idx+1)
        plt.plot(series)
        plt.plot(ar_model_fit.fittedvalues)
        plt.title('AR(%s) Fit'%ar_order, fontsize=16)
    
    plt.tight_layout()


'''
Above graphes show that both AR(6) and AR(10) are the best and very  close.
Is AR(10) really worth the extra four parameters??
To determine which is better, let's use AIC and BIC
AIC and BIC show how strong the model fit the data
More params of course fit better, but more params tend to over-fitting

l - a log likelihood
k - a number of parameters
n - a number of samples used for fitting

lower AIC = higher log likelihood or less parameters
AIC = 2 * k - 2l 

lower BIC = higher log likelihood or less parameters or less samples used in fitting
BIC = ln(n) * k - 2l
'''

# AIC criteria
for ar_order in ar_orders:
    print('AIC for AR(%s): %s'%(ar_order, fitted_model_dict[ar_order].aic))

print("=================================================")
# BIC criteria
for ar_order in ar_orders:
    print('BIC for AR(%s): %s'%(ar_order, fitted_model_dict[ar_order].bic))

print("AR(6) has the lowerest of both AIC and BIC. Therefore AR(6) is the best")

plt.show()