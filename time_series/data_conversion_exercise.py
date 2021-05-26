'''
Created on May 26, 2021

@author: zollen
'''

import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')


def perform_adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    

ts = pd.read_csv('original_series.csv')
ts.index = np.arange(1,len(ts)+1)

'''
The graph shows there is a log base growth
'''
if False:
    plt.figure(figsize=(10,4))
    plt.plot(ts)
    
    plt.xticks(np.arange(0,78,6), fontsize=14)
    plt.xlabel('Hours Since Published', fontsize=16)
    
    plt.yticks(np.arange(0,50000,10000), fontsize=14)
    plt.ylabel('Views', fontsize=16)
    plt.show()
    
'''
Orgianl Series: v(t)
Normalize: (v(t) -> n(t)): (v(t) - mean) / std
Exponentiate (n(t) -> e(t)): e(t) = e^(n(t))   <-- convert exponenal into straight line
First Difference(e(t) -> d(t)): d(t) = e(t) - e(t-1)

Final Equation for Conversion
d(t) = e^((v(t) - mean)/std) - e^((v(t-1) - mean)/std)
'''
    
'''
Normalize
'''  
mu = np.mean(ts).iloc[0]
sigma = np.std(ts).iloc[0]

norm_ts = (ts - mu) / sigma

if False:
    plt.figure(figsize=(10,4))
    plt.plot(norm_ts)
    
    plt.xticks(np.arange(0,78,6), fontsize=14)
    plt.xlabel('Hours Since Published', fontsize=16)
    
    plt.yticks(np.arange(-3,2), fontsize=14)
    plt.ylabel('Norm. Views', fontsize=16)
    
    plt.axhline(0, color='k', linestyle='--')
    plt.show()
    
'''
Exponentiate
'''
exp_ts = np.exp(norm_ts)

if False:
    plt.figure(figsize=(10,4))
    plt.plot(exp_ts)
    
    plt.xticks(np.arange(0,78,6), fontsize=14)
    plt.xlabel('Hours Since Published', fontsize=16)
    
    plt.yticks(np.arange(0,3.5,.5), fontsize=14)
    plt.ylabel('Exp. Norm. Views', fontsize=16)
    plt.show()
    

perform_adf_test(exp_ts)

'''
First Difference 
'''
diff_ts = exp_ts.diff().dropna()

if False:
    plt.figure(figsize=(10,4))
    plt.plot(diff_ts)
    
    plt.xticks(np.arange(0,78,6), fontsize=14)
    plt.xlabel('Hours Since Published', fontsize=16)
    
    plt.yticks(np.arange(-0.2,0.3,.1), fontsize=14)
    plt.ylabel('First Diff. \nExp. Norm. Views', fontsize=16)
    plt.show()
    

perform_adf_test(diff_ts)

if False:
    # PACF - lags 1, 2, 4, 5
    # ACF  - lags 1
    plot_pacf(diff_ts)
    plot_acf(diff_ts)
    plt.show()
    
model = ARMA(diff_ts, order=(4,1))
results = model.fit()

prediction_info = results.forecast(3)

predictions = prediction_info[0]
lower_bound = prediction_info[2][:,0]
upper_bound = prediction_info[2][:,1]

if False:
    plt.figure(figsize=(10,4))
    plt.plot(diff_ts)
    
    plt.xticks(np.arange(0,78,6), fontsize=14)
    plt.xlabel('Hours Since Published', fontsize=16)
    
    plt.yticks(np.arange(-0.2,0.3,.1), fontsize=14)
    plt.ylabel('First Diff. \nExp. Norm. Views', fontsize=16)
    
    plt.plot(np.arange(len(ts)+1, len(ts)+4), predictions, color='g')
    plt.fill_between(np.arange(len(ts)+1, len(ts)+4), lower_bound, upper_bound, color='g', alpha=0.1)
    plt.show()
    
'''
Undo Transformations
'''
def undo_transformations(predictions, series, mu, sigma):
    first_pred = sigma*np.log(predictions[0] + np.exp((series.iloc[-1]-mu)/sigma)) + mu
    orig_predictions = [first_pred]
    
    for i in range(len(predictions[1:])):
        next_pred = sigma*np.log(predictions[i+1] + np.exp((orig_predictions[-1]-mu)/sigma)) + mu
        orig_predictions.append(next_pred)
    
    return np.array(orig_predictions).flatten()

orig_preds = undo_transformations(predictions, ts, mu, sigma)
orig_lower_bound = undo_transformations(lower_bound, ts, mu, sigma)
orig_upper_bound = undo_transformations(upper_bound, ts, mu, sigma)

if True:
    plt.figure(figsize=(10,4))
    plt.plot(ts)
    
    plt.xticks(np.arange(0,78,6), fontsize=14)
    plt.xlabel('Hours Since Published', fontsize=16)
    
    plt.yticks(np.arange(0,50000,10000), fontsize=14)
    plt.ylabel('Views', fontsize=16)
    
    plt.plot(np.arange(len(ts)+1, len(ts)+4), orig_preds, color='g')
    plt.fill_between(np.arange(len(ts)+1, len(ts)+4), orig_lower_bound, orig_upper_bound, color='g', alpha=0.1)
    
    '''
    Good example of customize xtick,xlim and ytick,ylim
    '''
    plt.figure(figsize=(10,4))
    plt.plot(ts)
    
    plt.xticks(np.arange(0,78), fontsize=14)
    plt.xlabel('Hours Since Published', fontsize=16)
    
    plt.yticks(np.arange(40000,46000,1000), fontsize=14)
    plt.ylabel('Views', fontsize=16)
    
    plt.plot(np.arange(len(ts)+1, len(ts)+4), orig_preds, color='g')
    plt.fill_between(np.arange(len(ts)+1, len(ts)+4), orig_lower_bound, orig_upper_bound, color='g', alpha=0.1)
    plt.xlim(64,76)
    plt.ylim(40000, 45000)
    
    plt.show()
    
