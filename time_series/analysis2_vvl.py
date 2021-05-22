'''
Created on May 21, 2021

@author: zollen
'''

import pandas_datareader.data as web
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import arma_order_select_ic
from kedro.pipeline import node
from kedro.pipeline import Pipeline
from kedro.io import DataCatalog, MemoryDataSet
from kedro.runner import SequentialRunner
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

SHOW_GRAPHS = False
WEEKS_FOR_ANALYSIS = 24
PREDICTION_SIZE = 14
TEST_SIZE = int(WEEKS_FOR_ANALYSIS * 7 * 0.1)
TICKER = 'VVL.TO'


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC', regression="ctt")
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")    

def get_stock():
    start_date, end_date = datetime.now().date() - timedelta(weeks=WEEKS_FOR_ANALYSIS), datetime.now().date()
    prices = web.DataReader(TICKER, 'yahoo', start=start_date, end=end_date).Close
    prices.index = [d.date() for d in prices.index]
    
    prices = pd.DataFrame({'Date' : prices.index, 'Price' : prices.values})
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = prices.set_index('Date')
    prices = prices.asfreq(pd.infer_freq(prices.index), method="pad")
    prices['Price'] = prices['Price'].astype('float64')
     
    return prices

getStockNode = node(get_stock, inputs=None, outputs="trade_data")


def analysis(data):
     
    adfuller_test(data, name='Price')
    
    data['Price'] = data['Price'].diff()
    data.dropna(inplace=True)
    
    adfuller_test(data, name='Price')
    
    if False:
        _, ax = plt.subplots(2, 1, figsize = (10,8))
        plot_acf(data, lags = 24, ax = ax[0])
        plot_pacf(data, lags = 24, ax = ax[1])
        plt.show()
    
analysisNode = node(analysis, inputs="trade_data", outputs=None)
    

def optimize1(data):
    
    p = q = [0, 1, 2, 3, 4, 5, 6]
    pdq = list(itertools.product(p, [1,2], q))
    
    params = []
    params_s = []
    aics = []
    mses = []
    
    train_data = data.iloc[:len(data) - TEST_SIZE]
    test_data = data.iloc[len(data) - TEST_SIZE:]

    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product([0,1,2], [0,1,2], [0,1,2]))]
    
    total = len(pdq) * len(seasonal_pdq)
    current = 0
       
    for param in pdq:
        for param_seasonal in seasonal_pdq:
          
            try:
                current = current + 1 
                print("Progress: ", (current / total * 100), "%")
                print("settings: ", param, param_seasonal)
                mod = SARIMAX(train_data,
                            order=param,
                            seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
                results = mod.fit()
                
                pred = results.get_prediction(start = test_data.index[0],
                                              end = test_data.index[-1] + + timedelta(days = 1))
                
                params.append(param)
                params_s.append(param_seasonal)
                aics.append(results.aic)
                mses.append(mean_squared_error(test_data , pred.predicted_mean[1:]))  
                
            except:
                continue
            
            
    min_ind = aics.index(min(aics)) 
    bestparam = (params[min_ind], params_s[min_ind]) 
    print('best_param_aic:', bestparam, ' aic:', min(aics)) 
    min_ind = mses.index(min(mses)) 
    bestparam = (params[min_ind], params_s[min_ind]) 
    print('best_param_mse:', bestparam, ' mse:', min(mses))
   
optimize1Node = node(optimize1, inputs="trade_data", outputs=None)


def optimize2(data):
    # ARMA(p,q) = (0, 3) is the best.
    resDiff = arma_order_select_ic(data, max_ar=7, max_ma=7, ic=['aic','bic'], trend='nc')
    print('ARMA(p,q) =', resDiff['aic_min_order'], 'is the best.')

optimize2Node = node(optimize2, inputs="trade_data", outputs=None)
    


def train(data):
    
    train_data = data.iloc[:len(data) - TEST_SIZE]
    test_data = data.iloc[len(data) - TEST_SIZE:]
    
    mod = SARIMAX(train_data,
            order=(3, 1, 3),
            seasonal_order=(1, 0, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False)
    results = mod.fit()
    
    preds = results.forecast(TEST_SIZE)
    
    print("RMSE: %0.4f" % np.sqrt(mean_squared_error(test_data['Price'], preds)))
    
    if True:
        results.plot_diagnostics(figsize=(10,8))
        
    if True:
        plt.figure(figsize=(10,4))
        plt.plot(data)
        plt.plot(test_data.index, preds)
        plt.legend(('Data', 'Predictions'), fontsize=16)
        plt.title("Price vs Prediction", fontsize=20)
        plt.ylabel('Price', fontsize=16)
    
trainNode = node(train, inputs="trade_data", outputs=None)




# Create a data source
data_catalog = DataCatalog({"trade_data": MemoryDataSet()})

# Assign "nodes" to a "pipeline"
pipeline = Pipeline([ 
                        getStockNode,
                        analysisNode,
                     #   optimizeNode1.
                     #   optimizeNode12,
                     #   trainNode
                    ])

# Create a "runner" to run the "pipeline"
runner = SequentialRunner()

# Execute a pipeline
runner.run(pipeline, data_catalog)


plt.show()

