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
from statsmodels.tsa.statespace.sarimax import SARIMAX
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


def optimize(data):
    
    p = q = [0, 1, 2, 3, 4]
    pdq = list(itertools.product(p, [1,2,3], q))
    
    params = []
    params_s = []
    aics = []
    mses = []
    
    train_data = data.iloc[:len(data) - TEST_SIZE]
    test_data = data.iloc[len(data) - TEST_SIZE:]

    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product([0,1,2,3], [0,1], [0,1,2,3]))]
   
    for param in pdq:
        for param_seasonal in seasonal_pdq:
          
            try:
                mod = SARIMAX(train_data,
                            order=param,
                            seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
                results = mod.fit()
                
                pred = results.get_prediction(start = test_data.index[0],
                                              end = test_data.index[-1])
                
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
   
optimizeNode = node(optimize, inputs="trade_data", outputs=None)


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
                        optimizeNode
                     #   trainNode
                    ])

# Create a "runner" to run the "pipeline"
runner = SequentialRunner()

# Execute a pipeline
runner.run(pipeline, data_catalog)


plt.show()

