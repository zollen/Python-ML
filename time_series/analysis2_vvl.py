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
    '''
    Best AIC: 38.64968775676793 Best Order (1, 0, 3) Best Seasonal Order (0, 0, 0, 12)
    '''
    best_score = 999999
    best_param = None
    best_sparam = None
    p = q = range(0, 7)
    pdq = list(itertools.product(p, [0,1,2], q))

    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product([0,1,2], [0,1], [0,1,2]))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(data,
                            order=param,
                            seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
                results = mod.fit()
                if results.aic < best_score:
                    best_score = results.aic
                    best_param = param
                    best_sparam = param_seasonal
                print('SARIMAX{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    print()
    print()
    print("Best AIC: {} Best Order {} Best Seasonal Order {}".format(best_score, best_param, best_sparam))
     
   
optimizeNode = node(optimize, inputs="trade_data", outputs=None)


def train(data):
    
    '''
    Best AIC: 10.0 Best Order (1, 2, 2) Best Seasonal Order (0, 1, 1, 12)
    '''
    
    train_data = data.iloc[:len(data) - TEST_SIZE]
    test_data = data.iloc[len(data) - TEST_SIZE:]
    
    mod = SARIMAX(train_data,
            order=(1, 0, 3),
            seasonal_order=(0, 0, 0, 12),
            enforce_stationarity=False,
            enforce_invertibility=False)
    results = mod.fit()
    
    preds = results.predict(TEST_SIZE)
    
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
                    ])

# Create a "runner" to run the "pipeline"
runner = SequentialRunner()

# Execute a pipeline
runner.run(pipeline, data_catalog)


plt.show()

