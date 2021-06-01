'''
Created on May 18, 2021

@author: zollen
'''

import pandas_datareader.data as web
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from kedro.pipeline import node
from kedro.pipeline import Pipeline
from kedro.io import DataCatalog, MemoryDataSet
from kedro.runner import SequentialRunner
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

SHOW_GRAPHS = False
WEEKS_FOR_ANALYSIS = 72
TRAIN_SIZE=2120
TEST_SIZE = 14
TIME_SPLITS_CV = 6
TICKER = 'VVL.TO'


def perform_adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
  
def get_stock():
    start_date, end_date = datetime.now().date() - timedelta(weeks=WEEKS_FOR_ANALYSIS), datetime.now().date()
    vvl = web.DataReader(TICKER, 'yahoo', start=start_date, end=end_date).Close
    dwi = web.DataReader('^DJI', 'yahoo', start=start_date, end=end_date).Close
    spi = web.DataReader('^GSPTSE', 'yahoo', start=start_date, end=end_date).Close
    vvl.index = [d.date() for d in vvl.index]
    
    prices = pd.DataFrame({'Date' : vvl.index, 
                           'VVL.TO' : vvl.values, 
                           'DOW': dwi[vvl.index].values,
                           'TSX': spi[vvl.index].values })
    
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = prices.set_index('Date')
    prices = prices.asfreq(pd.infer_freq(prices.index), method="pad")
    prices['VVL.TO'] = prices['VVL.TO'].astype('float64')
    prices['DOW'] = prices['DOW'].astype('float64')
    prices['TSX'] = prices['TSX'].astype('float64')
    
    prices['DOW'] = prices['DOW'].fillna(method='ffill', axis=0)
    
  
    return prices

getStockNode = node(get_stock, inputs=None, outputs="trade_data")


def analysis_data(trade_data):
    
    if False:
        _, (a1, a2, a3, a4, a5) = plt.subplots(5, 1,figsize=(15,8))
        a1.plot(trade_data['VVL.TO'])
        a1.set_ylabel('VVL.TO', fontsize=8)
        a2.plot(trade_data['DOW'])
        a2.set_ylabel('DOW', fontsize=8)
        a3.plot(trade_data['TSX'])
        a3.set_ylabel('TSX', fontsize=8)
        
        print(trade_data[['VVL.TO', 'DOW', 'TSX']].corr())
        
        sb.regplot(x="VVL.TO", y="DOW", data=trade_data, 
           marker='.', fit_reg = False, scatter_kws = {'alpha' : 0.8}, ax=a4)
        sb.regplot(x="VVL.TO", y="TSX", data=trade_data, 
           marker='.', fit_reg = False, scatter_kws = {'alpha' : 0.8}, ax=a5)
        
    trade_data['VVL.TO'] = trade_data['VVL.TO'].diff()
    trade_data['DOW'] = trade_data['DOW'].diff()
    trade_data['TSX'] = trade_data['TSX'].diff()
    trade_data.dropna(inplace = True)
     
    if False:   
        perform_adf_test(trade_data['VVL.TO'])
        perform_adf_test(trade_data['DOW'])
        perform_adf_test(trade_data['TSX'])
    
    
    if False:
        # inconclusive
        _, a1 = plt.subplots(1, 1)
        a1.set_ylabel('NORMALIZE(VVL.TO)', fontsize=8)
        plot_pacf(trade_data['VVL.TO'], ax=a1, title="PACF Analysis of VVL.TO")
        
    if False:
        stl = STL(trade_data['VVL.TO'])
        result = stl.fit()
        
        seasonal, trend, resid = result.seasonal, result.trend, result.resid
        
        plt.figure(figsize=(8,6))
        
        plt.subplot(4,1,1)
        plt.plot(trade_data['VVL.TO'])
        plt.title('Original Series', fontsize=16)
        
        plt.subplot(4,1,2)
        plt.plot(trend)
        plt.title('Trend', fontsize=16)
        
        plt.subplot(4,1,3)
        plt.plot(seasonal)
        plt.title('Seasonal', fontsize=16)
        
        plt.subplot(4,1,4)
        plt.plot(resid)
        plt.title('Residual', fontsize=16)
        
        plt.tight_layout()

    
    

analysisNode = node(analysis_data, inputs=["trade_data"], outputs=None)



def test_model(trade_data):
    '''
    (2, 1, 2)(2, 1, 2, 3): 0.0785
    (2, 1, 2)(2, 1, 2, 4): 0.1442
    (3, 1, 3)(2, 1, 2, 4): 0.1391
    (3, 1, 3)(1, 1, 1, 4): 0.1391
    '''
    
    G_train = trade_data.iloc[-(TEST_SIZE)*2:]
    X_train = trade_data.iloc[-TRAIN_SIZE-TEST_SIZE:-TEST_SIZE]
    X_test = trade_data.iloc[-TEST_SIZE:]
    
    model = SARIMAX(X_train['VVL.TO'],
                    order=(2, 1, 2),
                    seasonal_order=(3, 1, 3, 4),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit() 
    
    preds = results.get_prediction(start = X_test.index[0],
                                   end = X_test.index[-1] + timedelta(days = 1))
    
    print("RMSE: %0.4f" % mean_squared_error(X_test['VVL.TO'], preds.predicted_mean[1:]))
    
    if True:
        plt.figure(figsize=(10,4))
        plt.plot(G_train['VVL.TO'])
        plt.plot(X_test.index, preds.predicted_mean[1:])
        plt.legend(('Data', 'Predictions'), fontsize=16)
        plt.title("Price vs Prediction", fontsize=20)
        plt.ylabel('Price', fontsize=16) 
                
    
    

testNode = node(test_model, inputs=["trade_data"], outputs=None)

# Create a data source
data_catalog = DataCatalog({"trade_data": MemoryDataSet()})

# Assign "nodes" to a "pipeline"
pipeline = Pipeline([ 
                        getStockNode,
                        analysisNode,
                        testNode
                    ])

# Create a "runner" to run the "pipeline"
runner = SequentialRunner()

# Execute a pipeline
runner.run(pipeline, data_catalog)


plt.show()