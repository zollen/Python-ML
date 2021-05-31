'''
Created on May 18, 2021

@author: zollen
'''

import pandas_datareader.data as web
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
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
        
    perform_adf_test(trade_data['VVL.TO'])
    perform_adf_test(trade_data['DOW'])
    perform_adf_test(trade_data['TSX'])
    
    
    if False:
        # inconclusive
        _, a1 = plt.subplots(1, 1)
        a1.set_ylabel('NORMALIZE(VVL.TO)', fontsize=8)
        plot_pacf(trade_data['VVL.TO'], ax=a1, title="PACF Analysis of VVL.TO")

    
    

analysisNode = node(analysis_data, inputs=["trade_data"], outputs=None)



def optimize_model(trade_data):
     
    if True:  
        params = []
        aics = []
        mses = []
        
        p = [0, 1, 2, 3, 4, 5]
        q = [0, 1, 2, 3, 4, 5]
        pdq = list(itertools.product(p, [1], q))
        pdq.remove((0,1,0))
       
        for param in pdq:
        
            try:
                
                tscv = TimeSeriesSplit(n_splits=TIME_SPLITS_CV, max_train_size=TRAIN_SIZE, test_size=TEST_SIZE)
                aics_t = []
                mses_t = []
                for train_index, test_index in tscv.split(trade_data):
                    
                    X_train, X_test = trade_data.iloc[train_index], trade_data.iloc[test_index]
                    
                    model = SARIMAX(X_train['VVL.TO'],
                                    order=param,
                                    seasonal_order=(2, 1, 2, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                    results = model.fit() 
                    
                    pred = results.get_prediction(start = X_test.index[0],
                                                  end = X_test.index[-1])
    
                    aics_t.append(results.aic)
                    mses_t.append(mean_squared_error(X_test['VVL.TO'].iloc[:-1], pred.predicted_mean[1:])) 
                
                params.append(param)
                aics.append(np.sum(aics_t) / TIME_SPLITS_CV)
                mses.append(np.sum(mses_t) / TIME_SPLITS_CV)  
                
            except:
                continue
                
                
        min_ind = aics.index(min(aics)) 
        bestparam = params[min_ind]
        print('best_param_aic:', bestparam, ' aic:', min(aics)) 
        min_ind = mses.index(min(mses)) 
        bestparam = params[min_ind]
        print('best_param_mse:', bestparam, ' mse:', min(mses))
        
        '''
        best_param_aic: (2, 1, 2)  aic: 30.706256895869064
        best_param_mse: (2, 1, 2)  mse: 0.14321048365277564
        '''

optimizeNode = node(optimize_model, inputs=["trade_data"], outputs=None)



def test_model(trade_data):
    
    G_train = trade_data.iloc[-(TEST_SIZE)*2:]
    X_train = trade_data.iloc[-TRAIN_SIZE-TEST_SIZE:-TEST_SIZE]
    X_test = trade_data.iloc[-TEST_SIZE:]
    
    model = SARIMAX(X_train['VVL.TO'],
                    order=(2, 1, 2),
                    seasonal_order=(2, 1, 2, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit() 
    
    preds = results.get_prediction(start = X_test.index[0],
                                   end = X_test.index[-1] + timedelta(days = 1))
    
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
                        optimizeNode,
                    #   testNode
                    ])

# Create a "runner" to run the "pipeline"
runner = SequentialRunner()

# Execute a pipeline
runner.run(pipeline, data_catalog)


plt.show()