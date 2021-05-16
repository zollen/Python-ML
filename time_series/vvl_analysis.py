'''
Created on May 14, 2021

@author: zollen
'''

import pandas_datareader.data as web
from datetime import datetime, timedelta
import sys
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error
import numpy as np
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
WEEKS_FOR_ANALYSIS = 24
TEST_SIZE = int(WEEKS_FOR_ANALYSIS * 7 * 0.1)
TICKER = 'VVL.TO'


if SHOW_GRAPHS:
    plt.figure(figsize=(10, 10))        



def plotSeries(series, title, pos):
    plt.subplot(5, 1, pos)
    plt.plot(series.index, series['Price'], color='red')
    plt.ylabel(title, fontsize=10)

    

def display(data):
    print(data)
    



def get_stock():
    start_date, end_date = datetime.now().date() - timedelta(weeks=WEEKS_FOR_ANALYSIS), datetime.now().date()
    prices = web.DataReader(TICKER, 'yahoo', start=start_date, end=end_date).Close
    prices.index = [d.date() for d in prices.index]
    
    prices = pd.DataFrame({'Date' : prices.index, 'Price' : prices.values})
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = prices.set_index('Date')
    prices = prices.asfreq(pd.infer_freq(prices.index), method="pad")
         
    if SHOW_GRAPHS:
        plotSeries(prices, "YYL.TO", 1)
     
    return prices
    
getStockNode = node(get_stock, inputs=None, outputs="trade_data")

'''
                 coef    std err          t      P>|t|       95.0% Conf. Int.
-----------------------------------------------------------------------------
alpha[7]       0.1569  4.192e-02      3.743  1.819e-04    [7.475e-02,  0.239]
RMSE: 1.2973
'''
def normalize1(data):
    # Normalize: put the data with averge of zero
    avg, dev = data['Price'].mean(), data['Price'].std()
    out = (data['Price'] - avg) / dev
    data['Price'] = out
    data['Price'] = data['Price'].astype('float64')
    
    if SHOW_GRAPHS:
        plotSeries(data, "Normalize", 2)
    
    # First Differences
    data['Price'] = data['Price'].diff()
    data = data.dropna()
    data['Price'] = data['Price'].astype('float64')
    
    if SHOW_GRAPHS:
        plotSeries(data, "First Difference", 3)
    
    data['Month'] = data.index.map(lambda d: datetime.strptime(d.strftime("%Y-%m-%d"), "%Y-%m-%d").month)
   
    # Clean Volatitiy
    std = data.groupby(data['Month']).std()
    std['Price'].fillna(sys.maxsize, inplace = True)
    data_std = data.index.map(lambda d : std.loc[d.month])
    data['Price'] = data['Price'] / data_std
    data['Price'] = data['Price'].astype('float64')
    
    if SHOW_GRAPHS:
        plotSeries(data, "Remove Volaitity", 4)
        
    data.drop(columns = ['Month'], inplace = True)   
    data['Day'] = data.index.map(lambda d: datetime.strptime(d.strftime("%Y-%m-%d"), "%Y-%m-%d").day)
    
    
    # Clean Seasonality 
    days_avgs = data.groupby(data['Day']).mean()
    data_avg = data.index.map(lambda d: days_avgs.loc[d.day])
    data['Price'] = data['Price'] - data_avg
    data['Price'] = data['Price'].astype('float64')
    
    if SHOW_GRAPHS:
        plotSeries(data, "Remove Seasonality", 5)
    
    
    data.drop(columns = ['Day'], inplace = True)
    
    return data

normalize1Node = node(normalize1, inputs="trade_data", outputs="normalize_data1")

'''
                 coef    std err          t      P>|t|       95.0% Conf. Int.
-----------------------------------------------------------------------------
alpha[7]       0.5724      0.175      3.275  1.058e-03      [  0.230,  0.915]
RMSE: 1.4538
'''
def normalize2(data):
    data['Price'] = 100 * data['Price'].pct_change()
    data.dropna(inplace=True)
    data['Price'] = data['Price'].astype('float64')
    return data

normalize2Node = node(normalize2, inputs="trade_data", outputs="normalize_data2")
    

def train_archmodel(data):

    test_data = data.iloc[len(data) - TEST_SIZE:]
    
    rolling_predictions = []
    for i in range(TEST_SIZE):
        train_data = data.iloc[:-(TEST_SIZE-i)]
        model = arch_model(train_data, p=7, q=0)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
        
    print("RMSE: %0.4f" % np.sqrt(mean_squared_error(test_data['Price'], rolling_predictions)))
    
    plt.figure(figsize=(10,4))
    plt.plot(data)
    plt.plot(test_data.index, rolling_predictions)
    plt.legend(('Data', 'Predictions'), fontsize=16)
    plt.title('VVL.TO', fontsize=20)
    plt.ylabel('Price', fontsize=16)

     
   

archModel1Node = node(train_archmodel, inputs="normalize_data1", outputs=None)
archModel2Node = node(train_archmodel, inputs="normalize_data2", outputs=None)

# Create a data source
data_catalog = DataCatalog({"trade_data": MemoryDataSet()})

# Assign "nodes" to a "pipeline"
pipeline = Pipeline([ 
                        getStockNode,
                        normalize1Node,
                        normalize2Node,
                        archModel1Node,
                        archModel2Node
                    ])

# Create a "runner" to run the "pipeline"
runner = SequentialRunner()

# Execute a pipeline
runner.run(pipeline, data_catalog)


plt.show()