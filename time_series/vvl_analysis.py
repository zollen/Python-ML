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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
TEST_SIZE = (WEEKS_FOR_ANALYSIS * 7 * 0.1)
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
     
    plotSeries(prices, "YYL.TO", 1)
     
    return prices
    
getStockNode = node(get_stock, inputs=None, outputs="trade_data")


def normalize(data):
    # Normalize: put the data with averge of zero
    avg, dev = data['Price'].mean(), data['Price'].std()
    out = (data['Price'] - avg) / dev
    data['Price'] = out
    data['Price'] = data['Price'].astype('float64')
    
    plotSeries(data, "Normalize", 2)
    
    # First Differences
    data['Price'] = data['Price'].diff()
    data = data.dropna()
    data['Price'] = data['Price'].astype('float64')
    
    plotSeries(data, "First Difference", 3)
    
    data['Month'] = data.index.map(lambda d: datetime.strptime(d.strftime("%Y-%m-%d"), "%Y-%m-%d").month)
   
    # Clean Volatitiy
    std = data.groupby(data['Month']).std()
    std['Price'].fillna(sys.maxsize, inplace = True)
    data_std = data.index.map(lambda d : std.loc[d.month])
    data['Price'] = data['Price'] / data_std
    data['Price'] = data['Price'].astype('float64')
    
    plotSeries(data, "Remove Volaitity", 4)
        
    data.drop(columns = ['Month'], inplace = True)   
    data['Day'] = data.index.map(lambda d: datetime.strptime(d.strftime("%Y-%m-%d"), "%Y-%m-%d").day)
    
    
    # Clean Seasonality 
    days_avgs = data.groupby(data['Day']).mean()
    data_avg = data.index.map(lambda d: days_avgs.loc[d.day])
    data['Price'] = data['Price'] - data_avg
    data['Price'] = data['Price'].astype('float64')
    
    plotSeries(data, "Remove Seasonality", 5)
    
    
    data.drop(columns = ['Day'], inplace = True)
    
    return data

normalizeNode = node(normalize, inputs="trade_data", outputs="normalize_data")


def analysis_stock(orig_data, normalize_data):
    
    plot_pacf(np.array(orig_data['Price'])**2, title="PACF(orignal)")
    plot_pacf(np.array(normalize_data['Price'])**2, title="PACF(normalize)")
    plt.show()
    
analysisNode = node(analysis_stock, inputs=["trade_data", "normalize_data"], outputs=None)


def train_archmodel(orig_data, normalize_data):
    
    model = arch_model(orig_data, p=12, q=12)
    model_fit = model.fit()
    
    print(model_fit.summary())

archModelNode = node(train_archmodel, inputs=["trade_data", "normalize_data"], outputs=None)

# Create a data source
data_catalog = DataCatalog({"trade_data": MemoryDataSet()})

# Assign "nodes" to a "pipeline"
pipeline = Pipeline([ 
                        getStockNode,
                        normalizeNode,
                  #      analysisNode,
                        archModelNode
                    ])

# Create a "runner" to run the "pipeline"
runner = SequentialRunner()

# Execute a pipeline
runner.run(pipeline, data_catalog)

if SHOW_GRAPHS:
    plt.show()