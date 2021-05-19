'''
Created on May 18, 2021

@author: zollen
'''

import pandas_datareader.data as web
from datetime import datetime, timedelta
import sys
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
PREDICTION_SIZE = 14
TEST_SIZE = int(WEEKS_FOR_ANALYSIS * 7 * 0.1)
TICKER = 'VVL.TO'

  
def get_stock():
    start_date, end_date = datetime.now().date() - timedelta(weeks=WEEKS_FOR_ANALYSIS), datetime.now().date()
    vvl = web.DataReader(TICKER, 'yahoo', start=start_date, end=end_date).Close
    dwi = web.DataReader('^DJI', 'yahoo', start=start_date, end=end_date).Close
    spi = web.DataReader('^GSPTSE', 'yahoo', start=start_date, end=end_date).Close
    vvl.index = [d.date() for d in vvl.index]
      
    prices = pd.DataFrame({'Date' : vvl.index, 
                           'VVL.TO' : vvl.values, 
                           'DOW': dwi.values,
                           'TSX': spi.values })
    
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = prices.set_index('Date')
    prices = prices.asfreq(pd.infer_freq(prices.index), method="pad")
    prices['VVL.TO'] = prices['VVL.TO'].astype('float64')
    prices['DOW'] = prices['DOW'].astype('float64')
    prices['TSX'] = prices['TSX'].astype('float64')
   
    return prices

getStockNode = node(get_stock, inputs=None, outputs="trade_data")


def normalize(data):
    
    '''
    Let's normalize data
    '''
    avgs = data.mean()
    devs = data.std()

    data = (data - avgs) / devs

    '''
    Take first difference to remove the trend    
    '''
    data['VVL.TO'] = data['VVL.TO'].diff()
    data['VVL.TO'] = data['VVL.TO'].astype('float64')
    
    data['DOW'] = data['DOW'].diff()
    data['DOW'] = data['DOW'].astype('float64')
    
    data['TSX'] = data['TSX'].diff()
    data['TSX'] = data['TSX'].astype('float64')
    
    data.dropna(inplace=True)
    
    '''
    Remove Increasing Volatity
    '''
    data['Month'] = data.index.map(lambda d: datetime.strptime(d.strftime("%Y-%m-%d"), "%Y-%m-%d").month)
    monthly_volatity = data.groupby(data.index.month)['VVL.TO', 'DOW', 'TSX'].std()
    sample = pd.DataFrame({})
    sample['VVL.TO'] = data.index.map(lambda d: monthly_volatity.loc[d.month, 'VVL.TO'])
    sample['DOW'] = data.index.map(lambda d: monthly_volatity.loc[d.month, 'DOW'])
    sample['TSX'] = data.index.map(lambda d: monthly_volatity.loc[d.month, 'TSX'])
    data.drop(columns = ['Month'], inplace = True)
    
    data[['VVL.TO', 'DOW', 'TSX']] = data.values / sample.values
    
    
    '''
    Remove Seasonality  
    '''
    data['Day'] = data.index.map(lambda d: datetime.strptime(d.strftime("%Y-%m-%d"), "%Y-%m-%d").day)
    days_avgs = data.groupby(data['Day']).mean()
    sample = pd.DataFrame({})
    sample['VVL.TO'] = data.index.map(lambda d: days_avgs.loc[d.day, 'VVL.TO'])
    sample['DOW'] = data.index.map(lambda d: days_avgs.loc[d.day, 'DOW'])
    sample['TSX'] = data.index.map(lambda d: days_avgs.loc[d.day, 'TSX'])
    data.drop(columns = ['Day'], inplace = True)
      
    data[['VVL.TO', 'DOW', 'TSX']] = data.values - sample.values
    
    return data

normalizeNode = node(normalize, inputs="trade_data", outputs="normalize_data")


def train_VAR(trade_data, normalize_data):
    pass


# Create a data source
data_catalog = DataCatalog({"trade_data": MemoryDataSet()})

# Assign "nodes" to a "pipeline"
pipeline = Pipeline([ 
                        getStockNode,
                        normalizeNode
                    ])

# Create a "runner" to run the "pipeline"
runner = SequentialRunner()

# Execute a pipeline
runner.run(pipeline, data_catalog)


plt.show()