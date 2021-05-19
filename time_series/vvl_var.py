'''
Created on May 18, 2021

@author: zollen
'''

import pandas_datareader.data as web
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
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


def analysis_data(trade_data, normalize_data):
    if True:
        fig, (a1, a2, a3, a4, a5, a6) = plt.subplots(6, 1)
        fig.set_size_inches(10, 10)
        a1.set_ylabel('TRADE(WWL.TO)', fontsize=10)
        a2.set_ylabel('NORMALIZE(WWL.TO)', fontsize=10)
        a3.set_ylabel('TRADE(DOW)', fontsize=10)
        a4.set_ylabel('NORMALIZE(DOW)', fontsize=10)
        a5.set_ylabel('TRADE(TSX)', fontsize=10)
        a6.set_ylabel('NORMALIZE(TSX)', fontsize=10)
        plot_pacf(trade_data['VVL.TO'], ax=a1, title="PACF Analysis of VVL.TO")
        plot_pacf(normalize_data['VVL.TO'], ax=a2, title=None)
        plot_pacf(trade_data['DOW'], ax=a3, title=None)
        plot_pacf(normalize_data['DOW'], ax=a4, title=None)
        plot_pacf(trade_data['TSX'], ax=a5, title=None)
        plot_pacf(normalize_data['TSX'], ax=a6, title=None)

analysisNode = node(analysis_data, inputs=["trade_data", "normalize_data"], outputs=None)

def train_val(trade_data, normalize_data):
    '''
    Results for equation VVL.TO
                    coefficient       std. error           t-stat            prob
    -----------------------------------------------------------------------------
    L1.DOW            -0.504074         0.122882           -4.102           0.000
    L3.VVL.TO         -0.336660         0.131142           -2.567           0.010
    L5.TSX            -0.468865         0.129902           -3.609           0.000
    L10.VVL.TO        -0.272380         0.127437           -2.137           0.033  
    '''
    normalize_data = normalize_data[['VVL.TO', 'DOW', 'TSX']]
    model = VAR(normalize_data)
    model_fit = model.fit(maxlags=13) # it use maximum of 13 lags for both series
    print(model_fit.summary())


trainNode = node(train_val, inputs=["trade_data", "normalize_data"], outputs=None)


# Create a data source
data_catalog = DataCatalog({"trade_data": MemoryDataSet()})

# Assign "nodes" to a "pipeline"
pipeline = Pipeline([ 
                        getStockNode,
                        normalizeNode,
                        analysisNode,
                        trainNode
                    ])

# Create a "runner" to run the "pipeline"
runner = SequentialRunner()

# Execute a pipeline
runner.run(pipeline, data_catalog)


plt.show()