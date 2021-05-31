'''
Created on May 18, 2021

@author: zollen
@url: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.varmax.VARMAX.html
@url: https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_varmax.html
@url: https://docs.w3cub.com/statsmodels/examples/notebooks/generated/statespace_varmax
'''

import pandas_datareader.data as web
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
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
    start_date, end_date = datetime.now().date() - timedelta(weeks=WEEKS_FOR_ANALYSIS), datetime.now().date() - timedelta(days=1)
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
    if False:
        fig, (a1, a2, a3, a4, a5, a6) = plt.subplots(6, 1)
        fig.set_size_inches(10, 10)
        a1.set_ylabel('TRADE(WWL.TO)', fontsize=8)
        a2.set_ylabel('TRADE(DOW)', fontsize=8)
        a3.set_ylabel('TRADE(TSX)', fontsize=8)
        a4.set_ylabel('NORMALIZE(VVL.TO)', fontsize=8)
        a5.set_ylabel('NORMALIZE(DOW)', fontsize=8)
        a6.set_ylabel('NORMALIZE(TSX)', fontsize=8)
        plot_pacf(trade_data['VVL.TO'], ax=a1, title="PACF Analysis of VVL.TO")
        plot_pacf(trade_data['DOW'], ax=a2, title=None)
        plot_pacf(trade_data['TSX'], ax=a3, title=None)
        plot_pacf(normalize_data['VVL.TO'], ax=a4, title=None)
        plot_pacf(normalize_data['DOW'], ax=a5, title=None)
        plot_pacf(normalize_data['TSX'], ax=a6, title=None)

analysisNode = node(analysis_data, inputs=["trade_data", "normalize_data"], outputs=None)

def train_val(data):
    '''
   
    '''
    
    test_data = data.iloc[len(data) - TEST_SIZE:]
    train_data = data.iloc[:len(data) - TEST_SIZE]

    model = sm.tsa.VARMAX(train_data, order=(3,3), trend="ct")
    results = model.fit(maxiter=100, disp=True) 
    print(results.summary())
    exit()
    
    if False:
        # ACF plots for residuals with 2 / sqrt(T)b bounds
        results.plot_acorr()
         
    preds = results.forecast(y=test_data.values, steps=TEST_SIZE)
    
    print("RMSE: %0.4f" % np.sqrt(mean_squared_error(test_data['VVL.TO'], preds[:,0])))
 
    if True:
        plt.figure(figsize=(10,4))
        plt.plot(data['VVL.TO'])
        plt.plot(test_data.index, preds[:,0])
        plt.legend(('Data', 'Predictions'), fontsize=16)
        plt.title("Prediction vs Observed", fontsize=20)
        plt.ylabel('Price', fontsize=16)



trainNode = node(train_val, inputs="trade_data", outputs=None)


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