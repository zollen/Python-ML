'''
Created on Jun. 2, 2021

@author: zollen
'''


from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA

from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

train_df = pd.read_csv('train.csv')
train_df = train_df[(train_df['store'] == 1) & (train_df['item'] == 1)] # item 1 in store 1

train_df['date'] = pd.to_datetime(train_df['date'])
train_df = train_df.set_index('date')
train_df = train_df.asfreq(pd.infer_freq(train_df.index), method="pad")
train_df['sales'] = train_df['sales'].astype('float64')
y = train_df['sales']
y_to_train = y.iloc[:(len(y)-365)]
y_to_test = y.iloc[(len(y)-365):] 



autoETS = AutoETS(auto=True, sp=12, n_jobs=-1)
autoETS.fit(y_to_train)
y_forecast = autoETS.predict(y_to_test.index)
print("RMSE: %0.4f" % mean_squared_error(y_to_test, y_forecast))
if True:
    plt.figure(figsize=(16,8))
    plt.plot(y_to_test)
    plt.plot(y_to_test.index, y_forecast)
    plt.legend(('Data', 'Predictions'), fontsize=16)
    plt.title("Price vs Prediction", fontsize=20)
    plt.ylabel('Price', fontsize=16) 
    plt.show()