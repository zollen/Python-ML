'''
Created on Jun. 2, 2021

@author: zollen
'''

from datetime import datetime
import pandas as pd
import numpy as np
from tbats import TBATS
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

train_df = pd.read_csv('train.csv')
train_df = train_df[(train_df['store'] == 1) & (train_df['item'] == 1)] # item 1 in store 1

train_df = train_df.set_index('date')
y = train_df['sales']
y_to_train = y.iloc[:(len(y)-365)]
y_to_test = y.iloc[(len(y)-365):] 

y_forecast = None
if __name__ == '__main__':  
    estimator = TBATS(seasonal_periods=(7, 365.25))
    model = estimator.fit(y_to_train)
    y_forecast = model.forecast(steps=365)
    print("=========================")
    print(model.summary())
    print("=========================")
    # Time series analysis
    print(model.y_hat) # in sample prediction
    print(model.resid) # in sample residuals
    print(model.aic)
    
    # Reading model parameters
    print(model.params.alpha)
    print(model.params.beta)
    print(model.params.x0)
    print(model.params.components.use_box_cox)
    print(model.params.components.seasonal_harmonics)
    
    print("RMSE: %0.4f" % np.sqrt(mean_squared_error(y_to_test, y_forecast)))
    if False:
        plt.figure(figsize=(16,8))
        plt.plot(y_to_test)
        plt.plot(y_to_test.index, y_forecast)
        plt.legend(('Data', 'Predictions'), fontsize=16)
        plt.title("Price vs Prediction", fontsize=20)
        plt.ylabel('Price', fontsize=16) 
        plt.show()
    
    
print(datetime.now().strftime("%H:%M:%S"))