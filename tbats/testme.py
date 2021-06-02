'''
Created on Jun. 2, 2021

@author: zollen
'''

import pandas as pd
from tbats import TBATS, BATS


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
    print("IN MAIN: ", y_forecast)
    
print(y_forecast)