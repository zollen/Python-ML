'''
Created on Apr. 7, 2021

@author: zollen
'''

import yfinance as yf
import pandas as pd

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

tickerSymbol = "MSFT"

tickerData = yf.Ticker(tickerSymbol)

ticketDf = tickerData.history(period='1d', start = '2010-1-1', end='2020-3-20')

print(ticketDf.head())

