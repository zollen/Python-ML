'''
Created on May 10, 2021

@author: zollen
'''

import pandas_datareader.data as web
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
from pandas_datareader._utils import RemoteDataError
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

def plot_stock_trend_and_returns(ticker, titles, start_date, end_date, all_returns):
    
    #get the data for this ticker
    prices = web.DataReader(ticker, 'yahoo', start=start_date, end=end_date).Close
    prices.index = [d.date() for d in prices.index]
    
    plt.figure(figsize=(10,6))
    
    #plot stock price
    plt.subplot(2,1,1)
    plt.plot(prices)
    plt.title(titles[0], fontsize=16)
    plt.ylabel('Price ($)', fontsize=14)
    
    #plot stock returns
    plt.subplot(2,1,2)
    plt.plot(all_returns[0], all_returns[1], color='g')
    plt.title(titles[1], fontsize=16)
    plt.ylabel('Pct. Return', fontsize=14)
    plt.axhline(0, color='k', linestyle='--')
    
    plt.tight_layout()
    
    plt.show()
    
def perform_analysis_for_stock(ticker, start_date, end_date, return_period_weeks, verbose=False):
    """
    Inputs:
        ticker: the ticker symbol to analyze
        start_date: the first date considered in simulation
        end_date: the last date considered in simulation
        return_period_weeks: the number of weeks in which to calculate returns
        verbose: True if you want to print simulation steps
        
    Outputs:
        average and standard deviation of returns for simulated runs of this ticker within the given date range
    """
    
    #get the data for this ticker
    try:
        prices = web.DataReader(ticker, 'yahoo', start=start_date, end=end_date).Close
    #could not find data on this ticker
    except (RemoteDataError, KeyError):
        #return default values
        return -np.inf, np.inf, None
    
    prices.index = [d.date() for d in prices.index]
    
    #this will store all simulated returns
    pct_return_after_period = []
    buy_dates = []

    #assume we buy the stock on each day in the range
    for buy_date, buy_price in prices.iteritems():
        #get price of the stock after given number of weeks
        sell_date = buy_date + timedelta(weeks=return_period_weeks)
        
        try:
            sell_price = prices[prices.index == sell_date].iloc[0]
        #trying to sell on a non-trading day, skip
        except IndexError:
            continue
        
        #compute the percent return
        pct_return = (sell_price - buy_price)/buy_price
        pct_return_after_period.append(pct_return)
        buy_dates.append(buy_date)
        
        if verbose:
            print('Date Buy: %s, Price Buy: %s'%(buy_date,round(buy_price,2)))
            print('Date Sell: %s, Price Sell: %s'%(sell_date,round(sell_price,2)))
            print('Return: %s%%'%round(pct_return*100,1))
            print('-------------------')
    
    #if no data collected return default values
    if len(pct_return_after_period) == 0:
        return -np.inf, np.inf, None
    
    #report average and deviation of the percent returns
    return np.mean(pct_return_after_period), np.std(pct_return_after_period), [buy_dates, pct_return_after_period]


#start date for simulation. 
#Further back means more training data but risk of including patterns that no longer exist
#More recent means less training data but only using recent patterns
start_date, end_date = datetime(2020,4,1), datetime.now().date()

#set number of weeks in which you want to see return
return_period_weeks = 4

#I want at least this much average return
min_avg_return  = 0.1

#I want at most this much volatility in return
max_dev_return = 0.07

series_tickers = [
        ["XIU.TO", "iShare S&P/TSX 60 index ETF"],
        ["XIC.TO", "iShare Core S&P/TSX Capped Composite ETF"],
        ["XSP.TO", "iShares Core S&P 500 Index ETF (CAD- Hedged)"],
        ["XEF.TO", "iShares Core MSCI EAFE IMI Index ETF"],
        ["XBB.TO", "iShares Core Canadian Universe Bond Index ETF"],
        ["XUS.TO", "iShares Core S&P 500 Index ETF"],
        ["XSB.TO", "iShares Core Canadian Short Term Bond Index ETF"],
        ["XUU.TO", "iShares Core S&P U.S. Total Market Index ETF"],
        ["XDV.TO", "iShares Canadian Select Dividend Index ETF"],
        ["XAW.TO", "iShares Core MSCI All Country World ex Canada Index ETF"],
        ["XCB.TO", "iShares Canadian Corporate Bond Index ETF"],
        ["XSH.TO", "iShares Core Canadian Short Term Corporate Bond Index ETF"],
        ["CPD.TO", "iShares S&P/TSX Canadian Preferred Share Index ETF"],
        ["XQQ.TO", "iShares NASDAQ 100 Index ETF (CAD-Hedged)"],
        ["XFN.TO", "iShares S&P/TSX Capped Financials Index ETF"],
        ["XRE.TO", "iShares S&P/TSX Capped REIT Index ETF"],
        ["XIN.TO", "iShares MSCI EAFE Index ETF (CAD-Hedged)"],
        ["XEG.TO", "iShares S&P/TSX Capped Energy Index ETF"],
        ["XGD.TO", "iShares S&P/TSX Global Gold Index ETF"],
        ["XEI.TO", "iShares S&P/TSX Composite High Dividend Index ETF"],
        ["XEC.TO", "iShares Core MSCI Emerging Markets IMI Index ETF"],
        ["CBO.TO", "iShares 1-5 Year Laddered Corporate Bond Index ETF"],
        ["CDZ.TO", "iShares S&P/TSX Canadian Dividend Aristocrats Index ETF"],
        ["XGRO.TO", "iShares Core Growth ETF Portfolio"],
        ["FIE.TO", "iShares Canadian Financial Monthly Income ETF"],
        ["CGL.TO", "iShares Gold Bullion ETF"],
        ["XFN.TO", "iShares Core MSCI EAFE IMI Index ETF (CAD-Hedged)"],
        ["XBAL.TO", "iShares Core Balanced ETF Portfolio"],
        ["XTR.TO", "iShares Diversified Monthly Income ETF"],
        ["CWW.TO", "iShares Global Water Index ETF"],
        ["COW.TO", "iShares Global Agriculture Index ETF"],
        ["CGLC.TO", "iShares Gold Bullion ETF"],
        ["XBM.TO", "iShares S&P/TSX Global Base Metals Index ETF"],
        ["CGR.TO", "iShares Global Real Estate Index ETF"],
        ["CIF.TO", "iShares Global Infrastructure Index ETF"],
        ["XPF.TO", "iShares S&P/TSX North American Preferred Stock Index ETF"],
        ["SVR.TO", "iShares Silver Bullion ETF"],
        ["SVRC.TO", "iShares Silver Bullion ETF"]
    ]


for ticker, name in series_tickers:
    avg_return, dev_return, all_returns = perform_analysis_for_stock(ticker, start_date, end_date, return_period_weeks)

    print(ticker, name)

    if avg_return > min_avg_return and dev_return < max_dev_return:
        title_price = '%s\n%s'%(ticker, name)
        title_return = 'Avg Return: %s%% | Dev Return: %s%%'%(round(100*avg_return,2), round(100*dev_return,2))
        plot_stock_trend_and_returns(ticker, [title_price, title_return], start_date, end_date, all_returns)
        

