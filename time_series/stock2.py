'''
Created on Jun. 14, 2021

@author: zollen
@url: https://www.youtube.com/watch?v=Vyr5dthe-2s
@code: https://github.com/ritvikmath/YouTubeVideoCode/blob/main/ARMA%20Stock%20Forecasting.ipynb
'''

import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import seaborn as sb
from tqdm import tqdm
from statsmodels.tools.sm_exceptions import ValueWarning, HessianInversionWarning, ConvergenceWarning
import warnings

#in practice do not supress these warnings, they carry important information about the status of your model
warnings.filterwarnings('ignore', category=ValueWarning)
warnings.filterwarnings('ignore', category=HessianInversionWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

sb.set_style('whitegrid')

tickerSymbol = 'AAPL'
data = yf.Ticker(tickerSymbol)

start_date = datetime(2021, 1, 1)
end_date = datetime(2021, 4, 1)

prices = data.history(start=start_date, end=end_date).Close
returns = prices.pct_change().dropna()

if False:
    figure, ax = plt.subplots(2, 1, figsize=(10, 6))
    plt.subplots_adjust(hspace=1)
    ax[0].plot(prices)
    ax[0].set_xlim(start_date, end_date)
    ax[0].set_ylabel('Prices', fontsize=20)
    
    
    ax[1].plot(returns)
    ax[1].set_xlim(start_date, end_date)
    ax[1].set_ylabel('Return', fontsize=20)


if False:
    figure, ax = plt.subplots(2, 1, figsize=(10, 6))
    plt.subplots_adjust(hspace=1)
    # AR(5) and MA(5)
    plot_acf(returns, ax = ax[0])
    plot_pacf(returns, ax = ax[1])






def run_simulation(returns, prices, amt, order, thresh, verbose=False, plot=True):
    if type(order) == float:
        thresh = None
        
    curr_holding = False
    buy_price = 0.00
    pred = 0
    events_list = []
    init_amt = amt

    #go through dates
    for date, r in tqdm (returns.iloc[14:].items(), total=len(returns.iloc[14:])):
        #if you're currently holding the stock, sell it
        if curr_holding:
            sell_price = prices.loc[date]
            curr_holding=False
            ret = (sell_price-buy_price)/buy_price
            amt *= (1+ret)
            events_list.append(('s', date, ret))
            
            if verbose:
                print('Sold at $%s'%sell_price)
                print('Predicted Return: %s'%round(pred,4))
                print('Actual Return: %s'%(round(ret, 4)))
                print('=======================================')
            continue

        #get data til just before current date
        curr_data = returns[:date]
        
        if type(order) == tuple:
            try:
                #fit model
                model = ARIMA(curr_data, order=order).fit(maxiter=200)

                #get forecast
                pred = model.forecast()[0][0]

            except:
                pred = thresh - 1



        #if you predict a high enough return and not holding, buy stock
        if (not curr_holding) and \
        ((type(order) == float and np.random.random() < order) 
         or (type(order) == tuple and pred > thresh)
         or (order == 'last' and curr_data[-1] > 0)):
            
            curr_holding = True
            buy_price = prices.loc[date]
            events_list.append(('b', date))
            if verbose:
                print('Bought at $%s'%buy_price)
                
    if verbose:
        print('Total Amount: $%s'%round(amt,2))
        
    #graph
    if plot:
    
        plt.figure(figsize=(10,4))
        plt.plot(prices[14:])

        y_lims = (int(prices.min()*.95), int(prices.max()*1.05))
        shaded_y_lims = int(prices.min()*.5), int(prices.max()*1.5)

        for idx, event in enumerate(events_list):
            plt.axvline(event[1], color='k', linestyle='--', alpha=0.4)
            if event[0] == 's':
                color = 'green' if event[2] > 0 else 'red'
                plt.fill_betweenx(range(*shaded_y_lims), 
                                  event[1], events_list[idx-1][1], color=color, alpha=0.1)

        tot_return = round(100*(amt / init_amt - 1), 2)
        tot_return = str(tot_return) + '%'
        plt.title("%s Price Data\nThresh=%s\nTotal Amt: $%s\nTotal Return: %s"%(tickerSymbol, thresh, round(amt,2), tot_return), fontsize=20)
        plt.ylim(*y_lims)

    
    return amt



if False:
    '''
    Baseline Model: Random Buying
    Total Return: 3.69%
    '''
    run_simulation(returns, prices, 100, 0.5, None, verbose=False)
    
    if False:
        # let's run 1000 times
        final_amts = [run_simulation(returns, prices, 100, 0.5, None, verbose=False, plot=False) for _ in range(1000)]
        plt.figure(figsize=(10,4))
        sb.distplot(final_amts)
        plt.axvline(np.mean(final_amts), color='k', linestyle='--')
        plt.axvline(100, color='g', linestyle='--')
        plt.title('Avg: $%s\nSD: $%s'%(round(np.mean(final_amts),2), round(np.std(final_amts),2)), fontsize=20)

if False:
    '''
    Slighly Better Model: If last return was positive, buy
    Total Return: -6.39%
    '''
    run_simulation(returns, prices, 100, 'last', None, verbose=False)
    
if False:
    '''
    Try AR(1) Model with threshold 0
    Total Return: -2.86%
    '''
    run_simulation(returns, prices, 100, (1,0,0), 0, verbose=False)
        
if False:
    '''
    Try AR(1) Model with threshold 0.001
    Total Return: -4.91%
    '''
    run_simulation(returns, prices, 100, (1,0,0), 0.001, verbose=False)  
        
if False:
    '''
    Try AR(1) Model with threshold 0.005
    Total Return: -3.14%
    '''
    run_simulation(returns, prices, 100, (1,0,0), 0.005, verbose=False)  


if False:
    '''
    Try AR(5) Model with threshold 0
    Total Return: 3.88%
    '''
    run_simulation(returns, prices, 100, (5,0,0), 0, verbose=False)  
        
if False:
    '''
    Try AR(5) Model with threshold 0.001
    Total Return: 3.88%
    '''
    run_simulation(returns, prices, 100, (5,0,0), 0.001, verbose=False)
    
if False:
    '''
    Try AR(5) Model with threshold 0.005
    Total Return: 10.00%
    '''
    run_simulation(returns, prices, 100, (5,0,0), 0.005, verbose=False)  
    
    
if True:
    '''
    Try ARIMA(5,5) Model with threshold 0
    Total Return: -2.71%
    '''
    run_simulation(returns, prices, 100, (5,0,5), 0, verbose=False)  
        
if True:
    '''
    Try ARIMA(5,5) Model with threshold 0.001
    Total Return: -2.71%
    '''
    run_simulation(returns, prices, 100, (5,0,5), 0.001, verbose=False)
    
if True:
    '''
    Try ARIMA(5,5) Model with threshold 0.005
    Total Return: -1.51%
    '''
    run_simulation(returns, prices, 100, (5,0,5), 0.005, verbose=False)      
    
'''
It appears ARIMA(5,5) with any threshold do poorly, overfitting?
Sometimes, simple models do better, complicated models not necessary do well
'''      

plt.tight_layout()
plt.show()
