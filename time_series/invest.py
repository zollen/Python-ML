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
start_date, end_date = datetime.now().date() - timedelta(weeks=24), datetime.now().date()

#set number of weeks in which you want to see return
return_period_weeks = 12

#I want at least this much average return
min_avg_return  = 0.1

#I want at most this much volatility in return
max_dev_return = 0.03

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
        ["XBAL.TO", "iShares Core Balanced ETF Portfolio"],
        ["XTR.TO", "iShares Diversified Monthly Income ETF"],
        ["CWW.TO", "iShares Global Water Index ETF"],
        ["COW.TO", "iShares Global Agriculture Index ETF"],
        ["XBM.TO", "iShares S&P/TSX Global Base Metals Index ETF"],
        ["CGR.TO", "iShares Global Real Estate Index ETF"],
        ["CIF.TO", "iShares Global Infrastructure Index ETF"],
        ["XPF.TO", "iShares S&P/TSX North American Preferred Stock Index ETF"],
        ["SVR.TO", "iShares Silver Bullion ETF"],
        ["XGRO.TO", "iShares Core Growth ETF Portfolio"],
        ["XEQT.TO", "iShares All-Equity ETF Portfolio"],
        ["XDIV.TO", "iShares Core MSCI Canadian Quality Dividend Index ETF"],
        
        ["VCN.TO", "FTSE Canada All Cap Index ETF"],
        ["VCE.TO", "FTSE Canada Index ETF"],
        ["VRE.TO", "FTSE Canadian Capped REIT Index ETF"],
        ["VDY.TO", "FTSE Canadian High Dividend Yield Index ETF"],
        ["VIU.TO", "FTSE Developed All Cap ex North America Index ETF"],
        ["VDU.TO", "FTSE Developed All Cap ex U.S. Index ETF"],
        ["VEF.TO", "FTSE Developed All Cap ex U.S. Index ETF (CAD-hedged)"],
        ["VEE.TO", "FTSE Emerging Markets All Cap Index ETF"],
        ["VXC.TO", "FTSE Global All Cap ex Canada Index ETF"],
        ["VVO.TO", "Global Minimum Volatility ETF"],
        ["VMO.TO", "Global Momentum Factor ETF"],
        ["VVL.TO", "Global Value Factor ETF"],
        ["VFV.TO", "S&P 500 Index ETF"],
        ["VSP.TO", "S&P 500 Index ETF (CAD-hedged)"],
        ["VGG.TO", "U.S. Dividend Appreciation Index ETF"],
        ["VGH.TO", "U.S. Dividend Appreciation Index ETF (CAD-hedged)"],
        ["VUN.TO", "U.S. Total Market Index ETF"],
        ["VUS.TO", "U.S. Total Market Index ETF (CAD-hedged)"],
        ["VAB.TO", "Canadian Aggregate Bond Index ETF"],
        ["VCB.TO", "Canadian Corporate Bond Index ETF"],
        ["VGV.TO", "Canadian Government Bond Index ETF"],
        ["VLB.TO", "Canadian Long-Term Bond Index ETF"],
        ["VSB.TO", "Canadian Short-Term Bond Index ETF"],
        ["VSC.TO", "Canadian Short-Term Corporate Bond Index ETF"],
        ["VBG.TO", "Global ex-U.S. Aggregate Bond Index ETF (CAD-hedged)"],
        ["VBU.TO", "U.S. Aggregate Bond Index ETF (CAD-hedged)"],
        ["VGRO.TO", "Vanguard Growth ETF Portfolio"],
        ["VBAL.TO", "Vanguard Balanced ETF Portfolio"],
        ["VEQT.TO", "Vanguard ALL-Equity ETF Portfolio"],
        
        ["HAB.TO", "Horizons Active Corporate Bond ETF"],
        ["HAD.TO", "Horizons Active Cdn Bond ETF"],
        ["HAF.TO", "Horizons Active Global Fixed Income ETF"],
        ["HEMB.TO", "Horizons Active Emerging Markets Bond ETF"],
        ["HFR.TO", "Horizons Active Ultra-Short Term ETF F"],
        ["HMP.TO", "Horizons Active Cdn Municipal Bond ETF"],
        ["HPR.TO", "Horizons Active Preferred Share ETF"],
        ["HUF.TO", "Horizons Active Ultra-Short Term US Bond ETF"],
        ["HYI.TO", "Horizons Active High Yield Bond ETF"],
        ["HSL.TO", "Horizons Active Floating Rate Senior Loan ETF"],
        ["HYI.TO", "Horizons Active High Yield Bond ETF"],
        ["HSL.TO", "Horizons Active Floating Rate Senior Loan ETF"],
        ["HAL.TO", "Horizons Active Cdn Dividend ETF"],
        ["HAZ.TO", "Horizons Active Global Dividend ETF"],
        ["HEX.TO", "Horizons Enhanced Income Equity ETF"],
        ["HEA.TO", "Horizons Enhanced Income US Equity (USD) ETF"],
        ["HEJ.TO", "Horizons Enhanced Income International Equity ETF"],
        ["HEE.TO", "Horizons Enhanced Income Energy ETF"],
        ["HEF.TO", "Horizons Enhanced Income Financials ETF"],
        ["HEP.TO", "Horizons Enhanced Income Gold Producers ETF"],
        ["HGY.TO", "Horizons Gold Yield ETF"],
        ["HAC.TO", "Horizons Seasonal Rotation ETF"],
        ["MIND.TO", "Horizons Active A.I. Global Equity ETF"],
        ["HARC.TO", "Horizons Absolute Return Global Currency ETF"],
        ["HARB.TO", "Horizons Tactical Absolute Return Bond ETF"],
        ["HRAA.TO", "Horizons ReSolve Adaptive Asset Allocation ETF"],
        ["HBAL.TO", "Horizons Balanced TRI ETF Portfolio"],
        ["HCON.TO", "Horizons Conservative TRI ETF Portfolio"],
        ["HGRO.TO", "Horizons Growth TRI ETF Portfolio"],
        
        ["ZAG.TO", "BMO Aggregate Bond Index ETF"],
        ["ZDB.TO", "BMO Discount Bond Index ETF"],
        ["ZSB.TO", "BMO Short-Term Bond Index ETF"],
        ["ZCPB.TO", "BMO Core Plus Bond Fund"],
        ["ZMSB.TO", "BMO Global Multi-Sector Bond Fund"],
        ["ZGSB.TO", "BMO Global Strategic Bond Fund"],
        ["ZST.TO", "BMO Ultra Short-Term Bond ETF"],
        ["ZGB.TO", "BMO Government Bond Index ETF"],
        ["ZFS.TO", "BMO Short Federal Bond Index ETF"],
        ["ZFM.TO", "BMO Mid Federal Bond Index ETF"],
        ["ZFL.TO", "BMO Long Federal Bond Index ETF"],
        ["ZRR.TO", "BMO Real Return Bond Index ETF"],
        ["ZPS.TO", "BMO Short Provincial Bond Index ETF"],
        ["ZMP.TO", "BMO Mid Provincial Bond Index ETF"],
        ["ZPL.TO", "BMO Long Provincial Bond Index ETF"],
        ["ZTIP.TO", "BMO Short-Term US TIPS Index ETF"],
        ["ZMBS.TO", "BMO Canadian MBS Index ETF"],
        ["ZCB.TO", "BMO Corporate Bond Index ETF"],
        ["ESGB.TO", "BMO ESG Corporate Bond Index ETF"],
        ["ZQB.TO", "BMO High Quality Corporate Bond Index ETF"],
        ["ZBBB.TO", "BMO BBB Corporate Bond Index ETF"],
        ["ZCS.TO", "BMO Short Corporate Bond Index ETF"],
        ["ZCM.TO", "BMO Mid Corporate Bond Index ETF"],
        ["ZLC.TO", "BMO Long Corporate Bond Index ETF"],
        ["ZSU.TO", "BMO Short-Term US Hedged to CAD Index ETF"],
        ["ZIC.TO", "BMO Mid-Term US Bond Index ETF"],
        ["ZMU.TO", "BMO Mid-Term US Hedged to CAD Index ETF"],
        ["ESGF.TO", "BMO ESG US Bond Hedged to CAD Index ETF"],
        ["ZHY.TO", "BMO High Yield US Bond Hedged to CAD Index ETF"],
        ["ZJK.TO", "BMO High Yield US Corporate Bond Index ETF"],
        ["ESGH.TO", "BMO ESG High Yield US Corporate Bond Index ETF"],
        ["ZFH.TO", "BMO Floating Rate High Yield ETF"],
        ["ZEF.TO", "BMO Emerging Markets Bond Hedged to CAD Index ETF"],
        ["ZPR.TO", "BMO Laddered Preferred Share Index ETF"],
        ["ZUP.TO", "BMO US Preferred Share Index ETF"],
        ["ZHP.TO", "BMO US Preferred Share Hedged to CAD Index ETF"],
        ["ZCON.TO", "BMO Conservative ETF"],
        ["ZBAL.TO", "BMO Balanced ETF"],
        ["ZESG.TO", "BMO Balanced ESG ETF"],
        ["ZGRO.TO", "BMO Growth ETF"],
        ["ZMI.TO", "BMO Monthly Income ETF"],
        ["ZWC.TO", "BMO Canadian High Dividend Covered Call ETF"],
        ["ZWB.TO", "BMO Covered Call Canadian Banks ETF"],
        ["ZWK.TO", "BMO Covered Call US Banks ETF"],
        ["ZWT.TO", "BMO Covered Call Technology ETF"],
        ["ZWU.TO", "BMO Covered Call Utilities ETF"],
        ["ZWA.TO", "BMO Covered Call DowJones Hedged to CAD ETF"],
        ["ZWH.TO", "BMO US High Dividend Covered Call ETF"],
        ["ZWS.TO", "BMO US High Dividend Covered Call Hedged to CAD ETF"],
        ["ZWP.TO", "BMO Europe High Dividend Covered Call ETF"],
        ["ZWE.TO", "BMO Europe High Dividend Covered Call Hedged to CAD ETF"],
        ["ZWG.TO", "BMO Global High Dividend Covered Call ETF"],
        ["ZPAY.TO", "BMO Premium Yield ETF"],
        ["ZPW.TO", "BMO US Put Write ETF"],
        ["ZPH.TO", "BMO US Put Write Hedged to CAD ETF"],
        ["ZCN.TO", "BMO S&P/TSX Capped Composite Index ETF"],
        ["ESGA.TO", "BMO MSCI Canada ESG Leaders Index ETF"],
        ["ZDV.TO", "BMO Canadian Dividend ETF"],
        ["ZLB.TO", "BMO Low Volatility Canadian Equity ETF"],
        ["ZVC.TO", "BMO MSCI Canada Value Index ETF"],
        ["ZSP.TO", "BMO S&P 500 Index ETF"],
        ["ZUE.TO", "BMO S&P 500 Hedged to CAD Index ETF"],
        ["ZMID.TO", "BMO S&P US Mid Cap Index ETF"],
        ["ZSML.TO", "BMO S&P US Small Cap Index ETF"],
        ["ZDJ.TO", "BMO Dow Jones Industrial Average Hedged to CAD Index ETF"],
        ["ZQQ.TO", "BMO NASDAQ 100 Equity Hedged to CAD Index ETF"],
        ["ZNQ.TO", "BMO NASDAQ 100 Equity Index ETF"],
        ["ESGY.TO", "BMO MSCI USA ESG Leaders Index ETF"],
        ["ZUQ.TO", "BMO MSCI USA High Quality Index ETF"],
        ["ZDY.TO", "BMO US Dividend ETF"],
        ["ZUD.TO", "BMO US Dividend Hedged to CAD ETF"],
        ["ZLU.TO", "BMO Low Volatility US Equity Hedged to CAD ETF"],
        ["ZVU.TO", "BMO MSCI USA Value Index ETF"],
        ["ZFC.TO", "BMO SIA Focused Canadian Equity Fund ETF Series"],
        ["ZFN.TO", "BMO SIA Focused North American Equity Fund ETF Series"],
        ["ZZZD.TO", "BMO Tactical Dividend ETF Fund"],
        ["WOMN.TO", "BMO Women In Leadership Fund"],
        ["ZEA.TO", "BMO MSCI EAFE Index ETF"],
        ["ZDM.TO", "BMO MSCI EAFE Hedged to CAD Index ETF"],
        ["ESGE.TO", "BMO MSCI EAFE ESG Leaders Index ETF"],
        ["ZINN.TO", "BMO MSCI Innovation Index ETF"],
        ["ZGEN.TO", "BMO MSCI Genomic Innovation Index ETF"],
        ["ZFIN.TO", "BMO MSCI Fintech Innovation Index ETF"],
        ["ZINT.TO", "BMO MSCI Next Gen Internet Innovation Index ETF"],
        ["ZAUT.TO", "BMO MSCI Tech & Industrial Innovation Index ETF"],
        ["ZCLN.TO", "BMO Clean Energy Index ETF"],
        ["ZDI.TO", "BMO International Dividend ETF"],
        ["ZDH.TO", "BMO International Dividend Hedged to CAD ETF"],
        ["ZLI.TO", "BMO Low Volatility International Equity ETF"],
        ["ZLD.TO", "BMO Low Volatility International Equity Hedged to CAD ETF"],
        ["ZEQ.TO", "BMO MSCI Europe High Quality Hedged to CAD Index ETF"],
        ["ZGQ.TO", "BMO MSCI All Country World High Quality Index ETF"],
        ["ESGG.TO", "BMO MSCI Global ESG Leaders Index ETF"],
        ["ZEM.TO", "BMO MSCI Emerging Markets Index ETF"],
        ["ZLE.TO", "BMO Low Volatility Emerging Markets Equity ETF"],
        ["ZCH.TO", "BMO China Equity Index ETF"],
        ["ZID.TO", "BMO India Equity Index ETF"],
        ["ZEB.TO", "BMO Equal Weight Banks Index ETF"],
        ["ZEO.TO", "BMO Equal Weight Oil & Gas Index ETF"],
        ["ZUT.TO", "BMO Equal Weight Utilities Index ETF"],
        ["ZRE.TO", "BMO Equal Weight REITs Index ETF"],
        ["ZIN.TO", "BMO Equal Weight Industrials Index ETF"],
        ["ZHU.TO", "BMO Equal Weight US Health Care Index ETF"],
        ["ZUH.TO", "BMO Equal Weight US Health Care Hedged to CAD Index ETF"],
        ["ZBK.TO", "BMO Equal Weight US Banks Index ETF"],
        ["ZUB.TO", "BMO Equal Weight US Banks Hedged to CAD Index ETF"],
        ["ZGI.TO", "BMO Global Infrastructure Index ETF"],
        ["ZMT.TO", "BMO Equal Weight Global Base Metals Hedged to CAD Index ETF"],
        ["ZGD.TO", "BMO Equal Weight Global Gold Index ETF"],
        ["DISC.TO", "BMO Global Consumer Discretionary Hedged to CAD Index ETF"],
        ["STPL.TO", "BMO Global Consumer Staples Hedged to CAD Index ETF"],
        ["COMM.TO", "BMO Global Communications Index ETF"],
        ["ZJG.TO", "BMO Junior Gold Index ETF"],  
        
        ["RBNK.TO", "RBC CANADIAN Bank YIELD INDEX ETF"],
        ["RLB.TO", "RBC 1-5 Year Laddered Canadian Bond ETF"],
        ["RBO.TO", "RBC 1-5 Year Laddered Corporate Bond ETF"],
        ["RQI.TO", "RBC Target 2021 Corporate Bond Index ETF"],
        ["RQJ.TO", "RBC Target 2022 Corporate Bond Index ETF"],
        ["RQK.TO", "RBC Target 2023 Corporate Bond Index ETF"],
        ["RQL.TO", "RBC Target 2024 Corporate Bond Index ETF"],
        ["RQN.TO", "RBC Target 2025 Corporate Bond Index ETF"],
        ["RQO.TO", "RBC Target 2026 Corporate Bond Index ETF"],
        ["RQP.TO", "RBC Target 2027 Corporate Bond Index ETF"],
        ["RPSB.TO", "RBC PH&N Short Term Canadian Bond ETF"],
        ["RUSB.TO", "RBC Short Term U.S. Corporate Bond ETF"],
        ["RBDI.TO", "RBC BlueBay Global Diversified Income (CAD Hedged) ETF"],
        ["RPF.TO", "RBC Canadian Preferred Share ETF"],
        ["RCD.TO", "RBC Quant Canadian Dividend Leaders ETF"],
        ["RCE.TO", "RBC Quant Canadian Equity Leaders ETF"],
        ["RUD.TO", "RBC Quant U.S. Dividend Leaders ETF"],
        ["RUE.TO", "RBC Quant U.S. Equity Leaders ETF"],
        ["RUBY.TO", "RBC U.S. Banks Yield Index ETF"],
        ["RUBH.TO", "RBC U.S. Banks Yield (CAD Hedged) Index ETF"],
        ["RPD.TO", "RBC Quant European Dividend Leaders ETF"],
        ["RID.TO", "RBC Quant EAFE Dividend Leaders ETF"],
        ["RIE.TO", "RBC Quant EAFE Equity Leaders ETF"],
        ["RIEH.TO", "RBC Quant EAFE Equity Leaders (CAD Hedged) ETF"],
        ["RXD.TO", "RBC Quant Emerging Markets Dividend Leaders ETF"],
        ["RXE.TO", "RBC Quant Emerging Markets Equity Leaders ETF"]        
    ]

goodETFs = []

for ticker, name in series_tickers:
    avg_return, dev_return, all_returns = perform_analysis_for_stock(ticker, start_date, end_date, return_period_weeks)

    print("{:8} {:80} | {:0.4f}, {:0.4f}".format(ticker, name, avg_return, dev_return))

    if avg_return > min_avg_return and dev_return < max_dev_return:
        goodETFs.append([ticker, name, avg_return, dev_return])
        
        if False:
            title_price = '%s\n%s'%(ticker, name)
            title_return = 'Avg Return: %s%% | Dev Return: %s%%'%(round(100*avg_return,2), round(100*dev_return,2))
            plot_stock_trend_and_returns(ticker, [title_price, title_return], start_date, end_date, all_returns)
        

print()
print("======= WAIT WEEKS PERIOD {} MIN AVG RETURN {} MAX AVG DEV {} ==========".format(return_period_weeks, min_avg_return, max_dev_return))

for ticker, name, avg_return, dev_return in goodETFs:
    print("{:8} {:80} | {:0.4f}, {:0.4f}".format(ticker, name, avg_return, dev_return))