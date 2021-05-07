'''
Created on May 6, 2021

@author: zollen
'''


import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from scipy.stats import pearsonr

def parser(s):
    return datetime.strptime(s, '%Y-%m')

ice_cream_heater_df = pd.read_csv('ice_cream_vs_heater.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

ice_cream_heater_df = ice_cream_heater_df.asfreq(pd.infer_freq(ice_cream_heater_df.index))

if False:
    plt.figure(figsize=(12,6))
    ice_cream, = plt.plot(ice_cream_heater_df['ice cream'])
    heater, = plt.plot(ice_cream_heater_df['heater'], color='red')
    
    for year in range(2004, 2021):
        plt.axvline(datetime(year,1,1), linestyle='--', color='k', alpha=0.5)
    
    plt.legend(['Ice Cream', 'Heater'], fontsize=16)
    plt.show()


'''
Let's normalize and cleanup the data
'''

avgs = ice_cream_heater_df.mean()
devs = ice_cream_heater_df.std()

for col in ice_cream_heater_df.columns:
    ice_cream_heater_df[col] = (ice_cream_heater_df[col] - avgs.loc[col]) / devs.loc[col]
    
if False:
    plt.figure(figsize=(12,6))
    ice_cream, = plt.plot(ice_cream_heater_df['ice cream'])
    heater, = plt.plot(ice_cream_heater_df['heater'], color='red')
    
    for year in range(2004, 2021):
        plt.axvline(datetime(year,1,1), linestyle='--', color='k', alpha=0.5)
        
    plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    
    plt.legend(['Ice Cream', 'Heater'], fontsize=16)
    plt.show()

'''
Take first difference to remove the trend    
'''
ice_cream_heater_df = ice_cream_heater_df.diff().dropna()

if False:
    plt.figure(figsize=(12,6))
    ice_cream, = plt.plot(ice_cream_heater_df['ice cream'])
    heater, = plt.plot(ice_cream_heater_df['heater'], color='red')
    
    for year in range(2004, 2021):
        plt.axvline(datetime(year,1,1), linestyle='--', color='k', alpha=0.5)
        
    plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    plt.ylabel('First Difference', fontsize=18)
    
    plt.legend(['Ice Cream', 'Heater'], fontsize=16)
    plt.show()
    
if False:
    plt.figure(figsize=(12,6))
    ice_cream, = plt.plot(ice_cream_heater_df['ice cream'])
    
    for year in range(2004, 2021):
        plt.axvline(datetime(year,1,1), linestyle='--', color='k', alpha=0.5)
        
    plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    plt.ylabel('First Difference', fontsize=18)
    
    plt.legend(['Ice Cream'], fontsize=16)
    plt.show()
    
'''
Remove Increasing Volatity
'''
annual_volatility = ice_cream_heater_df.groupby(ice_cream_heater_df.index.year).std()

print(annual_volatility)

ice_cream_heater_df['ice_cream_annual_vol'] = ice_cream_heater_df.index.map(lambda d: annual_volatility.loc[d.year, 'ice cream'])
ice_cream_heater_df['heater_annual_vol'] = ice_cream_heater_df.index.map(lambda d: annual_volatility.loc[d.year, 'heater'])

print(ice_cream_heater_df)

ice_cream_heater_df['ice cream'] = ice_cream_heater_df['ice cream'] / ice_cream_heater_df['ice_cream_annual_vol']
ice_cream_heater_df['heater'] = ice_cream_heater_df['heater'] / ice_cream_heater_df['heater_annual_vol']

if False:
    plt.figure(figsize=(12,6))
    ice_cream, = plt.plot(ice_cream_heater_df['ice cream'])
    
    for year in range(2004, 2021):
        plt.axvline(datetime(year,1,1), linestyle='--', color='k', alpha=0.5)
        
    plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    plt.ylabel('First Yearly Volatility', fontsize=18)
    
    plt.legend(['Ice Cream'], fontsize=16)
    plt.show()
    
if False:
    plt.figure(figsize=(12,6))
    ice_cream, = plt.plot(ice_cream_heater_df['ice cream'])
    heater, = plt.plot(ice_cream_heater_df['heater'], color='red')
    
    for year in range(2004, 2021):
        plt.axvline(datetime(year,1,1), linestyle='--', color='k', alpha=0.5)
        
    plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    plt.ylabel('First Yearly Volatility', fontsize=18)
    
    plt.legend(['Ice Cream', 'Heater'], fontsize=16)
    plt.show()
    
'''
Remove Seasonality  
'''
month_avgs = ice_cream_heater_df.groupby(ice_cream_heater_df.index.month).mean()

print(month_avgs)

ice_cream_heater_df['ice_cream_month_avg'] = ice_cream_heater_df.index.map(lambda d: month_avgs.loc[d.month, 'ice cream'])
ice_cream_heater_df['heater_month_avg'] = ice_cream_heater_df.index.map(lambda d: month_avgs.loc[d.month, 'heater'])

print(ice_cream_heater_df)

ice_cream_heater_df['ice cream'] = ice_cream_heater_df['ice cream'] - ice_cream_heater_df['ice_cream_month_avg']
ice_cream_heater_df['heater'] = ice_cream_heater_df['heater'] - ice_cream_heater_df['heater_month_avg']

print(ice_cream_heater_df)

if False:
    plt.figure(figsize=(12,6))
    ice_cream, = plt.plot(ice_cream_heater_df['ice cream'])
    heater, = plt.plot(ice_cream_heater_df['heater'], color='red')
    
    for year in range(2004, 2021):
        plt.axvline(datetime(year,1,1), linestyle='--', color='k', alpha=0.5)
        
    plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    plt.ylabel('Remove Seasonality', fontsize=18)
    
    plt.legend(['Ice Cream', 'Heater'], fontsize=16)
    plt.show()
    
'''
PACF - Heater 
It shows Lag 1, 2 are significant
AR(2)
'''
if False:
    plot_pacf(ice_cream_heater_df['heater'])
    plt.show()
    
'''
PACF - Ice Cream 
'''
if False:
    plot_pacf(ice_cream_heater_df['ice cream'])
    plt.show()

'''
PACF - Ice Cream does not reveal enough info, so let use another technique
Using Pearson Correlation between "heater" and lagged "ice cream"
We need the correlation value to be very small correlation value and P value < 0.05
Only lag 13: correlation around 0.2 and P value < 0.05
Lag 13 is basically one year ago - one month ago
'''    
for lag in range(1, 14):
    heater_series = ice_cream_heater_df['heater'].iloc[lag:]
    lagged_ice_cream_series = ice_cream_heater_df['ice cream'].iloc[:-lag]
    print('Lag: %s'%lag)
    print(pearsonr(heater_series, lagged_ice_cream_series))
    print('------')

'''
Let's train the model
''' 
ice_cream_heater_df = ice_cream_heater_df[['ice cream', 'heater']]
model = VAR(ice_cream_heater_df)
model_fit = model.fit(maxlags=13) # it use maximum of 13 lags for both series
print(model_fit.summary())

'''
Results for equation heater
================================================================================
                   coefficient       std. error           t-stat            prob
--------------------------------------------------------------------------------
const                 0.005855         0.036062            0.162           0.871
L1.ice cream         -0.033113         0.084202           -0.393           0.694
L1.heater            -0.405367         0.077900           -5.204           0.000
L2.ice cream         -0.169804         0.092018           -1.845           0.065
L2.heater            -0.193569         0.084371           -2.294           0.022
L3.ice cream         -0.048999         0.095980           -0.511           0.610
L3.heater            -0.016958         0.085362           -0.199           0.843
L4.ice cream         -0.007633         0.094226           -0.081           0.935
L4.heater             0.009474         0.086950            0.109           0.913
L5.ice cream         -0.020253         0.095520           -0.212           0.832
L5.heater            -0.050607         0.089856           -0.563           0.573
L6.ice cream          0.040645         0.097944            0.415           0.678
L6.heater            -0.006504         0.095070           -0.068           0.945
L7.ice cream         -0.053440         0.097538           -0.548           0.584
L7.heater            -0.053795         0.100292           -0.536           0.592
L8.ice cream          0.074170         0.097997            0.757           0.449
L8.heater            -0.059001         0.100176           -0.589           0.556
L9.ice cream         -0.004720         0.096633           -0.049           0.961
L9.heater            -0.000168         0.100139           -0.002           0.999
L10.ice cream        -0.007920         0.094996           -0.083           0.934
L10.heater            0.040322         0.099970            0.403           0.687
L11.ice cream        -0.076564         0.094922           -0.807           0.420
L11.heater            0.022568         0.099181            0.228           0.820
L12.ice cream        -0.111959         0.095263           -1.175           0.240
L12.heater            0.082608         0.095266            0.867           0.386
L13.ice cream         0.203451         0.092065            2.210           0.027
L13.heater           -0.155390         0.083744           -1.856           0.064
================================================================================

The summary shows both ice cream and heater, but we want to best model the heater series.
lets scroll to the heater section. To make thing easy, we looks at the prob columns
(P value of less than 0.05)
L1.heater       0.000  <- no suprise PACF shows Lag 1 heater is useful
L2.heater       0.022  <- no suprise PACF shows Lag 2 heater is useful
L13.ice cream   0.027  <- Pearson correlation shows lag 13 ice cream is useful

Our Final Model is:
heater(t) = -0.405367 heater(t-1) - 0.193569 heater(t-2) + 0.203451 icecream(t-13)
'''

