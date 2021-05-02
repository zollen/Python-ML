'''
Created on May 1, 2021

@author: zollen
@url: https://www.youtube.com/watch?v=7_Js8h709Dw&list=PLvcbYUQ5t0UHOLnBzl46_Q6QKtFgfMGc3&index=34&ab_channel=ritvikmath
'''

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)


def parser(s):
    return datetime.strptime(s, '%Y-%m')

ice_cream_heater_df = pd.read_csv('ice_cream_vs_heater.csv', parse_dates=[0], 
                                  index_col=0, squeeze=True, date_parser=parser)

ice_cream_heater_df = ice_cream_heater_df.asfreq(pd.infer_freq(ice_cream_heater_df.index))

heater_series = ice_cream_heater_df.heater

print(heater_series)

def plot_series(series):
    plt.figure(figsize=(12,6))
    plt.plot(heater_series, color='red')
    plt.ylabel('Search Frequency for "Heater"', fontsize=16)

    for year in range(2004, 2021):
        plt.axvline(datetime(year,1,1), linestyle='--', color='k', alpha=0.5)

        
if False:
    plot_series(heater_series)
    plt.show()
    
    
'''
Normalize
'''
avg, dev = heater_series.mean(), heater_series.std()
heater_series = (heater_series - avg) / dev

if False:
    plot_series(heater_series)
    plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    plt.show()
    
    
'''
Take First Difference to remove Trend
'''
heater_series = heater_series.diff().dropna()  # dropna() remove the first NAN value
if False:
    plot_series(heater_series)
    plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    plt.show()
    
'''
Remove Increasing Volatitiy
'''
annual_volatility = heater_series.groupby(heater_series.index.year).std()
print(annual_volatility)

heater_annual_vol = heater_series.index.map(lambda d: annual_volatility.loc[d.year])
print(heater_annual_vol)

heater_series = heater_series / heater_annual_vol

if False:
    plot_series(heater_series)
    plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    plt.show()
    
month_avgs = heater_series.groupby(heater_series.index.month).mean()
print(month_avgs)

heater_month_avg = heater_series.index.map(lambda d: month_avgs.loc[d.month])
print(heater_month_avg)

heater_series = heater_series - heater_month_avg

if False:
    plot_series(heater_series)
    plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    plt.show()