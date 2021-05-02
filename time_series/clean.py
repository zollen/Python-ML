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



def parser(s):
    return datetime.strptime(s, '%Y-%m')

ice_cream_heater_df = pd.read_csv('ice_cream_vs_heater.csv', parse_dates=[0], 
                                  index_col=0, squeeze=True, date_parser=parser)

ice_cream_heater_df = ice_cream_heater_df.asfreq(pd.infer_freq(ice_cream_heater_df.index))

heater_series = ice_cream_heater_df.heater

print(heater_series)

def plot_series(series, title):
    plt.plot(series, color='red')
    plt.ylabel(title, fontsize=10)

    for year in range(2004, 2021):
        plt.axvline(datetime(year,1,1), linestyle='--', color='k', alpha=0.5)


plt.figure(figsize=(10, 10))        

plt.subplot(5, 1, 1)
plot_series(heater_series, 'Heater Data')

    
    
'''
Normalize
'''
avg, dev = heater_series.mean(), heater_series.std()
heater_series = (heater_series - avg) / dev


plt.subplot(5, 1, 2)
plot_series(heater_series, 'Normalized')
plt.axhline(0, linestyle='--', color='k', alpha=0.3)

    
    
'''
Take First Difference to remove Trend
'''
heater_series = heater_series.diff().dropna()  # dropna() remove the first NAN value

plt.subplot(5, 1, 3)
plot_series(heater_series, 'First Difference')
plt.axhline(0, linestyle='--', color='k', alpha=0.3)

    
'''
Remove Increasing Volatitiy
'''
annual_volatility = heater_series.groupby(heater_series.index.year).std()
# calculate the standard deviation of each year
print(annual_volatility)

heater_annual_vol = heater_series.index.map(lambda d: annual_volatility.loc[d.year])
# map the yearly standard deviation in each time series data based on the year each data belongs to
print(heater_annual_vol)

heater_series = heater_series / heater_annual_vol

plt.subplot(5, 1, 4)
plot_series(heater_series, 'Remove Volatity')
plt.axhline(0, linestyle='--', color='k', alpha=0.3)


'''
Remove Seasonality(seasonal patterns)
'''    
month_avgs = heater_series.groupby(heater_series.index.month).mean()
# calculate the average of each month
print(month_avgs)

heater_month_avg = heater_series.index.map(lambda d: month_avgs.loc[d.month])
# map the monthly average in each time series data based on the month each data belongs to
print(heater_month_avg)

heater_series = heater_series - heater_month_avg

plt.subplot(5, 1, 5)
plot_series(heater_series, 'Remove Seasonality')
plt.axhline(0, linestyle='--', color='k', alpha=0.3)


plt.show()
    
'''
Finally we should conduct some formal tests for checking true stationary with unit-root test, and others.. 
'''