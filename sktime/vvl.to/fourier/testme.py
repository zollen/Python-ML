'''
Created on Jun. 12, 2021

@author: zollen
@url: https://pythontic.com/visualization/signals/fouriertransform_fft
'''

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import fft
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')


end_date = datetime(2021, 6, 4)
start_date = end_date - timedelta(weeks=64)


def get_stock(TICKER):
   
    vvl = web.DataReader(TICKER, 'yahoo', start=start_date, end=end_date).Close
    vvl.index = [d.date() for d in vvl.index]
    
    prices = pd.DataFrame({
                            'Date' : vvl.index, 
                            'Prices' : vvl.values, 
                           })
    
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = prices.set_index('Date')
    prices = prices.asfreq(pd.infer_freq(prices.index), method="pad")
    prices['Prices'] = prices['Prices'].astype('float64')
    
    return prices

y = get_stock('VVL.TO')


# How many time points are needed i,e., Sampling Frequency

samplingFrequency   = len(y);

 

# At what intervals time points are sampled

samplingInterval       = 1 / samplingFrequency;

 
 

# Create subplot

figure, axis = plt.subplots(2, 1, figsize=(10, 8))

plt.subplots_adjust(hspace=1)

 
 

# Time domain representation for sine wave 2

axis[0].set_title('VVL.TO')
axis[0].plot(y)
axis[0].set_xlabel('Time')
axis[0].set_ylabel('Prices')




# Frequency domain representation

fourierTransform = fft(y['Prices'].values)/len(y)           # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(y)/2))] # Exclude sampling frequency

 

tpCount     = len(y['Prices'].values)
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/samplingFrequency
frequencies = values/timePeriod

 

# Frequency domain representation

axis[1].set_title('Fourier transform depicting the frequency components')
axis[1].plot(frequencies, abs(fourierTransform))
axis[1].set_xlabel('Frequency')
axis[1].set_ylabel('Amplitude')
axis[1].set_xlim(0, 20)
 
plt.tight_layout()
plt.show()