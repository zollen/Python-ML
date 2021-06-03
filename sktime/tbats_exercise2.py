'''
Created on Jun. 2, 2021

@author: zollen
'''

from datetime import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.tbats import TBATS
from sklearn.metrics import mean_squared_error
from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')
 
y = load_airline()
#plot_series(y);
y_to_train, y_to_test = temporal_train_test_split(y, test_size=36)
fh = ForecastingHorizon(y_to_test.index, is_relative=False)
#plot_series(y_to_train, y_to_test, labels=["y_train", "y_test"])
#plt.show()

if False:
    yy = pd.DataFrame({'Date': y.index.to_timestamp(freq='M'), 'Price': y.values})
    yy = yy.set_index('Date')
    yy['Price'] = yy['Price'].astype('float64')
    yy = yy.asfreq(pd.infer_freq(yy.index), method="pad")
    stl = STL(yy)
    result = stl.fit()
    
    seasonal, trend, resid = result.seasonal, result.trend, result.resid
    plt.figure(figsize=(8,6))
    
    plt.subplot(4,1,1)
    plt.plot(yy)
    plt.title('Original Series', fontsize=16)
    
    plt.subplot(4,1,2)
    plt.plot(trend)
    plt.title('Trend', fontsize=16)
    
    plt.subplot(4,1,3)
    plt.plot(seasonal)
    plt.title('Seasonal', fontsize=16)
    
    plt.subplot(4,1,4)
    plt.plot(resid)
    plt.title('Residual', fontsize=16)
    
    plt.tight_layout()
    plt.show()



model = TBATS(sp=12, use_trend=True, use_box_cox=False)
model.fit(y_to_train)
y_forecast = model.predict(fh)

print("RMSE: %0.4f" % np.sqrt(mean_squared_error(y_to_test, y_forecast)))
plot_series(y_to_train, y_to_test, pd.Series(data=y_forecast, index=y_to_test.index), labels=["y_train", "y_test", "y_pred"])
plt.show()
    
