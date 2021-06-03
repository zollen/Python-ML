'''
Created on Jun. 2, 2021

@author: zollen
'''


from sktime.forecasting.ets import AutoETS
from sktime.forecasting.base import ForecastingHorizon

from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split

from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')


y = load_airline()
#plot_series(y);
y_to_train, y_to_test = temporal_train_test_split(y, test_size=36)
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

fh = ForecastingHorizon(y_to_test.index, is_relative=False)


model = AutoETS(auto=True, sp=12, seasonal="additive", n_jobs=-1)
model.fit(y_to_train)
print(model.summary())
y_forecast = model.predict(fh)

print("RMSE: %0.4f" % np.sqrt(mean_squared_error(y_to_test, y_forecast)))
plot_series(y_to_train, y_to_test, y_forecast, labels=["y_train", "y_test", "y_pred"])
plt.show()