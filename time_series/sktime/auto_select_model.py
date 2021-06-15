'''
Created on Jun. 2, 2021

@author: zollen
@url: https://github.com/alan-turing-institute/sktime/blob/main/examples/01_forecasting.ipynb
'''


from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import MultiplexForecaster

from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split, ForecastingGridSearchCV, SlidingWindowSplitter
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from statsmodels.tsa.seasonal import STL
import pandas as pd
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


models = MultiplexForecaster(
            forecasters=[
                ("ets", ExponentialSmoothing(trend="add", seasonal="additive", sp=12)),
                ("arima", ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12), suppress_warnings=True)),
                ("tbats", TBATS(sp=12, use_trend=True, use_box_cox=True))
                ]
            )
cv = SlidingWindowSplitter(initial_window=int(len(y_to_train) * 0.5), window_length=30)
forecaster_param_grid = {"selected_forecaster": ["ets", "arima", 'tbats' ]}
gscv = ForecastingGridSearchCV(models, cv=cv, param_grid=forecaster_param_grid)
                
gscv.fit(y_to_train)
y_forecast = gscv.predict(fh)

print(gscv.best_forecaster_)
print(gscv.best_score_)
print(gscv.best_params_)

print("RMSE: %0.4f" % mean_absolute_percentage_error(y_to_test, y_forecast))
plot_series(y_to_train, y_to_test, y_forecast, labels=["y_train", "y_test", "y_pred"])
plt.show()
