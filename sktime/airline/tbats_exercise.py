'''
Created on Jun. 2, 2021

@author: zollen
'''

from datetime import datetime
import pandas as pd
from statsmodels.tsa.seasonal import STL
from tbats import TBATS
from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

if __name__ == '__main__':  
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


if __name__ == '__main__':  
    estimator = TBATS(seasonal_periods=(12, 30.5))
    model = estimator.fit(y_to_train)
    y_forecast = model.forecast(steps=36)
    print("=========================")
    print(model.summary())
    print("=========================")
    # Time series analysis
    print(model.y_hat) # in sample prediction
    print(model.resid) # in sample residuals
    print(model.aic)
    
    # Reading model parameters
    print(model.params.alpha)
    print(model.params.beta)
    print(model.params.x0)
    print(model.params.components.use_box_cox)
    print(model.params.components.seasonal_harmonics)
    
    print("RMSE: %0.4f" % mean_absolute_percentage_error(y_to_test, y_forecast))
    plot_series(y_to_train, y_to_test, pd.Series(data=y_forecast, index=y_to_test.index), labels=["y_train", "y_test", "y_pred"])
    plt.show()
    
print(datetime.now().strftime("%H:%M:%S"))