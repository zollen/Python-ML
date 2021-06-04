'''
Created on Jun. 2, 2021

@author: zollen
'''

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction

from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series

from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer

from sktime.forecasting.model_selection import temporal_train_test_split, ForecastingGridSearchCV, SlidingWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError, mean_absolute_percentage_error

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

model =  make_reduction(RandomForestRegressor(), 
                        scitype="tabular-regressor", window_length=15, strategy="recursive")
model.fit(y_to_train)
y_forecast = model.predict(fh)

print("REDUCT(RandomForestRegressor) RMSE: %0.4f" % mean_absolute_percentage_error(y_to_test, y_forecast))
plot_series(y_to_train, y_to_test, y_forecast, labels=["y_train", "y_test", "y_pred"])



if True:
    
    regressor_param_grid = {"n_estimators": [100, 200, 300]}
    forecaster_param_grid = {"window_length": [7, 12, 15]}
    
    regressor = BayesSearchCV(estimator = RandomForestRegressor(), 
                              search_spaces = regressor_param_grid,
                              scoring = make_scorer(mean_absolute_percentage_error, greater_is_better=False, needs_threshold=False),
                              n_jobs=20, 
                              n_iter=100,
                              return_train_score=False,
                              refit=True,
                              random_state = 23
                              )
    
    model =  make_reduction(regressor, scitype="tabular-regressor", strategy="recursive")

    # We fit the forecaster on the initial window, and then use temporal
    # cross-validation to find the optimal parameter.
    cv = SlidingWindowSplitter(initial_window=int(len(y_to_train) * 0.8), window_length=60)
    gscv = ForecastingGridSearchCV(
        model, 
        strategy="refit", 
        cv=cv, 
        param_grid=forecaster_param_grid,
        scoring=MeanAbsolutePercentageError()
    )
    gscv.fit(y_to_train)
    y_forecast = gscv.predict(fh)
    print(gscv.best_forecaster_)
    print(gscv.best_score_)
    print(gscv.best_params_)
    print(gscv.cv_results_)

    print("Optimize(RandomForestRegressor) RMSE: %0.4f" % mean_absolute_percentage_error(y_to_test, y_forecast))
    plot_series(y_to_train, y_to_test, y_forecast, labels=["y_train", "y_test", "y_pred"])


plt.show()
