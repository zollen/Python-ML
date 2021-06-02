'''
Created on Jun. 2, 2021

@author: zollen
'''

from sktime.forecasting.base import ForecastingHorizon
from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.compose import make_reduction

from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split

from sklearn.metrics import mean_squared_error
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



fh = ForecastingHorizon(y_to_test.index, is_relative=False)


regressor = KNeighborsRegressor(n_neighbors=1)
model =  make_reduction(regressor, window_length=15, strategy="recursive")
model.fit(y_to_train)
y_forecast = model.predict(fh)

print("RMSE: %0.4f" % np.sqrt(mean_squared_error(y_to_test, y_forecast)))
plot_series(y_to_train, y_to_test, y_forecast, labels=["y_train", "y_test", "y_pred"])
plt.show()
