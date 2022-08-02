'''
Created on Aug. 1, 2022

@author: STEPHEN
@url: https://towardsdatascience.com/three-approaches-to-feature-engineering-for-time-series-2123069567be
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error
from sklego.preprocessing import RepeatingBasisFunction
import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_columns=100
pd.options.display.max_rows = 5000
pd.set_option('display.width', 1000)
sb.set_style("whitegrid")

# for reproducibility
np.random.seed(42)
# generate the DataFrame with dates
range_of_dates = pd.date_range(start="2017-01-01",
                               end="2020-12-30")
X = pd.DataFrame(index=range_of_dates)
# create a sequence of day numbers
X["day_nr"] = range(len(X))
X["day_of_year"] = X.index.day_of_year
# generate the components of the target
signal_1 = 3 + 4 * np.sin(X["day_nr"] / 365 * 2 * np.pi)
signal_2 = 3 * np.sin(X["day_nr"] / 365 * 4 * np.pi + 365/2)
noise = np.random.normal(0, 0.85, len(X))
# combine them to get the target series
y = signal_1 + signal_2 + noise

TRAIN_END = 3 * 365



results_df = pd.DataFrame(columns=['actuals', 'model_1'])
results_df['actuals'] = y


if False:
    '''
    Approach #1: Dummy variable (pd.get_dummies). Not ideally as it generates step-like predictions because
    of dummy columns values [1, 0], but we want the predicted values to be continuous
    '''

    X_1 = pd.DataFrame(
        data=pd.get_dummies(X.index.month, drop_first=True, prefix="month")
    )
    X_1.index = X.index
    
    model_1 = LinearRegression().fit(X_1.iloc[:TRAIN_END],
                                     y.iloc[:TRAIN_END])
    results_df["model_1"] = model_1.predict(X_1)
    results_df[["actuals", "model_1"]].plot(figsize=(16,4),
                                            title="Fit using month dummies")
    plt.axvline(date(2020, 1, 1), c="m", linestyle="--");

    # plot
    y.plot(figsize=(16,4), title="Generated time series");
    plt.show()
    

if False:
    '''
    Approach #2: Using sin() and cos() to break down the date columns
    '''
    
    def sin_transformer(period):
        return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

    def cos_transformer(period):
        return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
    
    X_2 = X.copy()
    X_2["month"] = X_2.index.month
    X_2["month_sin"] = sin_transformer(12).fit_transform(X_2)["month"]
    X_2["month_cos"] = cos_transformer(12).fit_transform(X_2)["month"]
    X_2["day_sin"] = sin_transformer(365).fit_transform(X_2)["day_of_year"]
    X_2["day_cos"] = cos_transformer(365).fit_transform(X_2)["day_of_year"]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16,8))
    X_2[["month_sin", "month_cos"]].plot(ax=ax[0])
    X_2[["day_sin", "day_cos"]].plot(ax=ax[1])
    plt.suptitle("Cyclical encoding with sine/cosine transformation");
    
    X_2_daily = X_2[["day_sin", "day_cos"]]
    model_2 = LinearRegression().fit(X_2_daily.iloc[:TRAIN_END],
                                     y.iloc[:TRAIN_END])
    results_df["model_2"] = model_2.predict(X_2_daily)
    results_df[["actuals", "model_2"]].plot(figsize=(16,4),
                                            title="Fit using sine/cosine features")
    
    plt.show()
    plt.axvline(date(2020, 1, 1), c="m", linestyle="--");
    
    

if True:
    rbf = RepeatingBasisFunction(n_periods=12,
                             column="day_of_year",
                             input_range=(1,365),
                             remainder="drop")
    
    rbf.fit(X)

    X_3 = pd.DataFrame(index=X.index,
                       data=rbf.transform(X))
    
    model_3 = LinearRegression().fit(X_3.iloc[:TRAIN_END],
                                 y.iloc[:TRAIN_END])
    results_df["model_3"] = model_3.predict(X_3)
    results_df[["actuals", "model_3"]].plot(figsize=(16,4),
                                            title="Fit using RBF features")
    plt.axvline(date(2020, 1, 1), c="m", linestyle="--");
    
    
    X_3.plot(subplots=True, figsize=(14, 8),
             sharex=True, title="Radial Basis Functions",
             legend=False);
    plt.show()