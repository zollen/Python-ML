'''
Created on Aug. 4, 2021

@author: zollen
@desc the best or most used kaggle hyperparameters tuning
@url: https://towardsdatascience.com/why-is-everyone-at-kaggle-obsessed-with-optuna-for-hyperparameter-tuning-7608fdca337c
'''


import optuna
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold
import warnings

warnings.filterwarnings('ignore')


def objective(trial, X, y, cv, scoring):
    
    rf_params = {
        "n_estimators": trial.suggest_int(name="n_estimators", low=100, high=2000),
        "max_depth": trial.suggest_float("max_depth", 3, 8),
        "max_features": trial.suggest_categorical(
            "max_features", choices=["auto", "sqrt", "log2"]
        ),
        "n_jobs": -1,
        "random_state": 0,
    }
    
    # perform CV
    regressor = RandomForestRegressor(**rf_params)
    scores = cross_validate(regressor, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    # compute RMSLE
    rmsle = np.sqrt(-scores["test_score"].mean())
    return rmsle




df = pd.read_csv('../data/iris.csv')
df['variety'] = df['variety'].map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})
X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = df['variety']

'''
GridSampler  : the same as GridSearch of Sklearn. Never use for large search spaces!
RandomSampler: the same as RandomizedGridSearch of Sklearn.
TPESampler   : Tree-structured Parzen Estimator sampler - bayesian optimization 
                using kernel fitting
CmaEsSampler : a sampler based on CMA ES algorithm (does not allow categorical 
                hyperparameters).
'''
# Create study that minimizes
study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=0))

# Wrap the objective inside a lambda with the relevant arguments
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Pass additional arguments inside another function
func = lambda trial: objective(trial, X, y, cv=kf, scoring="neg_mean_squared_log_error")

# Start optimizing with 100 trials
study.optimize(func, n_trials=100)

print(f"Optimized Params: {study.best_params}")
print(f"Optimized RMSLE: {study.best_value:.5f}")
