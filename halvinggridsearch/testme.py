'''
Created on Jun. 17, 2021

@author: zollen
'''

import pandas as pd
import time

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV

import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv("../data/iris.csv")
iris['variety'] = iris['variety'].map({'Setosa': 0, 'Versicolor': 0, 'Virginica': 1})

print(iris.head())
print(iris.info())

xgb_cl = xgb.XGBClassifier(objective="binary:logistic", verbose=None, seed=1121218)

X = iris[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = iris['variety']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1121218, stratify=y
)


_ = xgb_cl.fit(X_train, y_train)


preds = xgb_cl.predict(X_test)

param_grid = {
    "max_depth": [3, 4, 5, 7],
    "gamma": [0, 0.25, 1],
    "reg_lambda": [0, 1, 10],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.8],  # Fix subsample
    "colsample_bytree": [0.5],  # Fix colsample_bytree
}

n_candidates = 1

for params in param_grid.values():
    n_candidates *= len(params)

print("TOTAL SEARCH AREA: ", n_candidates)

grid_start_time = time.time()
grid_cv = GridSearchCV(xgb_cl, param_grid, scoring="roc_auc", n_jobs=-1, cv=3)
grid_cv.fit(X, y)
grid_end_time = time.time()


halv_rand_start_time = time.time()
halving_random_cv = HalvingRandomSearchCV(
    xgb_cl, param_grid, scoring="roc_auc", n_jobs=-1, n_candidates="exhaust", factor=4
)
halving_random_cv.fit(X, y)
halv_rand_end_time = time.time()




print(grid_cv.best_score_)
print(grid_cv.best_params_)
print("Search Time: ", grid_end_time - grid_start_time)
print("==============================================")
print(halving_random_cv.best_score_)
print(halving_random_cv.best_params_)
print("Search Time: ", halv_rand_end_time - halv_rand_start_time)
print("==============================================")


