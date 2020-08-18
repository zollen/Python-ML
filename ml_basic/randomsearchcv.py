'''
Created on Aug. 18, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)
pd.set_option('max_colwidth', 15)
pd.set_option('precision', 2)

label_column = [ 'class' ]
all_features_columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age' ]

PROJECT_DIR=str(Path(__file__).parent.parent)  
df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/pima-indians-diabetes.csv'))

alphas = np.array(  [ 1,  0.1, 0.01, 0.001, 0.0001,    0 ])
maxiters = np.array([ 1,  10,  100,   500,   1000, 5000, 10000 ])
tols = np.array([ 1, 1e-01, 1e-03, 1e-05, 1e-07 ])
solvers = np.array([ 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga' ])

param_grid = dict({ "alpha": alphas, "max_iter": maxiters, "tol":  tols, "solver": solvers })


model = Ridge()
grid = RandomizedSearchCV(estimator = model, param_distributions  = param_grid, n_jobs=50, n_iter = 50, random_state=7)
grid.fit(df[all_features_columns], df[label_column])

print(grid.cv_results_)
print("====================================================================================")
print("Best Score: ", grid.best_score_)
print("Best Alpha: ", grid.best_estimator_.alpha)
print("Best MaxIter: ", grid.best_estimator_.max_iter)
print("Best Tol: ", grid.best_estimator_.tol)
print("Best Solver: ", grid.best_estimator_.solver)



