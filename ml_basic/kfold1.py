'''
Created on Aug. 12, 2020

@author: zollen
'''

import os
from pathlib import Path

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd



pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)
pd.set_option('max_colwidth', 15)
pd.set_option('precision', 2)

label_column = [ 'class' ]
all_features_columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age' ]

PROJECT_DIR=str(Path(__file__).parent.parent)  
df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/pima-indians-diabetes.csv'))

print(df.shape)
print(df.info())
print("===== NUll values records =========")
print(df.isnull().sum())
print(df.describe())

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)

kfold = KFold(n_splits = 20, random_state = 7)
results = cross_val_score(model, df[all_features_columns], df[label_column], cv = kfold)
print(round(results.mean(), 2))
