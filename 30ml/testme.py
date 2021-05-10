'''
Created on Mar. 22, 2021

@author: zollen
@url: https://towardsdatascience.com/train-all-classification-or-regression-models-in-one-line-of-python-code-4ccdcaf01f0c
@desc: One wrapper for test the data with many classifiers or regressors
'''

import pyforest
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyClassifier
import warnings


warnings.filterwarnings("ignore")

train_df = pd.read_csv('../data/iris.csv')

feature_df = train_df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
label_df = train_df['variety']

X_train, X_test, y_train, y_test = train_test_split(feature_df, label_df, test_size=0.3, random_state=0)


clf = LazyClassifier(verbose=0,ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)
