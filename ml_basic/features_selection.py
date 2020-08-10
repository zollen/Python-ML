'''
Created on Aug. 3, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

PROJECT_DIR=str(Path(__file__).parent.parent) 
df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/iris.csv'))


LABEL = [ "variety" ]
FEATURES = [ "sepal.length","sepal.width","petal.length","petal.width" ]

np.set_printoptions(precision=2)

print("==== Select K Best Features Selection ====")
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(df[FEATURES], df[LABEL])

print(test.scores_)

data = fit.transform(df[FEATURES])
print(data[0:10])

print("==== LogisticRegression Features Selection ====")
model = LogisticRegression(max_iter=10000)
rfe = RFE(model, 3)
fit = rfe.fit(df[FEATURES], np.ravel(df[LABEL]))
print("Number of Features: ", fit.n_features_)
print("Selected Features: ", fit.n_features_to_select)
print("Feature Ranking: ", fit.ranking_)

print("============= PCA Features Selection ==========")
pca = PCA(n_components = 4)
fit = pca.fit(df[FEATURES])
print("Explained Variance: ", fit.explained_variance_ratio_)
print("Singular Value: ", fit.singular_values_)
print(fit.components_)
#print(pca.transform(df[FEATURES]))

print("============= ExtraTreeClassifier ==============")
model = ExtraTreesClassifier()
model.fit(df[FEATURES], np.ravel(df[LABEL]))
print(model.feature_importances_)