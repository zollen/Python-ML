'''
Created on Sep. 30, 2021

@author: zollen
@url: https://towardsdatascience.com/tricky-way-of-using-dimensionality-reduction-for-outlier-detection-in-python-4ee7665cdf99
@desc: UMAP (Uniform Manifold Approximation & Projection) is a dimensionality reduction 
    algorithm introduced in 2018. It combines the best features of PCA and tSNE — 
    it can scale to large datasets quickly and compete with PCA in terms of speed, and 
    project data to low dimensional space much more effectively and beautifully than tSNE.
    
    Isolation Forest - Return the anomaly score of each sample using the 
       IsolationForest algorithm
    
'''

import umap
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

df = pd.read_csv('../data/iris.csv')

df['variety'] = df['variety'].map({'Setosa': 0, 'Versicolor':1, 'Virginica':2})
X, y = df[['sepal.length', 'sepal.width', 'petal.length']], df['variety']

print(df.head())

manifold = umap.UMAP(n_components=2)
manifold.fit(X, y)

X_2d = manifold.transform(X)

iso = IsolationForest(n_estimators=3000, n_jobs=9)
labels = iso.fit_predict(X_2d)
outlier_idx = np.where(labels == -1)[0]
print("Outlier detected")
print(outlier_idx)
