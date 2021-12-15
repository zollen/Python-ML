'''
Created on Dec. 15, 2021

@author: zollen
@desc: Local Outlier Factor
'''

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

X = [[-1.1], [0.2], [101.1], [0.3]]
clf = LocalOutlierFactor(n_neighbors=2)
print(clf.fit_predict(X))
print(clf.negative_outlier_factor_)


