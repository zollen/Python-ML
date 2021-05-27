'''
Created on May 27, 2021

@author: zollen
@url: https://towardsdatascience.com/pythons-predict-proba-doesn-t-actually-predict-probabilities-and-how-to-fix-it-f582c21d63fc
'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sb
import warnings


warnings.filterwarnings("ignore")
sb.set_style('whitegrid')

data = pd.read_csv("../data/pima-indians-diabetes.csv")

feature_df = data[['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']]
label_df = data['class']


X_train, X_test, y_train, y_test = train_test_split(feature_df, label_df, test_size=0.3, random_state=0)

model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]

nbins = 10
y_means, proba_means = calibration_curve(y_test.values, probs, n_bins=nbins, strategy='quantile')

print(y_means)
print(proba_means)

plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Perfect calibration')
plt.plot(proba_means, y_means)
plt.show()