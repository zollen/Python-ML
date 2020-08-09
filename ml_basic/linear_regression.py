'''
Created on Aug. 9, 2020

@author: zollen
'''
import numpy as np
from sklearn import linear_model
import seaborn as sb
import sklearn.metrics as sm
import matplotlib.pyplot as plt

sb.set_style('whitegrid')
np.set_printoptions(precision=2)

train = np.array([ [ -0.52, 0.5   ],
                   [  1.00, -1.00 ],
                   [  1.98, 1.44  ],
                   [ -0.74, -0.62 ],
                   [  0.68, 0.62  ],
                   [  3.16, 1.34  ],
                   [  8.58, 4.62  ],
                   [  9.94, 5.44  ],
                   [   9.8, 4.04  ],
                   [  7.94, 5.34  ],
                   [  7.52, 3.26  ]])

test = np.array([  [  1.68, 1.12  ],
                   [  5.62, 3.64  ],
                   [  4.78, 2.22  ],
                   [ -1.80, -1.36 ],
                   [  7.68, 4.00  ],
                   [  10.8, 5.78  ]])


X_train, Y_train = np.expand_dims(train[:, 0], axis=1), np.expand_dims(train[:, 1], axis=1)
X_test, Y_test = np.expand_dims(test[:, 0], axis=1), np.expand_dims(test[:, 1], axis=1)


model = linear_model.LinearRegression()

model.fit(X_train, Y_train)

Y_test_pred = model.predict(X_test)

print(Y_test_pred)
print("Regressor model performance:")
print("Mean absolute error(MAE) =", round(sm.mean_absolute_error(Y_test, Y_test_pred), 2))
print("Mean squared error(MSE) =", round(sm.mean_squared_error(Y_test, Y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(Y_test, Y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(Y_test, Y_test_pred), 2))
print("R2 score =", round(sm.r2_score(Y_test, Y_test_pred), 2))

plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, Y_test_pred, color = 'black', linewidth = 1)

plt.show()