'''
Created on May 27, 2021

@author: zollen
@url: https://towardsdatascience.com/pythons-predict-proba-doesn-t-actually-predict-probabilities-and-how-to-fix-it-f582c21d63fc
'''

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sb
import warnings


warnings.filterwarnings("ignore")
sb.set_style('whitegrid')

def expected_calibration_error(y, proba, bins = 'fd'):
    bin_count, bin_edges = np.histogram(proba, bins = bins)
    n_bins = len(bin_count)
    bin_edges[0] -= 1e-8 # because left edge is not included
    bin_id = np.digitize(proba, bin_edges, right = True) - 1
    bin_ysum = np.bincount(bin_id, weights = y, minlength = n_bins)
    bin_probasum = np.bincount(bin_id, weights = proba, minlength = n_bins)
    bin_ymean = np.divide(bin_ysum, bin_count, out = np.zeros(n_bins), where = bin_count > 0)
    bin_probamean = np.divide(bin_probasum, bin_count, out = np.zeros(n_bins), where = bin_count > 0)
    ece = np.abs((bin_probamean - bin_ymean) * bin_count).sum() / len(proba)
    return ece


X, y = make_classification(
            n_samples = 15000, 
            n_features = 50, 
            n_informative = 30, 
            n_redundant = 20,
            weights = [.9, .1],
            random_state = 0
        )
X_train, X_valid, X_test = X[:5000], X[5000:10000], X[10000:]
y_train, y_valid, y_test = y[:5000], y[5000:10000], y[10000:]



model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)

proba_valid = model.predict_proba(X_valid)[:, 1]

nbins = 10
y_means, proba_means = calibration_curve(y_valid, proba_valid, n_bins=nbins, strategy='quantile')

plt.plot([0, 1], [0, 1], linestyle = '--')
plt.plot(proba_means, y_means) 
print("Random Forest %0.4f" % expected_calibration_error(y_means, proba_means))



'''
We will use the output of the classifier (on validation data) to fit the calibrator and 
finally predicting probabilities on test data.

Note: Calibration should *not* be carried out on the same data that has been used for 
        training the first classifier.
'''



'''
Approach #1
Isotonic Regression
'''
iso_reg = IsotonicRegression(y_min = 0, y_max = 1, out_of_bounds = 'clip').fit(proba_valid, y_valid)
iso_probs = iso_reg.predict(model.predict_proba(X_test)[:, 1])
y_means, proba_means = calibration_curve(y_test, iso_probs, n_bins=nbins, strategy='quantile')
plt.plot(proba_means, y_means) 
print("Random Forest + Isotonic Regression %0.4f" % expected_calibration_error(y_means, proba_means))

'''
Approach #2
Logistic Regression
'''
log_reg = LogisticRegression().fit(proba_valid.reshape(-1, 1), y_valid)
log_probs = log_reg.predict_proba(model.predict_proba(X_test)[:, 1].reshape(-1, 1))[:, 1]
y_means, proba_means = calibration_curve(y_test, log_probs, n_bins=nbins, strategy='quantile')
plt.plot(proba_means, y_means) 
print("Random Forest + Logistic Regression %0.4f" % expected_calibration_error(y_means, proba_means))


plt.legend(
    labels = ('Perfect calibration', 'Random Forest', 'Random Forest + Isotonic', 'Random Forest + Logistic'), 
    loc = 'lower right')
   
plt.show()   
'''
At this point we have three options for predicting probabilties
1. Random Forest
2. Random Forest + isotinic Regresion
3. Random Forest + Logistic Regression

Random Forest 0.0838
Random Forest + Isotonic Regression 0.0075
Random Forest + Logistic Regression 0.0191

Random Forest + Isotonic Regression has the least calibration error
'''

