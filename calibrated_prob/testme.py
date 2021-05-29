'''
Created on May 27, 2021

@author: zollen
@url: https://towardsdatascience.com/pythons-predict-proba-doesn-t-actually-predict-probabilities-and-how-to-fix-it-f582c21d63fc
'''

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import seaborn as sb
import warnings


warnings.filterwarnings("ignore")
sb.set_style('whitegrid')

def score_model(observed, predicted):
    accuracy = accuracy_score(observed, predicted)
    precision = precision_score(observed, predicted)
    recall = recall_score(observed, predicted)
    auc = roc_auc_score(observed, predicted)
    loss = log_loss(observed, predicted)
    
    return accuracy, precision, recall, auc, loss
    
def show_score (observed, predicted):
    print("Accuracy: %0.2f    Precision: %0.2f    Recall: %0.2f    AUC: %0.2f    Loss: %0.2f" %
           (score_model(observed, predicted)))

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

# pretending an average learner with un-calibrated probabilities
model = RandomForestClassifier()
model.fit(X_train, y_train)

proba_valid = model.predict_proba(X_valid)

nbins = 10
y_means, proba_means = calibration_curve(y_valid, 
                                         proba_valid[:, 1], 
                                         n_bins=nbins, strategy='quantile')

plt.plot([0, 1], [0, 1], linestyle = '--')
plt.plot(proba_means, y_means) 

show_score(y_test, model.predict(X_test))
print("ECE(Random Forest): %0.4f" % expected_calibration_error(y_means, proba_means))



'''
We will use the output of the classifier (on validation data) to fit the calibrator and 
finally predicting probabilities on test data.

Note: Calibration should *not* be carried out on the same data that has been used for 
        training the first classifier.
'''

test_probs = model.predict_proba(X_test)

'''
Approach #1
QuadraticDiscriminantAnalysis
'''
iso_model = QuadraticDiscriminantAnalysis().fit(proba_valid, y_valid)
y_means, proba_means = calibration_curve(y_test, 
                                        iso_model.predict_proba(test_probs)[:, 1], 
                                        n_bins=nbins, strategy='quantile')
plt.plot(proba_means, y_means) 
show_score(y_test, iso_model.predict(test_probs))
print("ECE(Random Forest + QuadraticDiscriminantAnalysis): %0.4f" % expected_calibration_error(y_means, proba_means))

'''
Approach #2
Logistic Regression
'''
log_model = LogisticRegression().fit(proba_valid, y_valid)
y_means, proba_means = calibration_curve(y_test, 
                                         log_model.predict_proba(test_probs)[:, 1], 
                                         n_bins=nbins, strategy='quantile')
plt.plot(proba_means, y_means) 
show_score(y_test, log_model.predict(test_probs))
print("ECE(Random Forest + Logistic Regression): %0.4f" % expected_calibration_error(y_means, proba_means))


plt.legend(
    labels = ('Perfect calibration', 'Random Forest', 'Random Forest + QuadraticDiscriminantAnalysis', 'Random Forest + Logistic Regression'), 
    loc = 'upper left')
   
plt.show()   
'''
At this point we have three options for predicting probabilties
1. Random Forest
2. Random Forest + QuadraticDiscriminantAnalysis
3. Random Forest + Logistic Regression

RF  : Accuracy: 0.91    Precision: 1.00    Recall: 0.16    AUC: 0.58    Loss: 3.08
RF+Q: Accuracy: 0.86    Precision: 0.41    Recall: 0.85    AUC: 0.85    Loss: 4.98
RF+L: Accuracy: 0.95    Precision: 0.89    Recall: 0.56    AUC: 0.78    Loss: 1.87
ECE(Random Forest)                                : 0.0723
ECE(Random Forest + QuadraticDiscriminantAnalysis): 0.2755
ECE(Random Forest + Logistic Regression)          : 0.0085

Random Forest + Logistic Regression has the least calibration error
'''

