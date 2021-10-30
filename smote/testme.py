'''
Created on Oct. 30, 2021

@author: zollen
@url: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
@desc SMOTE for rebalancing unbalanced data
'''
from numpy import mean
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
    n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)


counter = Counter(y)
print(counter)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(DecisionTreeClassifier(random_state=5), X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.5f' % mean(scores))

print()
print("Let's apply Over Sample for data with so few positive cases")

over = SMOTE(sampling_strategy='auto', random_state=5)
X_ovr, y_ovr = over.fit_sample(X, y)

counter = Counter(y_ovr)
print(counter)

scores = cross_val_score(DecisionTreeClassifier(random_state=5), X_ovr, y_ovr, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.5f' % mean(scores))


print()
print("Let's apply both Over + Under Sample for data with so few positive cases")


over = SMOTE(sampling_strategy='auto', random_state=5)
under = RandomUnderSampler(sampling_strategy='auto', random_state=5)


X_ovr, y_ovr = over.fit_sample(X, y)
X_ou, y_ou = under.fit_sample(X_ovr, y_ovr)

counter = Counter(y_ovr)
print(counter)

scores = cross_val_score(DecisionTreeClassifier(random_state=5), X_ou, y_ou, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.5f' % mean(scores))