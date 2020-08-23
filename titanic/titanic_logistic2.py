'''
Created on Aug. 4, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt


def plot_roc(model, features, labels):
    logit_roc_auc = roc_auc_score(labels, model.predict(features))
    fpr, tpr, thresholds = roc_curve(labels, model.predict_proba(features)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.plot(fpr, thresholds, 'g--', label='decision thresholds')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
##  Saving the graph as image png.
##    plt.savefig('Log_ROC')


pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)
np.random.seed(87)

label_column = [ 'survived' ]
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'family', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns


PROJECT_DIR=str(Path(__file__).parent)
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR , 'data/eval_processed.csv'))

train_df['family'] = train_df['n_siblings_spouses'] + train_df['parch']
test_df['family'] = test_df['n_siblings_spouses'] + test_df['parch']

labels = train_df[label_column]

print(train_df.info())
print(train_df.isnull().sum())
print(train_df.describe())

print("Original Sample Size: Total Number of Survived: ", len(train_df[train_df['survived'] == 1]), 
      "Total Number of Dead: ", len(train_df[train_df['survived'] == 0]))


## Rebalance the dataset by oversmapling with replacement
rebalancer = RandomOverSampler(random_state=0)
#train_df, labels = rebalancer.fit_sample(train_df[all_features_columns], train_df[label_column])
print("ReBalacned Data pays more attention to the less ocurrence result")
print("This is not a good example for rebalance data. This technique is good for finding out who has the disease is more important")
print("ReBalanced Size: Total Number of Survived: ", len(labels[labels['survived'] == 1]),
      "Total Number of Dead: ", len(labels[labels['survived'] == 0]))



for name in categorical_columns:
    encoder = preprocessing.LabelEncoder()
        
    keys = np.union1d(train_df[name].unique(), test_df[name].unique())
   
    if len(keys) == 2:
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    train_df[name] = encoder.transform(train_df[name].values)
    test_df[name] = encoder.transform(test_df[name].values)

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
train_df[all_features_columns] = scaler.fit_transform(train_df[all_features_columns].values)
test_df[all_features_columns] = scaler.transform(test_df[all_features_columns].values)


 
if False:
    param_grid = dict({ "penalty": [ 'l2' ],
                       "C": [ 0.01, 1.0 ],
                       "class_weight": [ None, 'balanced' ],
                       "fit_intercept": [ True, False ],
                       "intercept_scaling": [ 0.1, 0.5, 1.0 ],
                       "solver": ['lbfgs', 'liblinear' , 'saga'],
                       "max_iter": [ 50, 300, 500, 750 ] })
    model = RandomizedSearchCV(estimator = LogisticRegression(), 
                        param_distributions = param_grid, n_jobs=50, n_iter=100)

    model.fit(train_df[all_features_columns], labels.squeeze())

    print("====================================================================================")
    print("Best Score: ", model.best_score_)
    print("Best Penalty: ", model.best_estimator_.penalty)
    print("Best MaxIter: ", model.best_estimator_.max_iter)
    print("Best Class Weight: ", model.best_estimator_.class_weight)
    print("Best l1 ratio: ", model.best_estimator_.l1_ratio)
    print("Best Fit Intecept: ", model.best_estimator_.fit_intercept)
    print("Best Intercept Scaling: ", model.best_estimator_.intercept_scaling)
    print("Best Solver: ", model.best_estimator_.solver)

    exit()



logreg = LogisticRegression()
rfe = RFE(logreg, 7)
rfe = rfe.fit(train_df[all_features_columns], labels)
print(rfe.support_)
print(rfe.ranking_)

model=sm.Logit(labels, train_df[all_features_columns])
result=model.fit()
print(result.summary2())


if True:
    model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.001)
else:
    model = LogisticRegression(max_iter=500, solver='lbfgs')
    
model.fit(train_df[all_features_columns], labels)

print("================= TRAINING DATA =====================")
preds = model.predict(train_df[all_features_columns])
print("Accuracy: %0.2f" % accuracy_score(train_df[label_column], preds))
print("Precision: %0.2f" % precision_score(train_df[label_column], preds))
print("Recall: %0.2f" % recall_score(train_df[label_column], preds))
print("AUC-ROC: %0.2f" % roc_auc_score(train_df[label_column], preds))
print("Log Loss: %0.2f" % log_loss(train_df[label_column], preds))
print(confusion_matrix(train_df[label_column], preds))

print("================= TEST DATA =====================")
preds = model.predict(test_df[all_features_columns])
print("Accuracy: %0.2f" % accuracy_score(test_df[label_column], preds))
print("Precision: %0.2f" % precision_score(test_df[label_column], preds))
print("Recall: %0.2f" % recall_score(test_df[label_column], preds))
print("AUC-ROC: %0.2f" % roc_auc_score(test_df[label_column], preds))
print("Log Loss: %0.2f" % log_loss(test_df[label_column], preds))
print(confusion_matrix(test_df[label_column], preds))
print(classification_report(test_df[label_column], preds))


plot_roc(model, test_df[all_features_columns].values, test_df[label_column].values)
#plt.show()