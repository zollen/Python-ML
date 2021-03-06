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
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)

label_column = [ 'survived' ]
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns

PROJECT_DIR=str(Path(__file__).parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR , 'data/eval_processed.csv'))

print(train_df.info())
print(train_df.isnull().sum())
print(train_df.describe())

labels = train_df[label_column]

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

# OneHot Encoding    
if False:
    cat_columns = []

    for name in categorical_columns: 
        keys = np.union1d(train_df[name].unique(), test_df[name].unique())
    
        for key in keys:
            func = lambda x : 1 if x == key else 0
            train_df[name + "." + str(key)] = train_df[name].apply(func)
            test_df[name + "." + str(key)] = test_df[name].apply(func)
            cat_columns.append(name + "." + str(key))
 
    all_features_columns = numeric_columns + cat_columns

    print(train_df[all_features_columns].describe())

print(train_df.describe())

model = svm.SVC(kernel='linear', C=1.0)
rfe = RFE(model, 7)
rfe = rfe.fit(train_df[all_features_columns], labels)
print(rfe.support_)
print(rfe.ranking_)

model=sm.Logit(labels, train_df[all_features_columns])
result=model.fit()
print(result.summary2())

# alone are bigger than P0value 0.05, therefore we remove then
numeric_columns = [ 'fare' ]
all_features_columns = numeric_columns + categorical_columns

model=sm.Logit(labels, train_df[all_features_columns])
result=model.fit()
print(result.summary2())

#model = svm.SVC(kernel='linear', C=1.0)
model = svm.SVC(kernel='rbf', gamma ='auto', C=1.0)
#model = svm.NuSVC(nu = 0.5, kernel = 'rbf', gamma='auto')

model.fit(train_df[all_features_columns], labels)


print("================= TRAINING DATA =====================")
preds = model.predict(train_df[all_features_columns])
print("Accuracy: %0.2f " % accuracy_score(train_df[label_column], preds))
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

