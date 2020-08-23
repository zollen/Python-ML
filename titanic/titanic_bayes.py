'''
Created on Aug. 4, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)
pd.set_option('max_colwidth', 15)
pd.set_option('precision', 2)


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

for name in categorical_columns:
    encoder = preprocessing.LabelEncoder()
    encoder.fit(np.union1d(train_df[name].unique(), test_df[name].unique()))
    train_df[name] = encoder.transform(train_df[name].values)
    test_df[name] = encoder.transform(test_df[name].values)

train_df['family'] = train_df['n_siblings_spouses'] + train_df['parch']
test_df['family'] = test_df['n_siblings_spouses'] + test_df['parch']

categorical_columns = [ 'sex', 'family', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns

gnb = GaussianNB()

model = gnb.fit(train_df[all_features_columns].values, train_df[label_column].values)



print("================= TRAINING DATA =====================")
preds = model.predict(train_df[all_features_columns])
print("Accuracy: %0.2f" % accuracy_score(train_df[label_column], preds))
print("Precision: %0.2f" % precision_score(train_df[label_column], preds))
print("Recall: %0.2f" % recall_score(train_df[label_column], preds))
print("AUC-ROC: %0.2f" % roc_auc_score(train_df[label_column], preds))
print("Log Loss: %0.2f" % log_loss(train_df[label_column], preds))
print(confusion_matrix(train_df[label_column], preds))


print("================= TEST DATA =====================")
if True:
    preds = gnb.predict(test_df[all_features_columns].values)
else:
    preds = gnb.predict_proba(test_df[all_features_columns].values)
    binarizer = Binarizer(threshold=0.50).fit(preds)
    preds = binarizer.transform(preds)
    preds = np.argmax(preds, axis=1)


print("Accuracy: %0.2f" % accuracy_score(test_df[label_column], preds))
print("Precision: %0.2f" % precision_score(test_df[label_column], preds))
print("Recall: %0.2f" % recall_score(test_df[label_column], preds))
print("AUC-ROC: %0.2f" % roc_auc_score(test_df[label_column], preds))
print("Log Loss: %0.2f" % log_loss(test_df[label_column], preds))
print(confusion_matrix(test_df[label_column], preds))
print(classification_report(test_df[label_column], preds))
