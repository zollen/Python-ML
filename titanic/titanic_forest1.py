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
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)
np.random.seed(87)


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

print(train_df.info())
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
    

model = RandomForestClassifier(n_estimators = 50)
rfe = RFE(model, 7)
rfe = rfe.fit(train_df[all_features_columns], labels)
print(rfe.support_)
print(rfe.ranking_)

model=sm.Logit(labels, train_df[all_features_columns])
result=model.fit()
print(result.summary2())


train_df['family'] = train_df['n_siblings_spouses'] + train_df['parch']
test_df['family'] = test_df['n_siblings_spouses'] + test_df['parch']
# alone are bigger than P0value 0.05, therefore we remove then
numeric_columns = [ 'fare' ]
categorical_columns = [ 'sex', 'family', 'class', 'deck' ]
all_features_columns = numeric_columns + categorical_columns

model=sm.Logit(labels, train_df[all_features_columns])
result=model.fit()
print(result.summary2())

model = RandomForestClassifier(n_estimators = 50, max_depth=14)

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
if True:
    preds = model.predict(test_df[all_features_columns])
else:
    preds = model.predict_proba(test_df[all_features_columns])
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

