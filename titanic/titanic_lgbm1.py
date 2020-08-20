'''
Created on Aug. 10, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
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

func = lambda x : np.round(x, 2)

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR , 'data/eval.csv'))

print(train_df.info())
print(train_df.isnull().sum())
print(train_df.describe())

labels = train_df[label_column]


for name in categorical_columns + label_column:
    encoder = preprocessing.LabelEncoder()   
    keys = np.union1d(train_df[name].unique(), test_df[name].unique())
   
    if len(keys) == 2:
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    train_df[name] = encoder.transform(train_df[name].values)
    test_df[name] = encoder.transform(test_df[name].values)
    


print("=============== K Best Features Selection ==================")
model = SelectKBest(score_func=chi2, k=5)
kBest = model.fit(train_df[all_features_columns], labels)
print(np.stack((all_features_columns, func(kBest.scores_)), axis=1))

print("================ Decision Tree Best Features Selection ============")
model = DecisionTreeClassifier()
model.fit(train_df[all_features_columns], labels)
print(np.stack((all_features_columns, func(model.feature_importances_)), axis=1))

model = LGBMClassifier(num_leaves=32, max_depth=16, min_data_in_leaf=10)
model.fit(train_df[all_features_columns], train_df[label_column])
print("LightGBM Score: ", model.score(train_df[all_features_columns], train_df[label_column]))
print("================= TRAINING DATA =====================")
preds = model.predict(train_df[all_features_columns])
print("Accuracy: %0.2f" % accuracy_score(train_df[label_column], preds))
print("Precision: %0.2f" % precision_score(train_df[label_column], preds))
print("Recall: %0.2f" % recall_score(train_df[label_column], preds))
print("AUC-ROC: 0.2f" % roc_auc_score(train_df[label_column], preds))
print("Log Loss: 0.2f" % log_loss(train_df[label_column], preds))
print(confusion_matrix(train_df[label_column], preds))


print("================= TEST DATA =====================")
preds = model.predict(test_df[all_features_columns])
print("Accuracy: %02.f" % accuracy_score(test_df[label_column], preds))
print("Precision: %0.2f" % precision_score(test_df[label_column], preds))
print("Recall: %0.2f" % recall_score(test_df[label_column], preds))
print("AUC-ROC: %0.2f" % roc_auc_score(test_df[label_column], preds))
print("Log Loss: %0.2f" % log_loss(test_df[label_column], preds))
print(confusion_matrix(test_df[label_column], preds))
print(classification_report(test_df[label_column], preds))

