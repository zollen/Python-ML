'''
Created on Aug. 10, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
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

label_column = 'survived'
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


for name in categorical_columns:
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


if False:
    param_grid = dict({ "n_estimators": [ 50, 75, 100, 150 ],
                       "max_depth": [ 2, 5, 10, 15, 20, 25, 30 ],
                       "learning_rate": [ 0.001, 0.01, 0.1, 0.3, 0.5 ],
                       "gamma": [ 0, 1, 2, 3, 4 ],
                       "min_child_weight": [ 0, 1, 2, 3, 5, 7, 10 ],
                       "subsample": [ 0.2, 0.4, 0.6, 0.8, 1.0 ],
                       "reg_lambda": [ 0, 0.5, 1, 2, 3 ],
                       "reg_alpha": [ 0, 0.5, 1 ],
                       "max_leaves": [0, 1, 2, 5, 10 ],
                       "max_bin": [ 128, 256, 512, 1024 ] })
    model = RandomizedSearchCV(estimator = XGBClassifier(), 
                        param_distributions = param_grid, n_jobs=50, n_iter=100)

    model.fit(train_df[all_features_columns], labels.squeeze())

    print("====================================================================================")
    print("Best Score: ", model.best_score_)
    print("Best n_estimators: ", model.best_estimator_.n_estimators)
    print("Best max_depth: ", model.best_estimator_.max_depth)
    print("Best eta: ", model.best_estimator_.learning_rate)
    print("Best gamma: ", model.best_estimator_.gamma)
    print("Best min_child_weight: ", model.best_estimator_.min_child_weight)
    print("Best subsample: ", model.best_estimator_.subsample)
    print("Best lambda: ", model.best_estimator_.reg_lambda)
    print("Best alpha: ", model.best_estimator_.reg_alpha)

    exit()
    
if False:    
    model = XGBClassifier(n_estimators=150, max_depth=11, min_child_weight=3, gamma=0.2, subsample=0.6, nthread=16)
else:
    model = XGBClassifier(n_estimators=150, max_depth=10, min_child_weight=5, gamma=0.7, subsample=1.0, reg_lambda = 0.02, reg_alpha = 0.005, nthread=16)

model.fit(train_df[all_features_columns], train_df[label_column])
print("XGB Score: ", model.score(train_df[all_features_columns], train_df[label_column]))

print("================= TRAINING DATA =====================")
preds = model.predict(train_df[all_features_columns])
print("Accuracy: ", round(accuracy_score(train_df[label_column], preds), 2))
print("Precision: ", round(precision_score(train_df[label_column], preds), 2))
print("Recall: ", round(recall_score(train_df[label_column], preds), 2))
print('AUC-ROC:', round(roc_auc_score(train_df[label_column], preds), 2))
print("Log Loss: ", round(log_loss(train_df[label_column], preds), 2))
print(confusion_matrix(train_df[label_column], preds))


print("================= TEST DATA =====================")
preds = model.predict(test_df[all_features_columns])
print("Accuracy: ", round(accuracy_score(test_df[label_column], preds), 2))
print("Precision: ", round(precision_score(test_df[label_column], preds), 2))
print("Recall: ", round(recall_score(test_df[label_column], preds), 2))
print('AUC-ROC:', round(roc_auc_score(test_df[label_column], preds), 2))
print("Log Loss: ", round(log_loss(test_df[label_column], preds), 2))
print(confusion_matrix(test_df[label_column], preds))
print(classification_report(test_df[label_column], preds))



