'''
Created on Aug. 10, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier 
from sklearn.model_selection import GridSearchCV
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


if False:
### OneHotEncoder
    cat_columns = []

    for name in categorical_columns: 
        keys = np.union1d(train_df[name].unique(), test_df[name].unique())
    
        for key in keys:
            func = lambda x : 1 if x == key else 0
            train_df[name + "." + str(key)] = train_df[name].apply(func)
            test_df[name + "." + str(key)] = test_df[name].apply(func)
            cat_columns.append(name + "." + str(key))
   
    all_features_columns = numeric_columns + cat_columns

    encoder = preprocessing.LabelBinarizer()
    train_df[label_column] = encoder.fit_transform(train_df[label_column].values)
    test_df[label_column] = encoder.transform(test_df[label_column].values)
else:
### LabelEncoder
    for name in categorical_columns + label_column:
        encoder = preprocessing.LabelEncoder()   
        keys = np.union1d(train_df[name].unique(), test_df[name].unique())
   
        if len(keys) == 2:
            encoder = preprocessing.LabelBinarizer()
        
        encoder.fit(keys)
        train_df[name] = encoder.transform(train_df[name].values)
        test_df[name] = encoder.transform(test_df[name].values)
   


if False:
    param_grid = [ 
                    {  
                       "n_estimators": [ 75, 100, 125, 150 ],
                       
                       "boosting_type": [ 'gbdt', 'rf' ],
                       "max_leaves": [ 4, 10, 12, 16  ],
                       "max_depth": [ -1, 2, 6, 12, 22 ],
                       "min_data_in_leaf": [ 15, 20, 25, 30 ],
                       "max_bin": [ 150, 200, 255, 275, 300 ]  
                    },
                    {
                       "boosting_type": [ 'dart' ],
                       "max_drop": [ 45, 50, 55, 60, 65, 70 ],
                       "drop_rate": [ 0.1, 0.2, 0.3, 0.4, 0.5 ],
                       "xgboost_dart_mode": [ False, True ],
                       "n_estimators": [ 75, 100, 125, 150 ],
                       "max_leaves": [ 4, 6, 8, 10, 12, 14, 16  ],
                       "max_depth": [ -1, 2, 4, 6, 8, 10, 12, 18, 20, 22 ],
                       "min_data_in_leaf": [ 15, 20, 25, 30 ],
                       "max_bin": [ 100, 150, 200, 255, 275, 300, 325 ]  
                    },
                    {
                       "boosting_type": [ 'gross' ],
                       "top_rate": [ 0.1, 0.15, 0.2, 0.25, 0.30 ],
                       "other_rate": [ 0.05, 0.1, 0.15, 0.20 ],
                       "n_estimators": [ 75, 100, 125, 150 ],
                       "max_leaves": [ 4, 6, 8, 10, 12, 14, 16  ],
                       "max_depth": [ -1, 2, 4, 6, 8, 10, 12, 18, 20, 22 ],
                       "min_data_in_leaf": [ 15, 20, 25, 30 ],
                       "max_bin": [ 100, 150, 200, 255, 275, 300, 325 ]  
                    }
                ]
    model = GridSearchCV(estimator = LGBMClassifier(), 
                        param_grid = param_grid, n_jobs=50)

    model.fit(train_df[all_features_columns], labels.squeeze())
    print("====================================================================================")
    print("Best Score: ", model.best_score_)
    print("Best Boost: ", model.boosting_type)
    print("Best n_estimators: ", model.best_estimator_.n_estimators)
    print("Best max_depth: ", model.best_estimator_.max_depth)
    print("Best MaxLeaves: ", model.best_estimator_.max_leaves)
    print("Best MinDataInLeaf: ", model.best_estimator_.min_data_in_leaf)
    print("Best MaxBin: ", model.best_estimator_.max_bin)
    if model.boosting_type == 'dart':
        print("Best MaxDrop: ", model.best_estimator_.max_drop)
        print("Best DropRate: ", model.best_estimator_.drop_rate)
        print("Best XGBoost: ", model.best_estimator_.xgboost_dart_mode)
    if model.boosting_type == 'gross':
        print("Best TopRate: ", model.best_estimator_.top_rate)
        print("Best OtherRate: ", model.best_estimator_.other_rate)
   
    exit()

if False:
    model = LGBMClassifier(num_leaves=32, max_depth=16, min_data_in_leaf=10)
else:
    model = LGBMClassifier(n_estimators=50, max_depth=16, 
                        max_leaves=32, min_data_in_leaf=25)
    
model.fit(train_df[all_features_columns], train_df[label_column])
print("LightGBM Score: ", model.score(train_df[all_features_columns], train_df[label_column]))
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

