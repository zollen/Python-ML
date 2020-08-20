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
from sklearn.model_selection import RandomizedSearchCV
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

if False:
    param_grid = dict({ "n_estimators": [ 75, 100, 150, 200 ],
                       "boosting_type": [ 'gbdt', 'dart', 'goss', 'rf' ],
                       "max_leaves": [ 4, 6, 8, 16  ],
                       "max_depth": [ -1, 6, 12, 24 ],
                       "learning_rate": [ 0.0001, 0.001, 0.01, 1 ],
                       "subsample_for_bin": [ 150000, 200000, 250000, 300000 ],
                       "min_data_in_leaf": [ 15, 20, 25, 30 ],
                       "max_bin": [ 60, 100, 255, 500 ]  })
    model = RandomizedSearchCV(estimator = LGBMClassifier(), 
                        param_distributions = param_grid, n_jobs=50, n_iter=100)

    model.fit(train_df[all_features_columns], labels.squeeze())
    print("====================================================================================")
    print("Best Score: ", model.best_score_)
    print("Best n_estimators: ", model.best_estimator_.n_estimators)
    print("Best max_depth: ", model.best_estimator_.max_depth)
    print("Best LearningRate: ", model.best_estimator_.learning_rate)
    print("Best MaxLeaves: ", model.best_estimator_.max_leaves)
    print("Best SubSampleForBin: ", model.best_estimator_.subsample_for_bin)
    print("Best MinDataInLeaf: ", model.best_estimator_.min_data_in_leaf)
    print("Best MaxBin: ", model.best_estimator_.max_bin)
   
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

