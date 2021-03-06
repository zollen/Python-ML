'''
Created on Aug. 26, 2020

@author: zollen
'''

import os
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings


warnings.filterwarnings('ignore')

SEED = 87

pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)
np.random.seed(SEED)

label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Name', 'Ticket' ]
numeric_columns = [ 'Age', 'SibSp', 'Parch', 'Fare' ]
categorical_columns = [ 'Sex', 'Embarked',  'Pclass' ]
all_features_columns = numeric_columns + categorical_columns 

def score_model(observed, predicted):
    accuracy = accuracy_score(observed, predicted)
    precision = precision_score(observed, predicted)
    recall = recall_score(observed, predicted)
    auc = roc_auc_score(observed, predicted)
    loss = log_loss(observed, predicted)
    
    return accuracy, precision, recall, auc, loss

def display_score(accur, prec, recal, auc, loss):
    print("Accuracy: %0.2f    Precision: %0.2f    Recall: %0.2f    AUC: %0.2f    Loss: %0.2f" %
          (accur, prec, recal, auc, loss))
    
def show_score (observed, predicted):
    print("Accuracy: %0.2f    Precision: %0.2f    Recall: %0.2f    AUC: %0.2f    Loss: %0.2f" %
           (score_model(observed, predicted)))

def groups_generator(total, k, seed):
    
    groups = []
    
    done = False
    while done == False:
        for item in range(0, k):
            groups.append(item)
            if len(groups) == total:
                done = True
                break
            
    random.seed(seed)
    random.shuffle(groups)
    
    return groups

def cross_validation(df, labels, k, repeats, seeds):
    
    tbl = {}
    tbl['accuracy'], tbl['precision'], tbl['recall'], tbl['auc'],tbl['loss'] = [], [], [], [], []
        
    for i in range(0, repeats):
        kfold = GroupKFold(n_splits = k)
        groups = groups_generator(len(df), k, seeds[i])
        for train_index, test_index in kfold.split(df, groups = groups):
            x_train, y_train = df.iloc[train_index], labels.iloc[train_index]
            x_test, y_test = df.iloc[test_index], labels.iloc[test_index]
            classifier = LogisticRegression()
            classifier.fit(x_train, y_train)
            preds = classifier.predict(x_test)
            a1, a2, a3, a4, a5 = score_model(y_test, preds)
            tbl['accuracy'].append(a1)
            tbl['precision'].append(a2)
            tbl['recall'].append(a3)
            tbl['auc'].append(a4)
            tbl['loss'].append(a5)
            
    a1, a2, a3, a4, a5 = np.mean(tbl['accuracy']), np.mean(tbl['precision']), np.mean(tbl['recall']), np.mean(tbl['auc']), np.mean(tbl['loss'])
    
    display_score(a1, a2, a3, a4, a5)
    
    return a1, a2, a3, a4, a5

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'))


cat_columns = []

for name in categorical_columns:
    encoder = preprocessing.LabelEncoder()   
    keys = train_df[name].unique()

   
    if len(keys) == 2:
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    train_df[name] = encoder.transform(train_df[name].values)
    test_df[name] = encoder.transform(test_df[name].values)
    
    for key in keys:
        func = lambda x : 1 if x == key else 0
        train_df[name + '.' + str(key)] = train_df[name].apply(func)
        test_df[name + '.' + str(key)] = train_df[name].apply(func)
        cat_columns.append(name + '.' + str(key))
        

categorical_columns = cat_columns


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
    model = RandomizedSearchCV(estimator = LogisticRegression(), 
                        param_distributions = param_grid, n_jobs=-1, n_iter=100)

    model.fit(train_df[all_features_columns], train_df[label_column].squeeze())

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


model = LogisticRegression(max_iter=500, solver='lbfgs')
model.fit(train_df[all_features_columns], train_df[label_column].squeeze())
preds = model.predict(train_df[all_features_columns])
target = train_df[label_column].squeeze()
show_score(target, preds)
print(confusion_matrix(target, preds))
print(classification_report(target, preds))

print("================== CROSS VALIDATION ==================")
kfold = RepeatedStratifiedKFold(n_splits = 9, n_repeats = 5, random_state = 87)
results = cross_val_score(LogisticRegression(), 
                          train_df, target, cv = kfold)
print("9-Folds Cross Validation Accuracy: %0.2f" % results.mean())

print("LogisticRegression Score: ", model.score(train_df[all_features_columns], train_df[label_column]))


tbl = {}
tbl['accuracy'], tbl['precision'], tbl['recall'], tbl['auc'],tbl['loss'] = [], [], [], [], []
        
for k in range(4, 10):
    a1, a2, a3, a4, a5 = cross_validation(train_df[all_features_columns], target, k, 3, [0, 23, 87])
    tbl['accuracy'].append(a1)
    tbl['precision'].append(a2)
    tbl['recall'].append(a3)
    tbl['auc'].append(a4)
    tbl['loss'].append(a5)
print("============================= AVERAGE ===========================")
display_score(np.mean(tbl['accuracy']), np.mean(tbl['precision']), np.mean(tbl['recall']), 
              np.mean(tbl['auc']), np.mean(tbl['loss']))

if False:
    preds = model.predict(test_df)
    results = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': preds })
    results.to_csv(os.path.join(PROJECT_DIR, 'data/results.csv'), index = False)


