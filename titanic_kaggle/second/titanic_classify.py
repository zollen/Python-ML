'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
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

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(0)
sb.set_style('whitegrid')

label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Ticket' ]
numeric_columns = [ 'SibSp', 'Parch', 'Fare' ]
categorical_columns = [ 'Title', 'Pclass', 'Embarked' ]
all_features_columns = numeric_columns + categorical_columns 


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
            classifier = QuadraticDiscriminantAnalysis(reg_param = 0.1, tol = 0.0003)
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
    


PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'))



print(train_df.head())
target = train_df[label_column]
train_df.drop(columns = label_column, inplace = True)


for name in categorical_columns:
    encoder = preprocessing.LabelEncoder()
    keys = np.union1d(train_df[name].unique(), test_df[name].unique())
    
    if len(keys) == 2:
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    train_df[name] = encoder.transform(train_df[name].values)
    test_df[name] = encoder.transform(test_df[name].values)


train_df = pd.get_dummies(train_df, columns=categorical_columns)
test_df = pd.get_dummies(test_df, columns=categorical_columns)

scaler = MinMaxScaler()
train_df[numeric_columns] = scaler.fit_transform(train_df[numeric_columns])
test_df[numeric_columns] = scaler.transform(test_df[numeric_columns])


if False:
    param_grid = dict({ "reg_param": np.arange(0.0, 1.0, 0.1),
                       "tol": [ 3.0e-4, 2.0e-4, 1.0e-4, 3.0e-3, 2.0e-3, 1.0e-3, 3.0e-2, 2.0e-2, 1.0e-2   ]
                        })
    model = RandomizedSearchCV(estimator = QuadraticDiscriminantAnalysis(), 
                        param_distributions = param_grid, n_jobs=-1, n_iter=500)

    model.fit(train_df, target)

    print("====================================================================================")
    print("Best Score: ", model.best_score_)
    print("Best reg_param: ", model.best_estimator_.reg_param)
    print("Best tol: ", model.best_estimator_.tol)

    exit()

model = QuadraticDiscriminantAnalysis(reg_param = 0.1, tol = 0.0003)
model.fit(train_df, target)
print("================= TRAINING DATA =====================")
preds = model.predict(train_df)
show_score(target, preds)
print(confusion_matrix(target, preds))
print(classification_report(target, preds))

print("================== CROSS VALIDATION ==================")
kfold = RepeatedStratifiedKFold(n_splits = 9, n_repeats = 5, random_state = 87)
results = cross_val_score(QuadraticDiscriminantAnalysis(reg_param = 0.1, tol = 0.0003), 
                          train_df, target, cv = kfold)
print("9-Folds Cross Validation Accuracy: %0.2f" % results.mean())

tbl = {}
tbl['accuracy'], tbl['precision'], tbl['recall'], tbl['auc'],tbl['loss'] = [], [], [], [], []
        
for k in range(4, 10):
    a1, a2, a3, a4, a5 = cross_validation(train_df, target, k, 3, [0, 23, 87])
    tbl['accuracy'].append(a1)
    tbl['precision'].append(a2)
    tbl['recall'].append(a3)
    tbl['auc'].append(a4)
    tbl['loss'].append(a5)
print("============================= AVERAGE ===========================")
display_score(np.mean(tbl['accuracy']), np.mean(tbl['precision']), np.mean(tbl['recall']), 
              np.mean(tbl['auc']), np.mean(tbl['loss']))