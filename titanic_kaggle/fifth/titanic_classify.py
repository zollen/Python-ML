'''
Created on Aug. 26, 2020

@author: zollen
'''

import os
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import svm
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
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
np.random.seed(SEED)

label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Name' ]
numeric_columns = [ 'Age', 'Fare', 'Title', 'Size', 'Ticket', 'Sex', 
                   'Embarked',  'Pclass', 'Logistic' ]
categorical_columns =  []
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
            classifier = svm.SVC(kernel='rbf', gamma ='auto', C=1.0)
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

if len(categorical_columns) > 0:
    ## Don't build your own categorical columns, it won't work
    train_df = pd.get_dummies(train_df, columns = categorical_columns)
    test_df = pd.get_dummies(test_df, columns = categorical_columns)

    train_uni = set(train_df.columns).symmetric_difference(numeric_columns + ['PassengerId'] + label_column)
    test_uni = set(test_df.columns).symmetric_difference(numeric_columns + ['PassengerId'])

    categorical_columns = list(test_uni)
    all_features_columns = numeric_columns + categorical_columns 
    

if len(numeric_columns) > 0:
    num_columns = []
    for name in numeric_columns:
        if train_df[name].max() > 1:
            num_columns.append(name)
            
    scaler = MinMaxScaler()
    train_df[num_columns] = scaler.fit_transform(train_df[num_columns])
    test_df[num_columns] = scaler.transform(test_df[num_columns])

print(all_features_columns)



 
pca = PCA(n_components = 3)
ttrain_df = pd.DataFrame(pca.fit_transform(train_df[all_features_columns]))
ttest_df = pd.DataFrame(pca.transform(test_df[all_features_columns]))
    
if False:
    param_grid = dict({ 
                       "C": range(0, 500),
                       "degree": range(2, 20),
                       "kernel": [ 'linear', 'poly', 'sigmoid', 'rbf' ],
                       "gamma": [ 'auto', 'scale' ],
                       "max_iter": range(50, 5000)
                       })
    model = RandomizedSearchCV(estimator = svm.SVC(), 
                        param_distributions = param_grid, n_jobs=-1, n_iter=100)

    model.fit(ttrain_df, train_df[label_column])

    print("====================================================================================")
    print("Best Score: ", model.best_score_)
    print("Best C: ", model.best_estimator_.C)
    print("Best kernel: ", model.best_estimator_.kernel)
    print("Best degree: ", model.best_estimator_.degree)
    print("Best gamma: ", model.best_estimator_.gamma)
    print("Best max_iter: ", model.best_estimator_.max_iter)

    exit()


model = svm.SVC(kernel='rbf', gamma ='auto', C=1.0)
model.fit(ttrain_df, train_df[label_column].squeeze())
preds = model.predict(ttrain_df)
target = train_df[label_column].squeeze()
show_score(target, preds)
print(confusion_matrix(target, preds))
print(classification_report(target, preds))

print("================== CROSS VALIDATION ==================")
kfold = RepeatedStratifiedKFold(n_splits = 9, n_repeats = 5, random_state = 87)
results = cross_val_score(svm.SVC(kernel='rbf', gamma ='auto', C=1.0), 
                          ttrain_df, target, cv = kfold)
print("9-Folds Cross Validation Accuracy: %0.2f" % results.mean())

print("PCA/SVM Score: ", model.score(ttrain_df, train_df[label_column]))


tbl = {}
tbl['accuracy'], tbl['precision'], tbl['recall'], tbl['auc'],tbl['loss'] = [], [], [], [], []
        
for k in range(4, 10):
    a1, a2, a3, a4, a5 = cross_validation(ttrain_df, target, k, 3, [0, 23, 87])
    tbl['accuracy'].append(a1)
    tbl['precision'].append(a2)
    tbl['recall'].append(a3)
    tbl['auc'].append(a4)
    tbl['loss'].append(a5)
print("============================= AVERAGE ===========================")
display_score(np.mean(tbl['accuracy']), np.mean(tbl['precision']), np.mean(tbl['recall']), 
              np.mean(tbl['auc']), np.mean(tbl['loss']))

if True:
    preds = model.predict(ttest_df)
    results = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': preds })
    results.to_csv(os.path.join(PROJECT_DIR, 'data/results.csv'), index = False)


