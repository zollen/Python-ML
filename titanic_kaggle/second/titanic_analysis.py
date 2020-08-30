'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import re
import seaborn as sb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
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
numeric_columns = [ 'Age', 'SibSp', 'Parch', 'Fare' ]
categorical_columns = [ 'Title', 'Sex', 'Pclass', 'Embarked' ]
all_features_columns = numeric_columns + categorical_columns 

title_category = {
        "Capt.": "Army",
        "Col.": "Army",
        "Major.": "Army",
        "Jonkheer.": "Baronness",
        "Don.": "Baron",
        "Sir.": "Baron",
        "Dr.": "Doctor",
        "Rev.": "Clergy",
        "Countess.": "Baronness",
        "Dona.": "Baronness",
        "Mme.": "Mrs",
        "Mlle.": "Miss",
        "Ms.": "Mrs",
        "Mr.": "Mr",
        "Mrs.": "Mrs",
        "Miss.": "Miss",
        "Master.": "Master",
        "Lady.": "Baronness",
        "Girl.": "Girl",
        "Nurse.": "Nurse"
    }

def cross_validation(df, labels, k, repeats, seeds):
    
    tbl = {}
    tbl['accuracy'], tbl['precision'], tbl['recall'], tbl['auc'],tbl['loss'] = [], [], [], [], []
        
    for i in range(0, repeats):
        kfold = KFold(n_splits = k, shuffle = True, random_state = seeds[i])
        for train_index, test_index in kfold.split(df):
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
    
def map_title(rec):
    title = title_category[rec['Title']]
    sex = rec['Sex']
    
    if title == 'Doctor' and sex == 'male':
        return 'Doctor'
    else:
        if title == 'Doctor' and sex == 'female':
            return 'Nurse'

    if str(rec['Age']) != 'nan' and title == 'Miss' and rec['Age'] < 16:
        return 'Girl'
    
    if str(rec['Age']) == 'nan' and title == 'Miss' and rec['Parch'] > 0:
        return 'Girl'
    
    return title
    
def process(df):
    
    df['Title'] = df['Name'].apply(lambda x : re.search('[a-zA-Z]+\\.', x).group(0))
    df['Title'] = df.apply(map_title, axis = 1)

    df.drop(columns=[ 'Name', 'Cabin', 'Ticket'], inplace=True)

    df.loc[df['Embarked'].isna() == True, 'Embarked'] = 'C'
    df.loc[df['Fare'].isna() == True, 'Fare'] = 7.25

    for title in set(title_category.values()):
        df.loc[((df['Age'].isna() == True) & (df['Title'] == title)), 'Age'] = df.loc[((df['Age'].isna() == False) & (df['Title'] == title)), 'Age'].median()

    df['Age'] = df['Age'].astype('int32')

    print(df.groupby(['Title'])['Age'].median())

    for name in categorical_columns:
        encoder = preprocessing.LabelEncoder()   
        keys = df[name].unique()

        if len(keys) == 2:
            encoder = preprocessing.LabelBinarizer()
        
        encoder.fit(keys)
        df[name] = encoder.transform(df[name].values)
    
 
    labels = df[label_column]

    ## Drop Age and Sex but Keep Title improve score
    df.drop(columns=label_column, inplace=True)    
    df.drop(columns=[ 'Title' ], inplace=True)

    df = pd.get_dummies(train_df, columns=[ 'Pclass', 'Embarked' ])



    scaler = MinMaxScaler()   
    cols = list(df.columns)
    cols.remove('PassengerId')
    df[cols] = scaler.fit_transform(df[cols]) 
    
    return df, labels


PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

train_df, tlabels = process(train_df)

print(train_df.head())


if False:
    param_grid = dict({ "reg_param": np.arange(0.0, 1.0, 0.1),
                       "tol": [ 3.0e-4, 2.0e-4, 1.0e-4, 3.0e-3, 2.0e-3, 1.0e-3, 3.0e-2, 2.0e-2, 1.0e-2   ]
                        })
    model = RandomizedSearchCV(estimator = QuadraticDiscriminantAnalysis(), 
                        param_distributions = param_grid, n_jobs=-1, n_iter=500)

    model.fit(train_df, tlabels)

    print("====================================================================================")
    print("Best Score: ", model.best_score_)
    print("Best reg_param: ", model.best_estimator_.reg_param)
    print("Best tol: ", model.best_estimator_.tol)

    exit()

model = QuadraticDiscriminantAnalysis(reg_param = 0.1, tol = 0.0003)
model.fit(train_df, tlabels)
print("================= TRAINING DATA =====================")
preds = model.predict(train_df)
show_score(tlabels, preds)
print(confusion_matrix(tlabels, preds))
print(classification_report(tlabels, preds))

print("================== CROSS VALIDATION ==================")
kfold = RepeatedStratifiedKFold(n_splits = 9, random_state = 87)
results = cross_val_score(QuadraticDiscriminantAnalysis(reg_param = 0.1, tol = 0.0003), train_df, tlabels, cv = kfold)
print("9-Folds Cross Validation Accuracy: %0.2f" % results.mean())

tbl = {}
tbl['accuracy'], tbl['precision'], tbl['recall'], tbl['auc'],tbl['loss'] = [], [], [], [], []
        
for k in range(4, 10):
    a1, a2, a3, a4, a5 = cross_validation(train_df, tlabels, k, 3, [0, 23, 87])
    tbl['accuracy'].append(a1)
    tbl['precision'].append(a2)
    tbl['recall'].append(a3)
    tbl['auc'].append(a4)
    tbl['loss'].append(a5)
print("============================= AVERAGE ===========================")
display_score(np.mean(tbl['accuracy']), np.mean(tbl['precision']), np.mean(tbl['recall']), 
              np.mean(tbl['auc']), np.mean(tbl['loss']))