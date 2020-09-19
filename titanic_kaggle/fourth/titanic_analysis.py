'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import titanic_kaggle.fourth.titanic_lib as tb
import warnings

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(0)


label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Ticket', 'Room' ]
numeric_columns = [ 'Age', 'SibSp', 'Parch', 'Fare' ]
categorical_columns = [ 'Title', 'Sex', 'Embarked', 'Pclass'  ]
all_features_columns = numeric_columns + categorical_columns 




PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))


print(train_df.info())
print("=============== STATS ===================")
print(train_df.describe())
print("============== Training Total NULL ==============")
print(train_df.isnull().sum())
print("============== SKEW =====================")
print(train_df.skew())


tb.reeigneeringTitle(train_df)
tb.reeigneeringTitle(test_df)
train_df['Cabin'] = train_df['Cabin'].apply(tb.captureCabin) 
test_df['Cabin'] = test_df['Cabin'].apply(tb.captureCabin) 

titles = {
    "Army": 0,
    "Doctor": 1,
    "Nurse": 2,
    "Clergy": 3,
    "Baronness": 4,
    "Baron": 5,
    "Mr": 6,
    "Mrs": 7,
    "Miss": 8,
    "Master": 9,
    "Girl": 10,
    "Boy": 11,
    "GramPa": 12,
    "GramMa": 13
    }

train_df['Title'] = train_df['Title'].map(titles)
test_df['Title'] = test_df['Title'].map(titles)

def captureRoom(rec):
    
    if str(rec['Cabin']) != 'nan':
        x = re.findall("[a-zA-Z]+[0-9]{1}", rec['Cabin'])
        if len(x) == 0:
            x = re.findall("[a-zA-Z]{1}", rec['Cabin'])
        y = re.findall("[0-9]+", rec['Cabin'])
        if len(y) == 0:
            y = [ 0 ]

        rec['Cabin'] = x[0][0]
        rec['Room'] = int(str(y[0]))
        
    return rec

train_df  = train_df.apply(captureRoom, axis=1)
test_df = test_df.apply(captureRoom, axis=1)


tb.fill_by_classification(train_df, train_df, 'Embarked', [ 'Title', 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Pclass' ])    
tb.fill_by_regression(train_df, train_df, 'Age', [ 'Title', 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Pclass', 'Embarked' ])
tb.fill_by_classification(train_df, train_df, 'Cabin', [ 'Title', 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Age', 'Pclass', 'Embarked' ])
tb.fill_by_classification(train_df, train_df, 'Room', [ 'Title', 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Age', 'Pclass', 'Embarked' ])


allsamples = pd.concat([ train_df, test_df ])
tb.fill_by_regression(allsamples[allsamples['Age'].isna() == False], test_df, 'Fare', [ 'Title', 'Age', 'SibSp', 'Parch', 'Embarked', 'Sex', 'Pclass' ])
tb.fill_by_regression(pd.concat([ train_df, test_df ]), test_df, 'Age', [ 'Title', 'SibSp', 'Parch', 'Fare', 'Sex', 'Pclass', 'Embarked' ])
tb.fill_by_classification(pd.concat([ train_df, test_df ]), test_df, 'Cabin', [ 'Title', 'Age', 'SibSp', 'Parch', 'Embarked', 'Sex', 'Fare', 'Pclass' ])
tb.fill_by_classification(pd.concat([ train_df, test_df ]), test_df, 'Room', [ 'Title', 'Age', 'SibSp', 'Parch', 'Embarked', 'Sex', 'Fare', 'Pclass' ])

train_df['Room'] = train_df['Room'].astype('int32')
test_df['Room'] = test_df['Room'].astype('int32')

train_df.drop(columns = [])

outputs = ['PassengerId', 'Name', 'Ticket', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked',  'Pclass', 'Cabin', 'Room', 'Survived' ]
train_df[outputs].to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
outputs = ['PassengerId', 'Name', 'Ticket', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked',  'Pclass', 'Cabin', 'Room' ]
test_df[outputs].to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)

print("Done")
