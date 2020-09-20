'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import titanic_kaggle.lib.titanic_lib as tb
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


def captures(df):
    df['Room']  = df['Cabin'].apply(tb.captureRoom)
    df['Cabin'] = df['Cabin'].apply(tb.captureCabin) 
    df['Title'] = df['Title'].map(tb.titles)
    df['Sex'] = df['Sex'].map(tb.sexes)
    df['Embarked'] = df['Embarked'].map(tb.embarkeds)
    df['Cabin'] = df['Cabin'].map(tb.cabins)
    
def typecast(df):
    df['Room'] = df['Room'].astype('int64')
    df['Age'] = df['Age'].astype('int64')
    df['Sex'] = df['Sex'].astype('int64')
    df['Embarked'] = df['Embarked'].astype('int64')
    df['Cabin'] = df['Cabin'].astype('int64')
    df['Title'] = df['Title'].astype('int64')
    df['Fare'] = df['Fare'].astype('int64')


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

train_df.loc[train_df['Cabin'] == 'T', 'Cabin'] = 'A'
train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25
test_df.loc[test_df['Embarked'].isna() == True, 'Embarked'] = 'S' 
test_df.loc[test_df['Fare'].isna() == True, 'Fare'] = 7.25

tb.reeigneeringTitle(train_df)
tb.reeigneeringTitle(test_df)

captures(train_df)
captures(test_df)

tb.fill_by_regression(train_df, train_df, 'Age', [ 'Title', 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Pclass', 'Embarked' ])
tb.fill_by_classification(train_df, train_df, 'Cabin', [ 'Title', 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Age', 'Pclass', 'Embarked' ])

tb.fill_by_regression(tb.combine(train_df, test_df), test_df, 'Age', [ 'Title', 'SibSp', 'Parch', 'Fare', 'Sex', 'Pclass', 'Embarked' ])
tb.fill_by_classification(tb.combine( train_df, test_df ), test_df, 'Cabin', [ 'Title', 'Age', 'SibSp', 'Parch', 'Embarked', 'Sex', 'Fare', 'Pclass' ])

typecast(train_df)
typecast(test_df)

train_df.drop(columns = ['Name', 'Ticket'], inplace = True)
test_df.drop(columns = ['Name', 'Ticket'], inplace = True)

outputs = ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked',  'Pclass', 'Cabin', 'Room', 'Survived' ]
train_df[outputs].to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
outputs = ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked',  'Pclass', 'Cabin', 'Room' ]
test_df[outputs].to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)

print("Done")
