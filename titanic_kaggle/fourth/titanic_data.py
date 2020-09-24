'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
import titanic_kaggle.lib.titanic_lib as tb
import warnings

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(0)
pp = pprint.PrettyPrinter(indent=3) 

def binAge(df):
    df.loc[df['Age'] < 14, 'Age'] = 7
    df.loc[(df['Age'] < 19) & (df['Age'] >= 14), 'Age'] = 17
    df.loc[(df['Age'] < 22) & (df['Age'] >= 19), 'Age'] = 20
    df.loc[(df['Age'] < 25) & (df['Age'] >= 22), 'Age'] = 24
    df.loc[(df['Age'] < 28) & (df['Age'] >= 25), 'Age'] = 27
    df.loc[(df['Age'] < 31.8) & (df['Age'] >= 28), 'Age'] = 30
    df.loc[(df['Age'] < 36) & (df['Age'] >= 31.8), 'Age'] = 34
    df.loc[(df['Age'] < 41) & (df['Age'] >= 36), 'Age'] = 38
    df.loc[(df['Age'] < 50) & (df['Age'] >= 41), 'Age'] = 46
    df.loc[(df['Age'] < 80) & (df['Age'] >= 50), 'Age'] = 70
    df.loc[df['Age'] >= 80, 'Age'] = 80
    
def binFare(df):
    df.loc[df['Fare'] < 7.229, 'Fare'] = 4
    df.loc[(df['Fare'] < 7.75) & (df['Fare'] >= 7.229), 'Fare'] = 6
    df.loc[(df['Fare'] < 7.896) & (df['Fare'] >= 7.75), 'Fare'] = 7
    df.loc[(df['Fare'] < 8.05) & (df['Fare'] >= 7.896), 'Fare'] = 8
    df.loc[(df['Fare'] < 10.5) & (df['Fare'] >= 8.05), 'Fare'] = 9
    df.loc[(df['Fare'] < 13) & (df['Fare'] >= 10.5), 'Fare'] = 11
    df.loc[(df['Fare'] < 15.85) & (df['Fare'] >= 13), 'Fare'] = 14
    df.loc[(df['Fare'] < 24) & (df['Fare'] >= 15.85), 'Fare'] = 20
    df.loc[(df['Fare'] < 26.55) & (df['Fare'] >= 24), 'Fare'] = 25
    df.loc[(df['Fare'] < 33.308) & (df['Fare'] >= 26.55), 'Fare'] = 30
    df.loc[(df['Fare'] < 55.9) & (df['Fare'] >= 33.308), 'Fare'] = 45
    df.loc[(df['Fare'] < 83.158) & (df['Fare'] >= 55.9), 'Fare'] = 75
    df.loc[(df['Fare'] < 512.329) & (df['Fare'] >= 83.158), 'Fare'] = 100
    df.loc[df['Fare'] >= 300, 'Fare'] = 150


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

train_df.loc[train_df['Cabin'] == 'T', 'Cabin'] = 'A'
train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25
test_df.loc[test_df['Embarked'].isna() == True, 'Embarked'] = 'S' 
test_df.loc[test_df['Fare'].isna() == True, 'Fare'] = 7.25

tb.reeigneeringTitle(train_df)
tb.reeigneeringTitle(test_df)

tb.captures(train_df)
tb.captures(test_df)

tb.fill_by_regression(train_df, train_df, 'Age', [ 'Title', 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Pclass', 'Embarked' ])
tb.fill_by_classification(train_df, train_df, 'Cabin', [ 'Title', 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Age', 'Pclass', 'Embarked' ])

tb.fill_by_regression(tb.combine(train_df, test_df), test_df, 'Age', [ 'Title', 'SibSp', 'Parch', 'Fare', 'Sex', 'Pclass', 'Embarked' ])
tb.fill_by_classification(tb.combine( train_df, test_df ), test_df, 'Cabin', [ 'Title', 'Age', 'SibSp', 'Parch', 'Embarked', 'Sex', 'Fare', 'Pclass' ])

tb.reeigneeringFamilySize(train_df)
tb.reeigneeringFamilySize(test_df)

tb.typecast(train_df)
tb.typecast(test_df)

ttrain_df = train_df.copy()
ttest_df = test_df.copy()
binFare(ttrain_df)
binAge(ttrain_df)
binFare(ttest_df)
binAge(ttest_df)

                 
tbl = {
    "Title": np.union1d(ttrain_df['Title'].unique(), ttest_df['Title'].unique()),
    "Age": np.union1d(ttrain_df['Age'].unique(), ttest_df['Age'].unique()),
    "Sex": ttrain_df['Sex'].unique(),
    "Pclass": ttrain_df['Pclass'].unique(),
    "Cabin": np.union1d(ttrain_df['Cabin'].unique(), ttest_df['Cabin'].unique()),
    "Size": np.union1d(ttrain_df['Size'].unique(), ttest_df['Size'].unique()),
    "Fare": np.union1d(ttrain_df['Fare'].unique(), ttest_df['Fare'].unique()),
    "Embarked": ttrain_df['Embarked'].unique()    
    }

pp.pprint(tbl)

tb.navieBayes(ttrain_df, tbl)
columns = [ 'Title', 'Age', 'Sex', 'Pclass', 'Cabin', 'Size', 'Fare', 'Embarked' ]
coeffs = { "Title": 1.7516, "Age": 0.6860, "Sex": 0.3505, "Pclass": 0.1558, 
         "Cabin": -0.3116, "Size": 1.3449, "Fare": 0.5797, "Embarked": 0.8154 }
tb.reeigneeringSurvProb(ttrain_df, coeffs, columns)
tb.reeigneeringSurvProb(ttest_df, coeffs, columns )

train_df['Chance'] = ttrain_df['Chance']
test_df['Chance'] = ttest_df['Chance']

train_df['Cabin'] = train_df['Cabin'] * 1000 + train_df['Room']
test_df['Cabin'] = test_df['Cabin'] * 1000 + test_df['Room']


train_df.drop(columns = ['Name', 'Ticket', 'Title', 'Size', 'Room'], inplace = True)
test_df.drop(columns = ['Name', 'Ticket', 'Title', 'Size', 'Room'], inplace = True)

outputs = ['PassengerId', 'Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked',  'Pclass', 'Cabin', 'Chance' ]
train_df[outputs].to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
outputs = ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked',  'Pclass', 'Cabin', 'Chance' ]
test_df[outputs].to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)

print("Done")
