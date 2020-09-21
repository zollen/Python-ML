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
    df.loc[df['Age'] < 7, 'Age'] = 3
    df.loc[(df['Age'] < 15) & (df['Age'] >= 7), 'Age'] = 10
    df.loc[(df['Age'] < 20) & (df['Age'] >= 15), 'Age'] = 18
    df.loc[(df['Age'] < 30) & (df['Age'] >= 20), 'Age'] = 25
    df.loc[(df['Age'] < 40) & (df['Age'] >= 30), 'Age'] = 35
    df.loc[(df['Age'] < 50) & (df['Age'] >= 40), 'Age'] = 45
    df.loc[(df['Age'] < 60) & (df['Age'] >= 50), 'Age'] = 55
    df.loc[(df['Age'] < 70) & (df['Age'] >= 60), 'Age'] = 65
    df.loc[df['Age'] >= 70, 'Age'] = 75
    
def binFare(df):
    df.loc[df['Fare'] < 10, 'Fare'] = 5
    df.loc[(df['Fare'] < 20) & (df['Fare'] >= 10), 'Fare'] = 15
    df.loc[(df['Fare'] < 30) & (df['Fare'] >= 20), 'Fare'] = 25
    df.loc[(df['Fare'] < 40) & (df['Fare'] >= 30), 'Fare'] = 35
    df.loc[(df['Fare'] < 50) & (df['Fare'] >= 40), 'Fare'] = 45
    df.loc[(df['Fare'] < 60) & (df['Fare'] >= 50), 'Fare'] = 55
    df.loc[(df['Fare'] < 70) & (df['Fare'] >= 60), 'Fare'] = 65
    df.loc[(df['Fare'] < 80) & (df['Fare'] >= 70), 'Fare'] = 75
    df.loc[(df['Fare'] < 90) & (df['Fare'] >= 80), 'Fare'] = 85
    df.loc[(df['Fare'] < 100) & (df['Fare'] >= 90), 'Fare'] = 95
    df.loc[(df['Fare'] < 120) & (df['Fare'] >= 100), 'Fare'] = 110
    df.loc[(df['Fare'] < 200) & (df['Fare'] >= 120), 'Fare'] = 160
    df.loc[(df['Fare'] < 300) & (df['Fare'] >= 200), 'Fare'] = 250
    df.loc[df['Fare'] >= 300, 'Fare'] = 500


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
columns = [ 'Title', 'Sex', 'Pclass', 'Embarked', 'Size', 'Age', 'Fare', 'Cabin' ]
tb.reeigneeringSurvProb(ttrain_df, columns)
tb.reeigneeringSurvProb(ttest_df, columns )

train_df['Chance'] = ttrain_df['Chance']
test_df['Chance'] = ttest_df['Chance']

train_df['Cabin'] = train_df['Cabin'] * 1000 + train_df['Room']
test_df['Cabin'] = test_df['Cabin'] * 1000 + test_df['Room']


train_df.drop(columns = ['Name', 'Ticket', 'Title', 'Size', 'Room'], inplace = True)
test_df.drop(columns = ['Name', 'Ticket', 'Title', 'Size', 'Room'], inplace = True)

outputs = ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked',  'Pclass', 'Cabin', 'Survived' ]
train_df[outputs].to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
outputs = ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked',  'Pclass', 'Cabin' ]
test_df[outputs].to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)

print("Done")
