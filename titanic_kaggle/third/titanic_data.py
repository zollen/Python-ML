'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
import seaborn as sb
import warnings
import titanic_kaggle.lib.titanic_lib as tb

warnings.filterwarnings('ignore')

    
    

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(0)
sb.set_style('whitegrid')
pp = pprint.PrettyPrinter(indent=3) 

label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Ticket' ]
numeric_columns = [ 'Age', 'Fare' ]
categorical_columns = [ 'Sex', 'Title', 'Pclass', 'Embarked', 'Cabin' ]
all_features_columns = numeric_columns + categorical_columns 

## TO DO LIST
## 1. PCA and MeanShift analysis for Age and Fare
## 2. Mutli-steps group based medians approximation for Cabin
## 3. Rich women and Alive girl
def fillAge(src_df, dest_df):
    
    ages = src_df.groupby(['Title', 'Sex', 'SibSp', 'Parch'])['Age'].median()

    for index, value in ages.items():
            dest_df.loc[(dest_df['Age'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Sex'] == index[1]) &
                 (dest_df['SibSp'] == index[2]) &
                 (dest_df['Parch'] == index[3]), 'Age'] = value
                 
    ages = src_df.groupby(['Title', 'Sex', 'SibSp'])['Age'].median()

    for index, value in ages.items():
        dest_df.loc[(dest_df['Age'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Sex'] == index[1]) &
                 (dest_df['SibSp'] == index[2]), 'Age'] = value
               
    ages = src_df.groupby(['Title', 'Sex'])['Age'].median()

    for index, value in ages.items():
        dest_df.loc[(dest_df['Age'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Sex'] == index[1]), 'Age'] = value

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

def fillCabin(src_df, dest_df):
    cabins = src_df.groupby(['Title', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Embarked', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        dest_df.loc[(dest_df['Cabin'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Fare'] == index[1]) & 
                 (dest_df['Pclass'] == index[2]) &
                 (dest_df['SibSp'] == index[3]) &
                 (dest_df['Parch'] == index[4]) & 
                 (dest_df['Embarked'] == index[5]), 'Cabin'] = cabins[index[0], index[1], index[2], index[3], index[4], index[5]].idxmax()

    cabins = src_df.groupby(['Title', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        dest_df.loc[(dest_df['Cabin'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Fare'] == index[1]) & 
                 (dest_df['Pclass'] == index[2]) &
                 (dest_df['SibSp'] == index[3]) &
                 (dest_df['Parch'] == index[4]), 'Cabin'] = cabins[index[0], index[1], index[2], index[3], index[4]].idxmax()
                 
    cabins = src_df.groupby(['Title', 'Fare', 'Pclass', 'SibSp', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        dest_df.loc[(dest_df['Cabin'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Fare'] == index[1]) & 
                 (dest_df['Pclass'] == index[2]) &
                 (dest_df['SibSp'] == index[3]), 'Cabin'] = cabins[index[0], index[1], index[2], index[3]].idxmax()
                 
    cabins = src_df.groupby(['Title', 'Fare', 'Pclass', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        dest_df.loc[(dest_df['Cabin'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Fare'] == index[1]) & 
                 (dest_df['Pclass'] == index[2]), 'Cabin'] = cabins[index[0], index[1], index[2]].idxmax()
                 
    cabins = src_df.groupby(['Title', 'Fare', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        dest_df.loc[(dest_df['Cabin'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Fare'] == index[1]), 'Cabin'] = cabins[index[0], index[1]].idxmax()
                 
    cabins = src_df.groupby(['Title', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        dest_df.loc[(dest_df['Cabin'].isna() == True) &
                 (dest_df['Title'] == index[0]), 'Cabin'] = cabins[index[0]].idxmax()
                 
    dest_df.loc[(dest_df['Cabin'].isna() == True) & (
        (dest_df['Title'] == 3) | (dest_df['Title'] == 11)), 'Cabin' ] = 'X'
    
PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

train_df.loc[train_df['Cabin'] == 'T', 'Cabin'] = 'A'
train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25
test_df.loc[test_df['Embarked'].isna() == True, 'Embarked'] = 'S' 
test_df.loc[test_df['Fare'].isna() == True, 'Fare'] = 7.25


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

sexes = {
    'male': 0,
    'female': 1
    }  

train_df['Sex'] = train_df['Sex'].map(sexes)
test_df['Sex'] = test_df['Sex'].map(sexes)

all_df = pd.concat( [ train_df, test_df ], ignore_index = True )

fillAge(all_df, train_df)
fillAge(all_df, test_df)

train_df['Age'] = train_df['Age'].astype('int32')
test_df['Age'] = test_df['Age'].astype('int32')

binFare(all_df)
binAge(all_df)

fillCabin(all_df, train_df)
fillCabin(all_df, test_df)


train_df['Embarked'] = train_df['Embarked'].map(tb.embarkeds)
test_df['Embarked'] = test_df['Embarked'].map(tb.embarkeds)

train_df['Room']  = train_df['Cabin'].apply(tb.captureRoom)
test_df['Room']  = test_df['Cabin'].apply(tb.captureRoom)

train_df['Cabin'] = train_df['Cabin'].map(tb.cabins)
test_df['Cabin'] = test_df['Cabin'].map(tb.cabins)

tb.reeigneeringFamilySize(train_df)
tb.reeigneeringFamilySize(test_df)
                 
tbl = {
    "Title": np.union1d(train_df['Title'].unique(), test_df['Title'].unique()),
    "Age": np.union1d(train_df['Age'].unique(), test_df['Age'].unique()),
    "Sex": train_df['Sex'].unique(),
    "Pclass": train_df['Pclass'].unique(),
    "Cabin": np.union1d(train_df['Cabin'].unique(), test_df['Cabin'].unique()),
    "Size": np.union1d(train_df['Size'].unique(), test_df['Size'].unique()),
    "Fare": np.union1d(train_df['Fare'].unique(), test_df['Fare'].unique()),
    "Embarked": train_df['Embarked'].unique()    
    }

tb.navieBayes(train_df, tbl)
columns = [ 'Title', 'Sex', 'Pclass', 'Embarked', 'Size', 'Age', 'Fare', 'Cabin' ]
tb.reeigneeringSurvProb(train_df, columns)
tb.reeigneeringSurvProb(test_df, columns )





train_df.drop(columns = ['Name', 'Ticket', 'Title', 'Size'], inplace = True)
test_df.drop(columns = ['Name', 'Ticket', 'Title', 'Size'], inplace = True)


train_df.to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
test_df.to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)
