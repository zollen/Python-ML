'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
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
    
    df = dest_df.copy()
    
    binFare(df)
    binAge(df)
     
    cabins = src_df.groupby(['Title', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Embarked', 'Cabin'])['Cabin'].count()
    
    for index, _ in cabins.items():
        df.loc[(df['Cabin'].isna() == True) &
                 (df['Title'] == index[0]) & 
                 (df['Fare'] == index[1]) & 
                 (df['Pclass'] == index[2]) &
                 (df['SibSp'] == index[3]) &
                 (df['Parch'] == index[4]) & 
                 (df['Embarked'] == index[5]), 'Cabin'] = cabins[index[0], index[1], index[2], index[3], index[4], index[5]].idxmax()
        
    cabins = src_df.groupby(['Title', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        df.loc[(df['Cabin'].isna() == True) &
                 (df['Title'] == index[0]) & 
                 (df['Fare'] == index[1]) & 
                 (df['Pclass'] == index[2]) &
                 (df['SibSp'] == index[3]) &
                 (df['Parch'] == index[4]), 'Cabin'] = cabins[index[0], index[1], index[2], index[3], index[4]].idxmax()
                   
    cabins = src_df.groupby(['Title', 'Fare', 'Pclass', 'SibSp', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        df.loc[(df['Cabin'].isna() == True) &
                 (df['Title'] == index[0]) & 
                 (df['Fare'] == index[1]) & 
                 (df['Pclass'] == index[2]) &
                 (df['SibSp'] == index[3]), 'Cabin'] = cabins[index[0], index[1], index[2], index[3]].idxmax()
                 
    cabins = src_df.groupby(['Title', 'Fare', 'Pclass', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        df.loc[(df['Cabin'].isna() == True) &
                 (df['Title'] == index[0]) & 
                 (df['Fare'] == index[1]) & 
                 (df['Pclass'] == index[2]), 'Cabin'] = cabins[index[0], index[1], index[2]].idxmax()
                  
    cabins = src_df.groupby(['Title', 'Fare', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        df.loc[(df['Cabin'].isna() == True) &
                 (df['Title'] == index[0]) & 
                 (df['Fare'] == index[1]), 'Cabin'] = cabins[index[0], index[1]].idxmax()
                  
    cabins = src_df.groupby(['Title', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        df.loc[(df['Cabin'].isna() == True) &
                 (df['Title'] == index[0]), 'Cabin'] = cabins[index[0]].idxmax()
                 
    df.loc[(df['Cabin'].isna() == True) & (
        (df['Title'] == 3) | (dest_df['Title'] == 11)), 'Cabin' ] = 'X'
        
    dest_df['Cabin'] = df['Cabin']
    
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

train_df['Room']  = train_df['Cabin'].apply(tb.captureRoom)
test_df['Room']  = test_df['Cabin'].apply(tb.captureRoom)

train_df['Cabin'] = train_df['Cabin'].apply(tb.captureCabin) 
test_df['Cabin'] = test_df['Cabin'].apply(tb.captureCabin) 

train_df['Title'] = train_df['Title'].map(tb.titles)
test_df['Title'] = test_df['Title'].map(tb.titles)

train_df['Sex'] = train_df['Sex'].map(tb.sexes)
test_df['Sex'] = test_df['Sex'].map(tb.sexes)

train_df['Embarked'] = train_df['Embarked'].map(tb.embarkeds)
test_df['Embarked'] = test_df['Embarked'].map(tb.embarkeds)

all_df = pd.concat( [ train_df, test_df ], ignore_index = True )

fillAge(all_df, train_df)
fillAge(all_df, test_df)
fillAge(all_df, all_df)

train_df['Age'] = train_df['Age'].astype('int32')
test_df['Age'] = test_df['Age'].astype('int32')

binFare(all_df)
binAge(all_df)

fillCabin(all_df, train_df)
fillCabin(all_df, test_df)
fillCabin(all_df, all_df)

train_df['Cabin'] = train_df['Cabin'].map(tb.cabins)
test_df['Cabin'] = test_df['Cabin'].map(tb.cabins)
all_df['Cabin'] = all_df['Cabin'].map(tb.cabins)

tb.reeigneeringFamilySize(train_df)
tb.reeigneeringFamilySize(test_df)
tb.reeigneeringFamilySize(all_df)

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


"""
1. implementing Rich Women 
2. implementing Ticket 
"""


train_df.drop(columns = ['Name', 'Ticket', 'Title', 'Size', 'Room'], inplace = True)
test_df.drop(columns = ['Name', 'Ticket', 'Title', 'Size', 'Room'], inplace = True)


train_df.to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
test_df.to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)

print("Done")
