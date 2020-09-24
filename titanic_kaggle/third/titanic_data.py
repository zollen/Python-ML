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
from sklearn.preprocessing import Binarizer
from sklearn.metrics import accuracy_score

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

## pd.qcut() based boundaries yields better result
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

## pd.qcut() based boundaries yields better result    
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
    df.loc[(df['Fare'] < 1000) & (df['Fare'] >= 83.158), 'Fare'] = 100
    
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
columns = [ 'Title', 'Age', 'Sex', 'Pclass', 'Cabin', 'Size', 'Fare', 'Embarked' ]
coeffs = { "Title": 1.7661, "Age": 0.8183, "Sex": 0.7392, "Pclass": 0.8121, 
          "Cabin": -0.8381, "Size": 1.8125, "Fare": 0.2657, "Embarked": -0.2694 }
tb.reeigneeringSurvProb(ttrain_df, coeffs, columns)
tb.reeigneeringSurvProb(ttest_df, coeffs, columns )

train_df['Chance'] = ttrain_df['Chance']
test_df['Chance'] = ttest_df['Chance']

train_df['Cabin'] = train_df['Cabin'] * 1000 + train_df['Room']
test_df['Cabin'] = test_df['Cabin'] * 1000 + test_df['Room']

"""
1. Adding coef to each bayes features *use pso algo
2. Search common Name of relatives
3. implementing Ticket 
"""





ttrain_df['Bayes'] = Binarizer(threshold=0.5).fit_transform(
    np.expand_dims(train_df['Chance'].values, 1))
print("Bayes Accuracy: %0.4f" % accuracy_score(ttrain_df['Survived'], ttrain_df['Bayes']))

train_df.drop(columns = ['Name', 'Ticket', 'Title', 'Size', 'Room'], inplace = True)
test_df.drop(columns = ['Name', 'Ticket', 'Title', 'Size', 'Room'], inplace = True)


train_df.to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
test_df.to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)

print("Done")
