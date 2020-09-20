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
    df['Age'] = pd.qcut(df['Age'], 10, labels = [7, 17, 22, 25, 27, 28, 30, 36, 44, 64])
    
def binFare(df):
    df['Fare'] = pd.qcut(df['Fare'], 13, labels = [4, 6, 7, 8, 9, 12, 15, 19, 25, 30, 45, 70, 100])

def fillCabin(src_df, dest_df):
    
    cabins = src_df.groupby(['Title', 'Fare', 'Pclass', 'Embarked', 'Cabin'])['Cabin'].count()
    
    for index, _ in cabins.items():
        dest_df.loc[(dest_df['Cabin'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Fare'] == index[1]) & 
                 (dest_df['Pclass'] == index[2]) &
                 (dest_df['Embarked'] == index[3]), 'Cabin'] = cabins[index[0], index[1], index[2], index[3]].idxmax()
    
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

train_df['Cabin'] = train_df['Cabin'].map(tb.cabins)
test_df['Cabin'] = test_df['Cabin'].map(tb.cabins)

tb.reeigneeringFamilySize(train_df)
tb.reeigneeringFamilySize(test_df)

tb.typecast(train_df)
tb.typecast(test_df)

                 
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

print("Done")
