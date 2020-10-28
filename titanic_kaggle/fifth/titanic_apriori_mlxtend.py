'''
Created on Oct. 28, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
from mlxtend.frequent_patterns import apriori
import titanic_kaggle.lib.titanic_lib as tb
import warnings

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')

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
    
PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25
test_df.loc[test_df['Embarked'].isna() == True, 'Embarked'] = 'S' 
test_df.loc[test_df['Fare'].isna() == True, 'Fare'] = 7.25


tb.reeigneeringTitle(train_df)
tb.reeigneeringTitle(test_df)

all_df = pd.concat( [ train_df, test_df ], ignore_index = True )

fillAge(all_df, train_df)
fillAge(all_df, test_df)

binAge(train_df)
binAge(test_df)
binFare(train_df)
binFare(test_df)

train_df.drop(columns = ['PassengerId', 'Title', 'Ticket', 'Name', 'Cabin'], inplace = True)

print(train_df.columns)

check_df = train_df.copy()


for name in check_df.columns:
    for val in check_df[name].unique():
        check_df[name + "_" + str(val)] = check_df[name].apply(lambda x : True if x == val else False)
    
    check_df.drop(columns = [name], inplace = True)

freq_df = apriori(check_df, min_support=0.20, use_colnames=True)
freq_df['length'] = freq_df['itemsets'].apply(lambda x: len(x))
print(freq_df[freq_df['length'] >= 5])
