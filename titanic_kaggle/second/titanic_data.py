'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sb
import warnings
import titanic_kaggle.second.titanic_lib as tb

warnings.filterwarnings('ignore')


def combine(df1, df2):
    df = pd.concat([ df1, df2 ])
    df.set_index('PassengerId', inplace=True)
    return df

def accuracy(df, columns):
    func = tb.survivability(True, columns)
    df['Prediction'] = df.apply(func, axis = 1)
    total = len(df)
    good = len(df[((train_df['Prediction'] == 0) & (df['Survived'] == 0)) | 
              ((df['Prediction'] == 1) & (df['Survived'] == 1))])
    print("Accuracy(Chance): %0.4f" % (good / total))

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(0)
sb.set_style('whitegrid')

label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Ticket' ]
numeric_columns = [ 'Age', 'Fare' ]
categorical_columns = [ 'Sex', 'Title', 'Pclass', 'Embarked', 'Cabin' ]
all_features_columns = numeric_columns + categorical_columns 

## TO DO LIST
## 1. improve naviesBayes implementation
## 2. PCA and MeanShift analysis for Age and Fare
## 3. Mutli-steps group based medians approximation for Cabin
## 4. Rich women and Alive girl
    

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))


train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25
test_df.loc[test_df['Embarked'].isna() == True, 'Embarked'] = 'S' 
test_df.loc[test_df['Fare'].isna() == True, 'Fare'] = 7.25

tb.reeigneeringTitle(train_df)
tb.reeigneeringTitle(test_df)

tb.reeigneeringAge(train_df, train_df, 
                   [ 'Age', 'Title', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Survived' ])
tb.reeigneeringAge(combine(train_df, test_df), test_df, 
                   [ 'Age', 'Title', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass' ])

tb.reeigneeringFamilySize(train_df)
tb.reeigneeringFamilySize(test_df)

tb.reeigneeringFare(train_df)
tb.reeigneeringFare(test_df)

tb.reeigneeringCabin(combine(train_df, test_df), train_df)
tb.reeigneeringCabin(combine(train_df, test_df), test_df)

tb.navieBayes(train_df)

tb.reeigneeringSurvProb(train_df, [ 'Title', 'Sex', 'Pclass', 'Embarked', 'Size', 'Age', 'Fare' ])
tb.reeigneeringSurvProb(test_df, [ 'Title', 'Sex', 'Pclass', 'Embarked', 'Size', 'Age', 'Fare' ])



accuracy(train_df, [ 'Title', 'Sex', 'Pclass', 'Embarked', 'Size', 'Age', 'Fare' ])


train_df.drop(columns = [ 'Name', 'Ticket', 'Sex', 'SibSp', 'Parch', 'Prediction' ], inplace = True)
test_df.drop(columns = [ 'Name', 'Ticket', 'Sex', 'SibSp', 'Parch' ], inplace = True)


train_df.to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
test_df.to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)
