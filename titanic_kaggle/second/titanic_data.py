'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import re
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import seaborn as sb
from matplotlib import pyplot as plt
import warnings
import titanic_kaggle.second.titanic_lib as tb



warnings.filterwarnings('ignore')

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

tb.reeigneeringAge(train_df, [ 'Age', 'Title', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Survived' ])
tb.reeigneeringAge(test_df, [ 'Age', 'Title', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass' ])

tb.reeigneeringFare(train_df)
tb.reeigneeringFare(test_df)

tb.reeigneeringFamilySize(train_df)
tb.reeigneeringFamilySize(test_df)

all_df = pd.concat([ train_df, test_df ])
all_df.set_index('PassengerId', inplace=True)
tb.reeigneeringCabin(all_df, train_df)
all_df = pd.concat([ train_df, test_df ])
all_df.set_index('PassengerId', inplace=True)
tb.reeigneeringCabin(all_df, test_df)

tb.navieBayes(train_df)

tb.reeigneeringSurvProb(train_df, [ 'Title', 'Sex', 'Pclass', 'Embarked', 'Size', 'Age', 'Fare' ])
tb.reeigneeringSurvProb(test_df, [ 'Title', 'Sex', 'Pclass', 'Embarked', 'Size', 'Age', 'Fare' ])



train_df.drop(columns = [ 'Name', 'Ticket', 'Sex', 'SibSp', 'Parch'], inplace = True)
test_df.drop(columns = [ 'Name', 'Ticket', 'Sex', 'SibSp', 'Parch'], inplace = True)




train_df.to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
test_df.to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)
