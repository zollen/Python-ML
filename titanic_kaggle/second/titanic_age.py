'''
Created on Sep. 7, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import titanic_kaggle.second.titanic_lib as tb
import seaborn as sb
from matplotlib import pyplot as plt


pd.set_option('max_columns', None)
np.random.seed(0)
sb.set_style('whitegrid')


PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))

train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25
tb.reeigneeringTitle(train_df)
df = tb.normalize({}, train_df, ['Title', 'Sex', 'Embarked', 'Pclass' ])   
ages = KNNImputer(n_neighbors=13).fit_transform(
    df[[ 'Age', 'Title', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Survived' ]])   
train_df['Age'] = ages[:, 0]
train_df['Age'] = train_df['Age'].astype('int32')

tb.reeigneeringFamilySize(train_df)

tb.reeigneeringFare(train_df)


tb.reeigneeringCabin(train_df, train_df)

           
tbl = {
    "Title": train_df['Title'].unique(),
    "Age": train_df['Age'].unique(),
    "Sex": train_df['Sex'].unique(),
    "Pclass": train_df['Pclass'].unique(),
    "Cabin": train_df['Cabin'].unique(),
    "Size": train_df['Size'].unique(),
    "Fare": train_df['Fare'].unique(),
    "Embarked": train_df['Embarked'].unique()    
    }

tb.navieBayes(train_df, tbl)

tb.reeigneeringSurvProb(train_df, [ 'Title', 'Sex', 'Pclass', 'Embarked', 'Size', 'Age', 'Fare' ])

train_df.drop(columns = [ 'PassengerId', 'Name', 'Ticket', 'Sex', 'SibSp', 'Parch' ], inplace = True)

print(train_df.head())