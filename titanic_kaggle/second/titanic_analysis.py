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
## 1. Refactoring
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


if False:
    g = sb.FacetGrid(train_df, col = "Survived", row = "Pclass", size = 2)
    g.map(sb.distplot, "Age", bins = 25)
    plt.show()
    exit()
    
if False:
    dd = train_df[train_df['Cabin'].isna() == False]  
    sb.catplot(x = "Pclass", y = "Title", hue = "Cabin", kind = "swarm", data = dd)
    plt.show()
    exit()
    
if False:    
#    dd = train_df[train_df['Cabin'].isna() == False]    
    sb.factorplot(x = 'Pclass' ,y = 'Fare', hue = 'Survived', kind = 'violin', data = train_df)
    plt.show()
    exit()
    
if False:
    for name in categorical_columns:
        if name == 'Cabin':
            continue
        keys = np.union1d(train_df[name].unique(), test_df[name].unique())            
        for key in keys:
            train_df[name + "." + str(key)] = train_df[name].apply(lambda x : 1 if x == key else 0)
          
    train_df.drop(columns = categorical_columns, inplace = True)
  
    dd = train_df.copy()
    dd.drop(columns=['PassengerId'], inplace=True)
    corr = dd.corr() 
  
    mask = np.triu(np.ones_like(corr, dtype=np.bool))    
    plt.figure(figsize=(14, 10))   
    sb.heatmap(corr, mask=mask, cmap='RdBu_r', annot=True, linewidths=0.5, fmt='0.2f')
    plt.show()
    exit()
    
if False:
    dd = train_df[train_df['Cabin'].isna() == False]
    for name in categorical_columns:
        keys = dd[name].unique()
        for key in keys:
            dd[name + "." + str(key)] = dd[name].apply(lambda x : 1 if x == key else 0)
    dd.drop(columns = categorical_columns, inplace = True)
    dd.drop(columns=['PassengerId'], inplace=True)
    corr = dd.corr()
     
    mask = np.triu(np.ones_like(corr, dtype=np.bool))    
    plt.figure(figsize=(16, 12))   
    sb.heatmap(corr, mask=mask, cmap='RdBu_r', annot=True, linewidths=0.5, fmt='0.2f')
    plt.show()
    exit()
    
    
if False:
    dd = train_df[train_df['Cabin'].isna() == True]
    sb.catplot(x = "Pclass", y = "Sex", hue = "Survived", kind = "swarm", data = dd)
    plt.show()
    exit()

train_df.to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
test_df.to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)
