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
pd.set_option('max_rows', None)
np.random.seed(0)
sb.set_style('whitegrid')


PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))

tb.reeigneeringTitle(train_df)

ddff = tb.normalize({}, train_df, ['Title', 'Sex', 'Embarked', 'Pclass' ])
       
ages = KNNImputer(n_neighbors=13).fit_transform(
    ddff[[ 'Age', 'Title', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Survived' ]])
           
train_df['Age'] = ages[:, 0]
train_df.loc[train_df['Age'] < 1, 'Age'] = 1
train_df['Age'] = train_df['Age'].astype('int32')
train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25

tb.reeigneeringTitle(train_df)
columns = [ 'Age', 'Title', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Survived' ]
ddff = tb.normalize({}, train_df, ['Title', 'Sex', 'Embarked', 'Pclass' ])
imputer = KNNImputer(n_neighbors=13)   
ages = imputer.fit_transform(ddff[columns])   
train_df['Age'] = ages[:, 0]
train_df.loc[train_df['Age'] < 1, 'Age'] = 1

train_df['Cabin'] = train_df['Cabin'].apply(tb.captureCabin) 


if True:
    da = train_df[(train_df['Cabin'].isna() == False) & (train_df['Survived'] == 1)]
    sb.factorplot('Cabin', col = 'Pclass', data = da, kind="count")
    dd = train_df[(train_df['Cabin'].isna() == False) & (train_df['Survived'] == 0)]
    sb.factorplot('Cabin', col = 'Pclass', data = dd,  kind="count")
    plt.show()
    
print(train_df.loc[(train_df['Cabin'].isna() == False) & 
               (train_df['Pclass'] == 3) &
               (train_df['Title'] == 'Mr'), ['Age', 'Fare', 'SibSp', 'Parch', 'Embarked', 'Cabin', 'Survived']])

