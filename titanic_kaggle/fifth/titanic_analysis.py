'''
Created on Oct. 11, 2020

@author: zollen
'''
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import titanic_kaggle.lib.titanic_lib as tb
import warnings
from titanic_kaggle.lib.titanic_lib import captureRoom

warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
np.random.seed(0)
sb.set_style('whitegrid')
pp = pprint.PrettyPrinter(indent=3) 

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

train_df.loc[train_df['Cabin'] == 'T', 'Cabin'] = 'A'
train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25
test_df.loc[test_df['Embarked'].isna() == True, 'Embarked'] = 'S' 
test_df.loc[test_df['Fare'].isna() == True, 'Fare'] = 7.25

train_df['CabinPrefix'] = train_df['Cabin'].apply(tb.captureCabin)
test_df['CabinPrefix'] = test_df['Cabin'].apply(tb.captureCabin)
train_df['CabinPrefix'] = train_df['CabinPrefix'].map({ 'A': 0, 'B': 800, 'C': 400, 
                                           'D': 1200, 'E': 1000, 'F': 600, 'G': 200 })
test_df['CabinPrefix'] = test_df['CabinPrefix'].map({ 'A': 0, 'B': 800, 'C': 400, 
                                           'D': 1200, 'E': 1000, 'F': 600, 'G': 200 })

train_df['CabinRoom'] = train_df['Cabin'].apply(tb.captureRoom)
test_df['CabinRoom'] = test_df['Cabin'].apply(tb.captureRoom)

def calCabin(rec):
    prefix = rec['CabinPrefix']
    room = rec['CabinRoom']
    
    if str(prefix) != 'nan':
        return int(prefix) + int(room)
    
    return np.nan

train_df['Cabin'] = train_df.apply(calCabin, axis = 1)
test_df['Cabin'] = test_df.apply(calCabin, axis = 1)

train_df.drop(columns = ['CabinPrefix', 'CabinRoom'], inplace = True)
test_df.drop(columns= ['CabinPrefix', 'CabinRoom'], inplace = True)



    
def fillValues(name, df):
    columns, label = ['Sex', 'Fare', 'SibSp', 'Parch', 'Pclass', 'Embarked' ], [name]
    
    all_df = pd.concat([ train_df, test_df ])
    all_df.set_index('PassengerId', inplace=True)

    all_df = all_df[columns + label]
    all_df['Embarked'] = all_df['Embarked'].map({'S': 0, 'Q': 1, 'C': 2})
    all_df['Sex'] = all_df['Sex'].map({'male': 0, 'female': 1})
    all_df = pd.get_dummies(all_df, columns = ['Sex', 'Pclass', 'Embarked'])
    
    cols = set(all_df.columns)
    cols.remove(name)
    

    all_df_in = all_df.loc[all_df[name].isna() == False, cols]
    all_df_lb = all_df.loc[all_df[name].isna() == False, label]


    model = ExtraTreesRegressor(random_state = 0)
    model.fit(all_df_in, all_df_lb)
    
    
    work_df = df[columns + label]
    work_df['Embarked'] = work_df['Embarked'].map({'S': 0, 'Q': 1, 'C': 2})
    work_df['Sex'] = work_df['Sex'].map({'male': 0, 'female': 1})  
    work_df = pd.get_dummies(work_df, columns = ['Sex', 'Pclass', 'Embarked'])
    
    impute_df = work_df.loc[work_df[name].isna() == True, cols]
    
    imputed = model.predict(impute_df)
    df.loc[df[name].isna() == True, name] = imputed
    
    df[name] = df[name].astype('int64')

tmp_train_df, tmp_test_df = train_df.copy(), test_df.copy() 

fillValues('Age', tmp_train_df)
fillValues('Age', tmp_test_df)

fillValues('Cabin', tmp_train_df)
fillValues('Cabin', tmp_test_df)

train_df['Age'], test_df['Age'] = tmp_train_df['Age'], tmp_test_df['Age']
train_df['Cabin'], test_df['Cabin'] = tmp_train_df['Cabin'], tmp_test_df['Cabin']

train_df['Ticket'] = train_df['Ticket'].apply(tb.captureTicketId)
test_df['Ticket'] = test_df['Ticket'].apply(tb.captureTicketId)

def testme(name, *args):
    print(name)
    for value in args:
        print(value)
    
    
testme('hello', 1, 2, 3)

exit()
## study material
## https://www.kaggle.com/ash316/eda-to-prediction-dietanic
pd.crosstab([train_df.Sex, train_df.Survived], train_df.Pclass, margins = True).style.background_gradient(cmap = 'summer_r')
pd.crosstab(train_df.Title, train_df.Sex).T.style.background_gradient(cmap='summer_r')