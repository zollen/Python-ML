'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import re
import seaborn as sb
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(0)
sb.set_style('whitegrid')

label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Ticket' ]
numeric_columns = [ 'Age', 'SibSp', 'Parch', 'Fare' ]
categorical_columns = [ 'Title', 'Sex', 'Pclass', 'Embarked' ]
all_features_columns = numeric_columns + categorical_columns 

title_category = {
        "Capt.": "Army",
        "Col.": "Army",
        "Major.": "Army",
        "Jonkheer.": "Baronness",
        "Don.": "Baron",
        "Sir.": "Baron",
        "Dr.": "Doctor",
        "Rev.": "Clergy",
        "Countess.": "Baronness",
        "Dona.": "Baronness",
        "Mme.": "Mrs",
        "Mlle.": "Miss",
        "Ms.": "Mrs",
        "Mr.": "Mr",
        "Mrs.": "Mrs",
        "Miss.": "Miss",
        "Master.": "Master",
        "Lady.": "Baronness"
    }

def map_title(rec):
    title = title_category[rec['Title']]
    sex = rec['Sex']
    
    if title == 'Doctor' and sex == 'male':
        return 'Doctor'
    else:
        if title == 'Doctor' and sex == 'female':
            return 'Nurse'
        
    if title == 'Miss' and rec['Age'] < 16:
        return 'Girl'
    
    return title
    


PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

train_df['Title'] = train_df['Name'].apply(lambda x : re.search('[a-zA-Z]+\\.', x).group(0))
train_df['Title'] = train_df.apply(map_title, axis = 1)

train_df.drop(columns=[ 'Name', 'Cabin', 'Ticket'], inplace=True)

train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'C'


for name in categorical_columns:
    encoder = preprocessing.LabelEncoder()   
    keys = train_df[name].unique()

    if len(keys) == 2:
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    train_df[name] = encoder.transform(train_df[name].values)

print(train_df.head())
