'''
Created on Sep. 5, 2020

@author: zollen
'''

import os
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn import preprocessing
import numpy as np
import pandas as pd

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(0)

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'titanic_kaggle/data/train.csv'))


label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Ticket', 'Name', 'Cabin' ]
numeric_columns = [ 'Age', 'Fare' ]
categorical_columns = [ 'Sex', 'Pclass', 'Embarked' ]
all_features_columns = numeric_columns + categorical_columns 

train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.drop(columns=identity_columns, inplace=True)

for name in categorical_columns:
        encoder = preprocessing.LabelEncoder()   
        keys = train_df[name].unique()

        if len(keys) == 2:
            encoder = preprocessing.LabelBinarizer()

        encoder.fit(keys)
        train_df[name] = encoder.transform(train_df[name].values)
            


print(train_df.head())

print(train_df['Age'].isnull().sum())

imputer = KNNImputer(n_neighbors=1)
kk = imputer.fit_transform(train_df[['Sex', 'Age', 'Fare', 'Embarked', 'Pclass' ]])

print(kk[:,1].shape)
print(kk[:,1])
print(len(kk))

train_df['Age'] = kk[:, 1]

print(train_df['Age'])
    