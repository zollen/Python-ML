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
       
ages = KNNImputer(n_neighbors=13).fit_transform(ddff[[ 'Age', 'Title', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Survived' ]])
           
train_df['Age'] = ages[:, 0]
train_df.loc[train_df['Age'] < 1, 'Age'] = 1
train_df['Age'] = train_df['Age'].astype('int32')
train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25

plt.figure(figsize=(12, 10))

sb.scatterplot(x = 'Age', y = 'Fare', hue=train_df['Survived'].tolist(), data = train_df)

plt.show()