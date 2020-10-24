'''
Created on Sep. 7, 2020

@author: zollen
'''

import os
import re
import pprint
from pathlib import Path
import numpy as np
import pandas as pd
import titanic_kaggle.lib.titanic_lib as tb
import seaborn as sb
from matplotlib import pyplot as plt


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
test_df.loc[test_df['Cabin'] == 'T', 'Cabin'] = 'A'
test_df.loc[test_df['Embarked'].isna() == True, 'Embarked'] = 'S'
test_df.loc[test_df['Fare'].isna() == True, 'Fare'] = 7.25


lives, deads = tb.calculateFamilyMembers(train_df)

tb.reenigneeringFamilyMembers(train_df, lives, deads)
tb.reenigneeringFamilyMembers(test_df, lives, deads)

train_df['Ticket'] = train_df['Ticket'].apply(tb.captureTicketId)
test_df['Ticket'] = test_df['Ticket'].apply(tb.captureTicketId)

train_df['Ticket'] = np.log(train_df['Ticket'])
test_df['Ticket'] = np.log(test_df['Ticket'])


print(train_df.describe())

  
df = train_df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
corr = df.corr()   
mask = np.triu(np.ones_like(corr, dtype=np.bool))    
plt.figure(figsize=(14, 10))   
sb.heatmap(corr, mask=mask, cmap='RdBu_r', annot=True, linewidths=0.5, fmt='0.2f')
plt.show()
    
