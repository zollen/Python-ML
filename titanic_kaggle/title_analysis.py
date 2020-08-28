'''
Created on Aug. 28, 2020

@author: zollen
'''

import pandas as pd
import numpy as np
import re

SEED = 87

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(SEED)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

func = lambda x : re.search('[a-zA-Z]+\\.', x).group(0)
train_df['Title'] = train_df['Name'].apply(func)
test_df['Title'] = test_df['Name'].apply(func)

print(train_df[['PassengerId', 'Title']].groupby('Title').agg(['count']))

print(test_df[['PassengerId', 'Title']].groupby('Title').agg(['count']))

