'''
Created on Sep. 7, 2020

@author: zollen
'''

import os
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


print(test_df['Family'])


if False:
    dd = train_df[train_df['Sex'] == 'female']
    sb.catplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = dd)
    plt.show()
    
