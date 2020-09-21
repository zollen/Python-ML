'''
Created on Sep. 7, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import titanic_kaggle.second.titanic_lib as tb
import seaborn as sb
from matplotlib import pyplot as plt


pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(0)
sb.set_style('whitegrid')


PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))

train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25


tb.reeigneeringTitle(train_df)

titles = {
    "Army": 0,
    "Doctor": 1,
    "Nurse": 2,
    "Clergy": 3,
    "Baronness": 4,
    "Baron": 5,
    "Mr": 6,
    "Mrs": 7,
    "Miss": 8,
    "Master": 9,
    "Girl": 10,
    "Boy": 11,
    "GramPa": 12,
    "GramMa": 13
    }

train_df['Title'] = train_df['Title'].map(titles)



if True:
    sb.catplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = train_df)
    plt.show()
    
