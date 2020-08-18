'''
Created on Aug. 18, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)
pd.set_option('max_colwidth', 15)
pd.set_option('precision', 2)

label_column = [ 'class' ]
all_features_columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age' ]

PROJECT_DIR=str(Path(__file__).parent.parent)  
df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/pima-indians-diabetes.csv'))



