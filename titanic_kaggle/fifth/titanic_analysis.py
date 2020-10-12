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
import warnings

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

last = 0
for age in [ 5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 75, 80 ]:
    alivesBoy = len(train_df[(train_df['Age'] >= last) & 
                             (train_df['Age'] < age) & 
                             (train_df['Sex'] == 'male') &
                             (train_df['Survived'] == 1)])
    deadsBoy = len(train_df[(train_df['Age'] >= last) & 
                            (train_df['Age'] < age) &
                            (train_df['Sex'] == 'male') & 
                            (train_df['Survived'] == 0)])
    alivesGirl = len(train_df[(train_df['Age'] >= last) & 
                             (train_df['Age'] < age) & 
                             (train_df['Sex'] == 'female') &
                             (train_df['Survived'] == 1)])
    deadsGirl = len(train_df[(train_df['Age'] >= last) & 
                            (train_df['Age'] < age) &
                            (train_df['Sex'] == 'female') & 
                            (train_df['Survived'] == 0)])
    
    ratioBoy = 0 if alivesBoy + deadsBoy == 0 else alivesBoy / (alivesBoy + deadsBoy)
    ratioGirl = 0 if alivesGirl + deadsGirl == 0 else alivesGirl / (alivesGirl + deadsGirl)
    print("[%2d, %2d]: Male {%2d, %2d} ==> [%0.4f], Female {%2d, %2d} ==> [%0.4f]" % 
          (last, age, alivesBoy, deadsBoy, ratioBoy, alivesGirl, deadsGirl, ratioGirl))
    last = age