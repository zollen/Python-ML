'''
Created on Aug. 22, 2020

@author: zollen
'''
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('max_colwidth', 15)
pd.set_option('precision', 2)
np.random.seed(87)
sb.set_style('whitegrid')

label_column = [ 'survived' ]
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR , 'data/eval.csv'))

print(train_df.info())
print(train_df.isnull().sum())
print(train_df.describe())

for name in  categorical_columns:
    print("Total(valid [%s] records): %d" % (name, len(train_df[train_df[name] != 'unknown'])))
    print("Total('unknown' [%s] records): %d" % (name, len(train_df[train_df[name] == 'unknown'])))
    
"""
Since there is only one record with 'unknown' embark_town. We will deal with that record first.
"""
print(train_df[train_df['embark_town'] == 'unknown'])    


"""
Let's compare the column 'embark_town' with 'fare', 'class', 'deck'
The following visual analysis indicates that there is a very good chance this alone 
38 years old lady who paid 80 dollars of fare, First class, Deck 8 with no companion
probably came from either Southampton or Cherbourg
"""
if False:
    fig, (a1, a2, a3) = plt.subplots(1, 3)

    fig.set_size_inches(14 , 10)

    sb.swarmplot(x = "survived", y = "fare", hue = "embark_town", alpha = 0.9, data = train_df, ax = a1)
    a1.set_title('fare - embark_town')
    sb.countplot(x = "class", hue = "embark_town", data = train_df, ax = a2);
    a2.set_title('class - embark_town')
    sb.countplot(x = "deck", hue = "embark_town", data = train_df, ax = a3);
    a3.set_title('deck - embark_town')

    plt.show()

## *Never* use chained indexes (i.e. train_df[a][b][c][d] = 1..etc), each level of 
## indexing might return a copy instead of an actual view.    


train_df.loc[train_df['embark_town'] == 'unknown', 'embark_town'] = 'Cherbourg'
print(train_df.loc[48])




