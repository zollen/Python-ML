'''
Created on Mar. 1, 2021

@author: zollen
@url: https://www.kaggle.com/ryanholbrook/mutual-information

Mutual information describes relationships in terms of uncertainty. The mutual information (MI) between 
two quantities is a measure of the extent to which knowledge of one quantity reduces uncertainty 
about the other. If you knew the value of a feature, how much more confident would you be about the 
target?

'''
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import seaborn as sb
from matplotlib import pyplot as plt

def make_mi_scores(X, y):
    mi_scores = mutual_info_regression(X, y, discrete_features='auto')
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_utility_scores(scores):
    y = scores.sort_values(ascending=True)
    width = np.arange(len(y))
    ticks = list(y.index)
    plt.barh(width, y)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.show()
    
    
sb.set_style("whitegrid")

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'titanic_kaggle/data/train.csv'))

train_df.loc[train_df['Cabin'] == 'T', 'Cabin'] = 'A'
train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25

train_df.drop(columns = ['PassengerId', 'Ticket', 'Name', 'Age', 'Cabin'], inplace = True)

FEATURES = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
LABEL = 'Survived'

for name in train_df[FEATURES].select_dtypes("object"):
    train_df[name], uniques = train_df[name].factorize()
    print(name, " => ", uniques)
   
    
mi_scores = make_mi_scores(train_df[FEATURES], train_df[LABEL])
print(mi_scores)

plt.figure(dpi=100, figsize=(8, 5))
plot_utility_scores(mi_scores)