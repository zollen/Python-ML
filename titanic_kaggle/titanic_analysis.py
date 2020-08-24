'''
Created on Aug. 1, 2020

@author: zollen
'''

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
import statsmodels.api as sm
from sklearn import preprocessing
import seaborn as sb
from matplotlib import pyplot as plt

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
sb.set_style('whitegrid')

label_column = [ 'Survived']
numeric_columns = [ 'Age', 'SibSp', 'Parch', 'Fare' ]
string_columns = [ 'Ticket', 'Cabin' ]
categorical_columns = [ 'Sex', 'Embarked', 'Pclass' ]
all_features_columns = numeric_columns + categorical_columns 


train_df = pd.read_csv('data/train.csv')


print(train_df.info())
print("=============== STATS ===================")
print(train_df.describe())
print("============== Total NULL ==============")
print(train_df.isnull().sum())
print("============== SKEW =====================")
print(train_df.skew())

print("ALIVE: ", len(train_df[train_df['Survived'] == 1]))
print("DEAD: ", len(train_df[train_df['Survived'] == 0]))


subsetdf = train_df[(train_df['Age'].isna() == False) & (train_df['Embarked'].isna() == False)]

df = subsetdf.copy()


for name in categorical_columns:
    encoder = preprocessing.LabelEncoder()   
    keys = df[name].unique()

    if len(keys) == 2:
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    df[name] = encoder.transform(df[name].values)

print("======= SelectKBest =======")
model = SelectKBest(score_func=chi2, k=5)
kBest = model.fit(df[all_features_columns], df[label_column])
func = lambda x : np.round(x, 2)
print(np.stack((all_features_columns, func(kBest.scores_)), axis=1))

print("======= ExtermeDecisionTree =======")
model = ExtraTreesClassifier()
model.fit(df[all_features_columns], df[label_column])
print(np.stack((all_features_columns, func(model.feature_importances_)), axis=1))

print("======= Logit Maximum Likelihood Analysis ===========")
model=sm.Logit(df[label_column], df[all_features_columns])
result=model.fit()
print(result.summary2())

"""
Importants of each feature
==========================
SelectKBest: Fare > Sex > Age > Pclass > Parch > Embarked > SibSp
DecisionTree: Age, Sex > Fare > Pclass > Parch, SibSp > Embarked
Logit: Fare, Sex has the highest confident of not seeing invalid variants
"""


if False:
    sb.catplot(x = "Survived", y = "Age", hue = "Sex", kind = "swarm", data = subsetdf)
    sb.catplot(x = "Survived", y = "Age", hue = "Pclass", kind = "swarm", data = subsetdf)
    sb.catplot(x = "Survived", y = "Age", hue = "Embarked", kind = "swarm", data = subsetdf)
    plt.show()
    
"""
    The above plot shows that Age does play a role of the outcome of survivaility.
    The number of dots between each column at each Age group appears to be vary enough
    to warrent a consideration
"""