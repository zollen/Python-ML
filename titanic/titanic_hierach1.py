'''
Created on Aug. 11, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

"""
## AgglomerativeClustering - for unsupervised learning only
"""

pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)


label_column = [ 'survived' ]
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns

func = lambda x : np.round(x, 2)

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))

print(train_df.info())
print(train_df.isnull().sum())
print(train_df.describe())


for name in categorical_columns:
    encoder = preprocessing.LabelEncoder()    
    
    keys = train_df[name].unique();
   
    if len(keys) == 2:
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    train_df[name] = encoder.transform(train_df[name].values)
    

print("============== Recursive Features Elmination (RFE) ===============")
model = DecisionTreeClassifier()
rfe = RFE(model, 7)
rfe = rfe.fit(train_df[all_features_columns], train_df[label_column])
print(rfe.support_)
print(rfe.ranking_)

model=sm.Logit(train_df[label_column], train_df[all_features_columns])
result=model.fit()
print(result.summary2())

print("=============== K Best Features Selection ==================")
model = SelectKBest(score_func=chi2, k=5)
kBest = model.fit(train_df[all_features_columns], train_df[label_column])
print(np.stack((all_features_columns, func(kBest.scores_)), axis=1))

print("================ Decision Tree Best Features Selection ============")
model = DecisionTreeClassifier()
model.fit(train_df[all_features_columns], train_df[label_column])
print(np.stack((all_features_columns, func(model.feature_importances_)), axis=1))

# alone are bigger than P0value 0.05, therefore we remove then
numeric_columns = [ 'fare' ]
categorical_columns = [ 'sex', 'class', 'deck', 'alone' ]
all_features_columns = numeric_columns + categorical_columns

model=sm.Logit(train_df[label_column], train_df[all_features_columns])
result=model.fit()
print(result.summary2())

model =  AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')


print("================= TRAINING DATA =====================")
model.fit_predict(train_df[all_features_columns], train_df[label_column])
print("Accuracy: ", round(accuracy_score(train_df[label_column], model.labels_), 2))
print("Precision: ", round(precision_score(train_df[label_column], model.labels_), 2))
print("Recall: ", round(recall_score(train_df[label_column], model.labels_), 2))
print(confusion_matrix(train_df[label_column], model.labels_))
print(classification_report(train_df[label_column], model.labels_))

