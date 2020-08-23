'''
Created on Aug. 4, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


"""
## This is a Knn Clustering Example. Not Classification
"""

pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)


label_column = 'survived'
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns

func = lambda x : np.round(x, 2)

PROJECT_DIR=str(Path(__file__).parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'))

print(train_df.info())
print(train_df.isnull().sum())
print(train_df.describe())


labels = train_df[label_column]


for name in categorical_columns:
    encoder = preprocessing.LabelEncoder()   
    keys = train_df[name].unique()
   
    if len(keys) == 2:
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    train_df[name] = encoder.transform(train_df[name].values)
    

print("=============== K Best Features Selection ==================")
model = SelectKBest(score_func=chi2, k=5)
kBest = model.fit(train_df[all_features_columns], labels)
print(np.stack((all_features_columns, func(kBest.scores_)), axis=1))

print("================ Decision Tree Best Features Selection ============")
model = DecisionTreeClassifier()
model.fit(train_df[all_features_columns], labels)
print(np.stack((all_features_columns, func(model.feature_importances_)), axis=1))

# alone are bigger than P0value 0.05, therefore we remove then
#numeric_columns = [ 'fare' ]
#categorical_columns = [ 'sex', 'class', 'deck', 'alone' ]
#all_features_columns = numeric_columns + categorical_columns


NUM_OF_CLUSTERS = 2


model = KMeans(n_clusters = NUM_OF_CLUSTERS, random_state = 0)

print("================= TRAINING DATA =====================")
pca = PCA(n_components=2)
df = pca.fit_transform(train_df[all_features_columns])
preds = model.fit_predict(df, labels)
print("Accuracy: %0.2f" % accuracy_score(train_df[label_column], preds))
print("Precision: %0.2f" % precision_score(train_df[label_column], preds))
print("Recall: %0.2f" % recall_score(train_df[label_column], preds))
print("AUC-ROC: %0.2f" % roc_auc_score(train_df[label_column], preds))
print("Log Loss: %0.2f" % log_loss(train_df[label_column], preds))
print(confusion_matrix(train_df[label_column], preds))
print(classification_report(train_df[label_column], preds))

