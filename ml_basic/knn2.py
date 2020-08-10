'''
Created on Aug. 10, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.stats import mode

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('max_colwidth', 15)
pd.set_option('precision', 2)

CLUSTERS = 3
label_column = "variety" 
numeric_columns = [ "sepal.length", "sepal.width", "petal.length", "petal.width" ]

PROJECT_DIR=str(Path(__file__).parent.parent) 
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/iris.csv'))

print(train_df.head())


encoder = preprocessing.LabelEncoder()  
encoder.fit(train_df[label_column].unique())
train_df[label_column] = encoder.transform(train_df[label_column].values)



model = KMeans(n_clusters = CLUSTERS, random_state = 0)
df = PCA(n_components=2).fit_transform(train_df[numeric_columns])
preds = model.fit_predict(df)



print("================= TRAINING DATA =====================")
print("Accuracy: ", round(accuracy_score(train_df[label_column], preds), 2))
