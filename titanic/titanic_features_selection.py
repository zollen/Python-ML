'''
Created on Aug. 17, 2020

@author: zollen
'''

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn import svm
import statsmodels.api as sm
import warnings


warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)
pd.set_option('max_colwidth', 15)
pd.set_option('precision', 2)


label_column = 'survived'
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns


def score_board(res):
    c1 = np.array(all_features_columns)[np.argsort(res)]
    c2 = res[np.argsort(res)]
    
    func = lambda x: round(x, 2)
    ss = np.stack((c1, np.array([func(e) for e in c2])), axis=1)     
    print(ss[::-1])



PROJECT_DIR=str(Path(__file__).parent)  
df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))

for name in categorical_columns:
    encoder = preprocessing.LabelEncoder()   
    keys = df[name].unique()
   
    if len(keys) == 2:
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    df[name] = encoder.transform(df[name].values)
    

print(df.info())
print("===== NUll values records =========")
print(df.isnull().sum())
print(df.describe())


print("======== SelectKBest(F_Score_Regression) ========")
model1 = SelectKBest(score_func=f_regression, k=6)
model1.fit(df[all_features_columns], df[label_column])
score_board(model1.scores_)

print("======== SelectKBest(chi2) ========")
model2 = SelectKBest(score_func=chi2, k=6)
model2.fit(df[all_features_columns], df[label_column])
score_board(model2.scores_)

print("========= Recursive Feature Eliminiation ========")
model3 = RFE(svm.SVC(kernel='linear', C=1.0), 9)
model3.fit(df[all_features_columns], df[label_column])
print(np.stack((all_features_columns, model3.support_), axis=1))

print("========== Non-Linear RBF PCA analysis ==============")
model4 = KernelPCA(kernel='rbf', n_components=9)
model4.fit(df[all_features_columns], df[label_column])
#print(model4.explained_variance_)
#print(model4.explained_variance_ratio_)
print(model4.singular_values_)


model=sm.Logit(df[label_column], df[all_features_columns])
result=model.fit()
print(result.summary2())




