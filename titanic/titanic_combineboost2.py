'''
Created on Aug. 16, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier 
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import warnings

warnings.filterwarnings('ignore')
np.random.seed(87)

def result(observed, predicted):
    print("Accuracy: ", round(accuracy_score(observed, predicted), 2), 
          "Precision: ", round(precision_score(observed, predicted), 2),
          "Recall: ", round(recall_score(observed, predicted), 2),
          "AUC-ROC: ", round(roc_auc_score(observed, predicted), 2),
          "Log Loss: ", round(log_loss(observed, predicted), 2))
    
    
label_column = ['survived' ]
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns 

PROJECT_DIR=str(Path(__file__).parent.parent)  
df = pd.concat([ pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv')) , 
                 pd.read_csv(os.path.join(PROJECT_DIR , 'data/eval.csv')) ])

for name in categorical_columns + label_column:
    encoder = preprocessing.LabelEncoder()   
    keys = df[name].unique()
   
    if len(keys) == 2:
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    df[name] = encoder.transform(df[name].values)


model1 = CatBoostClassifier(verbose=False)

model2 = LGBMClassifier(num_leaves=32, max_depth=16, min_data_in_leaf=10)

model3 = XGBClassifier(n_estimators=150, max_depth=11, min_child_weight=3, gamma=0.2, subsample=0.6, nthread=16)

print("=========== Voting Classifier with 5 fold Cross Validation =============")
"""
If ‘hard’, uses predicted class labels for majority rule voting. 
If ‘soft’, predicts the class label based on the argmax of the sums of the predicted 
    probabilities, which is recommended for an ensemble of well-calibrated classifiers.
"""
model = VotingClassifier(estimators = [ ('cat',  model1), 
                                        ('lgbm', model2), 
                                        ('xgb',  model3) ], voting='soft')


for cls, label in zip([model1, model2, model3, model ], ['Cat', 'LGbm', 'XgB', 'Voting']):
    scores = cross_val_score(cls, df[all_features_columns], df[label_column], cv=5)
    print("[%s] Accuracy: %0.2f (+/- %0.2f)" % (label, scores.mean(), scores.std()))



