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
from catboost import CatBoostClassifier, Pool
import lightgbm as lgbm
from xgboost import XGBClassifier

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
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR , 'data/eval.csv'))

for name in categorical_columns + label_column:
    encoder = preprocessing.LabelEncoder()   
    keys = np.union1d(train_df[name].unique(), test_df[name].unique())
   
    if len(keys) == 2:
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    train_df[name] = encoder.transform(train_df[name].values)
    test_df[name] = encoder.transform(test_df[name].values)

"""
LightGBM can only load back to the basic booster (Not LGBMClassifier)
"""


model1 = CatBoostClassifier()
model1.load_model("models/catboost.model")

model2 = lgbm.Booster(model_file="models/lightgbm.model")

model3 = XGBClassifier()
model3.load_model("models/xgboost.model")

print("================= CatBoost Predictions =====================")
preds1 = model1.predict(test_df[all_features_columns])
result(test_df[label_column], preds1)

print("================= LightGBM Predictions =====================")
preds2 = model2.predict(test_df[all_features_columns], raw_score=True)
preds2 = np.squeeze(Binarizer(threshold=0.50).fit_transform(np.expand_dims(preds2, 1))).astype('int32')
result(test_df[label_column], preds2)

print("================= XGBoost Predictions =====================")
preds3 = model3.predict(test_df[all_features_columns])
result(test_df[label_column], preds3)

print("=========== Aggregate all three predictions and vote majority =============")
preds = np.stack((preds1, preds2, preds3), axis=1)
preds = np.apply_along_axis(lambda x : np.argmax(np.bincount(x)), 1, preds)
result(test_df[label_column], preds)






