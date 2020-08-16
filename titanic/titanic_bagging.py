'''
Created on Aug. 16, 2020

@author: zollen
'''
'''
Created on Aug. 16, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix


"""
A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random 
subsets of the original dataset and then aggregate their individual predictions (either by 
voting or by averaging) to form a final prediction. Such a meta-estimator can typically be 
used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by 
introducing randomization into its construction procedure and then making an ensemble 
out of it.

"""
 

label_column = [ 'survived' ]
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns


PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))

for name in categorical_columns + label_column:
    encoder = preprocessing.LabelEncoder()
        
    keys = train_df[name].unique()
   
    if len(keys) == 2:
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    train_df[name] = encoder.transform(train_df[name].values)


model1 = QuadraticDiscriminantAnalysis()
model1.fit(train_df[all_features_columns], train_df[label_column])
preds = model1.predict(train_df[all_features_columns])
print("============ LQA  =============")
print("Accuracy: ", round(accuracy_score(train_df[label_column], preds), 2))
print("Precision: ", round(precision_score(train_df[label_column], preds), 2))
print("Recall: ", round(recall_score(train_df[label_column], preds), 2))
print('AUC-ROC:', round(roc_auc_score(train_df[label_column], preds), 2))
print("Log Loss: ", round(log_loss(train_df[label_column], preds), 2))
print(confusion_matrix(train_df[label_column], preds))

model2 = BaggingClassifier(base_estimator=QuadraticDiscriminantAnalysis(), 
                        n_estimators=15)
model2.fit(train_df[all_features_columns], train_df[label_column])
preds = model2.predict(train_df[all_features_columns])
print("============ Bagging with LQA  =============")
print("Accuracy: ", round(accuracy_score(train_df[label_column], preds), 2))
print("Precision: ", round(precision_score(train_df[label_column], preds), 2))
print("Recall: ", round(recall_score(train_df[label_column], preds), 2))
print('AUC-ROC:', round(roc_auc_score(train_df[label_column], preds), 2))
print("Log Loss: ", round(log_loss(train_df[label_column], preds), 2))
print(confusion_matrix(train_df[label_column], preds))