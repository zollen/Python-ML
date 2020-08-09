'''
Created on Aug. 4, 2020

@author: zollen
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)
pd.set_option('max_colwidth', 15)
pd.set_option('precision', 2)


label_column = [ 'survived' ]
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns

train_df = pd.read_csv('C:\\Users\\zollen\\eclipse-workspace\\PythonExercise\\data\\train.csv')
test_df = pd.read_csv('C:\\Users\\zollen\\eclipse-workspace\\PythonExercise\\data\\eval.csv')


for name in categorical_columns:
    encoder = preprocessing.LabelEncoder()
    encoder.fit(np.union1d(train_df[name].unique(), test_df[name].unique()))
    train_df[name] = encoder.transform(train_df[name].values)
    test_df[name] = encoder.transform(test_df[name].values)

train_df['family'] = train_df['n_siblings_spouses'] + train_df['parch']
test_df['family'] = test_df['n_siblings_spouses'] + test_df['parch']

categorical_columns = [ 'sex', 'family', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns

gnb = GaussianNB()

model = gnb.fit(train_df[all_features_columns].values, train_df[label_column].values)



print("================= TRAINING DATA =====================")
preds = model.predict(train_df[all_features_columns])
print("Accuracy: ", round(accuracy_score(train_df[label_column], preds), 2))
print("Precision: ", round(precision_score(train_df[label_column], preds), 2))
print("Recall: ", round(recall_score(train_df[label_column], preds), 2))


print("================= TEST DATA =====================")
if True:
    preds = gnb.predict(test_df[all_features_columns].values)
else:
    preds = gnb.predict_proba(test_df[all_features_columns].values)
    binarizer = Binarizer(threshold=0.50).fit(preds)
    preds = binarizer.transform(preds)
    preds = np.argmax(preds, axis=1)


print("Accuracy: ", round(accuracy_score(test_df[label_column].values, preds), 2))
print("Precision: ", round(precision_score(test_df[label_column].values, preds), 2))
print("Recall: ", round(recall_score(test_df[label_column].values, preds), 2))
print(confusion_matrix(test_df[label_column], preds))
print(classification_report(test_df[label_column], preds))
