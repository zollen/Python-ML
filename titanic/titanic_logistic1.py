'''
Created on Aug. 4, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



tbl = {}

def normalize(col, keys):
    length = len(keys)
    for key, i in zip(keys, range(length)):
        tbl[name + "." + str(key)] =  float(i)
    tbl[name + ".total"] = float(length)
                
    testme = lambda x : tbl[name + "." + str(x)] / tbl[name + ".total"] if x > 0.0 else 0.0
    
    return list(map(testme, col))

def dummy(col, keys):
    return col


label_column = [ 'survived' ]
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns

PROJECT_DIR=str(Path(__file__).parent)
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR , 'data/eval_processed.csv'))

print(train_df.info())
print(train_df.describe())

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
train_df[numeric_columns] = scaler.fit_transform(train_df[numeric_columns].values)
test_df[numeric_columns] = scaler.transform(test_df[numeric_columns].values)


for name in categorical_columns:
    encoder = preprocessing.LabelEncoder()
    normalizer = normalize
    
    keys = np.union1d(train_df[name].unique(), test_df[name].unique())
   
    if len(keys) == 2:
        normalizer = dummy
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    train_df[name] = encoder.transform(train_df[name].values)
    test_df[name] = encoder.transform(test_df[name].values)
    
    keys = np.union1d(train_df[name].unique(), test_df[name].unique())
    
    train_df[name] = normalizer(train_df[name], keys)
    test_df[name] = normalizer(test_df[name].values, keys)


weights = np.array([ [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0] ])


    
def predict(data, weights):
    return 1.0 / ( 1.0 + np.exp(-1.0 * np.dot(data, weights)) )

def update_weights(data, labels, weights, lr):
    predictions = predict(data, weights)
    gradients = np.dot(data.T, predictions - labels)
    gradients /= len(labels)
    gradients *= lr
    weights -= gradients
    return weights

def compute_cost(data, labels, weights):
    predictions = predict(data, weights)
    cost1 = -1.0 * np.log(predictions)
    cost2 = -1.0 * np.log(1 - predictions)
    cost = cost1 + cost2
    return cost.sum() / len(labels)

def train(data, labels, weights, lr, iters):
    cost_history = []
    best = []
    best_cost = 1000
    for i in range(iters):
        weights = update_weights(data, labels, weights, lr)
        cost = compute_cost(data, labels, weights)
            
        if cost_history and best_cost > cost:
            best = weights.copy()
            best_cost = cost;
            
        cost_history.append(cost)
        if i % 1000 == 0:
            print ("Iteration: {} ==> Cost: {}".format(i, cost))
            
            
    return best, best_cost

def decide(data, weights):
    predictions = predict(data, weights)
    predictions = Binarizer(threshold=0.54).fit_transform(predictions)
    return predictions

weights, cost = train(train_df[all_features_columns].values, train_df[label_column].values, weights, 0.001, 20000)
print("Cost: ", cost)
print(weights)


print("================= TRAINING DATA =====================")
preds = decide(train_df[all_features_columns].values, weights)
print("Accuracy: %0.2f" % accuracy_score(train_df[label_column], preds))
print("Precision: %0.2f" % precision_score(train_df[label_column], preds))
print("Recall: %0.2f " % recall_score(train_df[label_column], preds))
print("Recall: %0.2f" % recall_score(train_df[label_column], preds))
print("AUC-ROC: %0.2f" % roc_auc_score(train_df[label_column], preds))
print("Log Loss: %0.2f" % log_loss(train_df[label_column], preds))
print(confusion_matrix(train_df[label_column], preds))

print("================= TEST DATA =====================")
preds = decide(test_df[all_features_columns].values, weights)
print("Accuracy: %0.2f" % accuracy_score(test_df[label_column], preds))
print("Precision: %0.2f" % precision_score(test_df[label_column], preds))
print("Recall: %0.2f" % recall_score(test_df[label_column], preds))
print("AUC-ROC: %0.2f" % roc_auc_score(test_df[label_column], preds))
print("Log Loss: %0.2f" % log_loss(test_df[label_column], preds))
print(confusion_matrix(test_df[label_column], preds))
print(classification_report(test_df[label_column], preds))

