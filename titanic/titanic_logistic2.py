'''
Created on Aug. 4, 2020

@author: zollen
'''

import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt


def plot_roc(model, features, labels):
    logit_roc_auc = roc_auc_score(labels, model.predict(features))
    fpr, tpr, thresholds = roc_curve(labels, model.predict_proba(features)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.plot(fpr, thresholds, 'g--', label='decision thresholds')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')

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


pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)

label_column = [ 'survived' ]
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns

train_df = pd.read_csv('C:\\Users\\zollen\\eclipse-workspace\\PythonExercise\\data\\train.csv')
test_df = pd.read_csv('C:\\Users\\zollen\\eclipse-workspace\\PythonExercise\\data\\eval.csv')

labels = train_df[label_column]

print(train_df.describe())

print("Original Sample Size: Total Number of Survived: ", len(train_df[train_df['survived'] == 1]), 
      "Total Number of Dead: ", len(train_df[train_df['survived'] == 0]))


## Rebalance the dataset by oversmapling with replacement
rebalancer = RandomOverSampler(random_state=0)
#train_df, labels = rebalancer.fit_sample(train_df[all_features_columns], train_df[label_column])
print("ReBalacned Data pays more attention to the less ocurrence result")
print("This is not a good example for rebalance data. This technique is good for finding out who has the disease is more important")
print("ReBalanced Size: Total Number of Survived: ", len(labels[labels['survived'] == 1]),
      "Total Number of Dead: ", len(labels[labels['survived'] == 0]))



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

 
#for ele in train_df.head(n=5).iterrows():
#    print(ele)



print(train_df.describe())

logreg = LogisticRegression()
rfe = RFE(logreg, 7)
rfe = rfe.fit(train_df[all_features_columns], labels)
print(rfe.support_)
print(rfe.ranking_)

model=sm.Logit(labels, train_df[all_features_columns])
result=model.fit()
print(result.summary2())

train_df['family'] = train_df['n_siblings_spouses'] + train_df['parch']
test_df['family'] = test_df['n_siblings_spouses'] + test_df['parch']
# alone are bigger than P0value 0.05, therefore we remove then
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'family', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns

model=sm.Logit(labels, train_df[all_features_columns])
result=model.fit()
print(result.summary2())

model = LogisticRegression(max_iter=500)
model.fit(train_df[all_features_columns], labels)

preds = model.predict(test_df[all_features_columns])

print("Accuracy: ", accuracy_score(test_df[label_column], preds))
print("Precision: ", precision_score(test_df[label_column], preds))
print("Recall: ", recall_score(test_df[label_column], preds))
print(confusion_matrix(test_df[label_column], preds))
print(classification_report(test_df[label_column], preds))

plot_roc(model, test_df[all_features_columns].values, test_df[label_column].values)
plt.show()