'''
Created on Aug. 26, 2020

@author: zollen
'''
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

SEED = 87

pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)
np.random.seed(SEED)

label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Name', 'Ticket' ]
numeric_columns = [ 'Age', 'SibSp', 'Parch', 'Fare' ]
categorical_columns = [ 'Sex', 'Embarked',  'Pclass', 'Cabin' ]
all_features_columns = numeric_columns + categorical_columns 


train_df = pd.read_csv('data/train_processed.csv')
test_df = pd.read_csv('data/test_processed.csv')


cat_columns = []

for name in categorical_columns:
    encoder = preprocessing.LabelEncoder()   
    keys = train_df[name].unique()
   
    if len(keys) == 2:
        encoder = preprocessing.LabelBinarizer()
        
    encoder.fit(keys)
    train_df[name] = encoder.transform(train_df[name].values)
    test_df[name] = encoder.transform(test_df[name].values)
    
    for key in keys:
        func = lambda x : 1 if x == key else 0
        train_df[name + '.' + str(key)] = train_df[name].apply(func)
        test_df[name + '.' + str(key)] = train_df[name].apply(func)
        cat_columns.append(name + '.' + str(key))
        

categorical_columns = cat_columns


if False:
    param_grid = dict({ "n_estimators": [ 50, 75, 100, 150 ],
                       "max_depth": [ 2, 5, 10, 15, 20, 25, 30 ],
                       "learning_rate": [ 0.001, 0.01, 0.1, 0.3, 0.5 ],
                       "gamma": [ 0, 1, 2, 3, 4 ],
                       "min_child_weight": [ 0, 1, 2, 3, 5, 7, 10 ],
                       "subsample": [ 0.2, 0.4, 0.6, 0.8, 1.0 ],
                       "reg_lambda": [ 0, 0.5, 1, 2, 3 ],
                       "reg_alpha": [ 0, 0.5, 1 ],
                       "max_leaves": [0, 1, 2, 5, 10 ],
                       "max_bin": [ 128, 256, 512, 1024 ] })
    model = RandomizedSearchCV(estimator = XGBClassifier(), 
                        param_distributions = param_grid, n_jobs=-1, n_iter=100)

    model.fit(train_df[all_features_columns], train_df[label_column].squeeze())

    print("====================================================================================")
    print("Best Score: ", model.best_score_)
    print("Best n_estimators: ", model.best_estimator_.n_estimators)
    print("Best max_depth: ", model.best_estimator_.max_depth)
    print("Best eta: ", model.best_estimator_.learning_rate)
    print("Best gamma: ", model.best_estimator_.gamma)
    print("Best min_child_weight: ", model.best_estimator_.min_child_weight)
    print("Best subsample: ", model.best_estimator_.subsample)
    print("Best lambda: ", model.best_estimator_.reg_lambda)
    print("Best alpha: ", model.best_estimator_.reg_alpha)

    exit()
    
   

model = XGBClassifier()
model.fit(train_df[all_features_columns], train_df[label_column])
print("XGB Score: ", model.score(train_df[all_features_columns], train_df[label_column]))

print("================= TRAINING DATA =====================")
preds = model.predict(train_df[all_features_columns])
print("Accuracy: %0.2f" % accuracy_score(train_df[label_column], preds))
print("Precision: %0.2f" % precision_score(train_df[label_column], preds))
print("Recall: %0.2f" % recall_score(train_df[label_column], preds))
print("AUC-ROC: %0.2f" % roc_auc_score(train_df[label_column], preds))
print("Log Loss: %0.2f" % log_loss(train_df[label_column], preds))
print(confusion_matrix(train_df[label_column], preds))
print(classification_report(train_df[label_column], preds))

kfold = StratifiedKFold(n_splits = 9, shuffle = True, random_state = SEED)
results = cross_val_score(XGBClassifier(), train_df[all_features_columns], train_df[label_column], cv = kfold)
print("9-Folds Cross Validation Accuracy: %0.2f" % results.mean())
print()
print()


preds = model.predict(test_df[all_features_columns])

result_df = pd.DataFrame({ "PassengerId": test_df['PassengerId'], "Survived" : preds })

print("========== TEST DATA ============")
print(result_df.head())

