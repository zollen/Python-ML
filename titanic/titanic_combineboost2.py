'''
Created on Aug. 16, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
import warnings

warnings.filterwarnings('ignore')
np.random.seed(87)
    
    
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


model1 = LogisticRegression()

model2 = DecisionTreeClassifier()

model3 = svm.SVC(kernel='rbf', gamma ='auto', C=1.0)


model4 = QuadraticDiscriminantAnalysis()

model5 = KNeighborsClassifier(n_neighbors = 3, p=1)

print("=========== Voting Classifier with 12 fold Cross Validation =============")
"""
If ‘hard’, uses predicted class labels for majority rule voting. 
If ‘soft’, predicts the class label based on the argmax of the sums of the predicted 
    probabilities, which is recommended for an ensemble of well-calibrated classifiers.
"""
model = VotingClassifier(estimators = [ ('logistic',  model1), 
                                        ('tree',      model2), 
                                        ('svm',       model3),
                                        ('qda',       model4),
                                        ('Knn',       model5) ], voting='hard')


"""
See also
StratifiedKFold
Takes group information into account to avoid building folds with imbalanced class distributions (for binary or multiclass classification tasks).

GroupKFold
example: https://www.programcreek.com/python/example/91158/sklearn.model_selection.GroupKFold
K-fold iterator variant with non-overlapping groups. User defines which data belongs to which fold

RepeatedKFold
Repeats K-Fold n times.
"""

kfold = KFold(n_splits = 12, shuffle=True, random_state=0)
for classifier, label in zip([model1, model2, model3, model4, model5, model ], 
                      ['logistic', 'DecisionTree', 'SVM', 'QDA', 'Knn', 'Voting']):
    scores = cross_val_score(classifier, df[all_features_columns], df[label_column], cv=kfold)
    print("[%s] Accuracy: %0.2f (+/- %0.2f)" % (label, scores.mean(), scores.std()))



