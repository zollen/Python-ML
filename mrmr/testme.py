'''
Created on May 31, 2021

@author: zollen
@url: https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b
@title: MRMR (Maximum Relevance — Minimum Redundancy) is a feature selection algorithm 
        that has gained new popularity after the pubblication — in 2019 — of this paper 
        by Uber engineers:
        
        MRMR was developed to overcome this issue. 
        “The best K features are not the K best features”
        we need to find the best group of features, not best individual features
        
        Why we need it?
            1. We don’t know the causal relationships linking the features and the 
            target variable;
            2. We have too many features;
            3. There is high redundancy within the features.
        
        Maximum Relevance - Minimum Redundancy” is so called because — at each iteration 
        — we want to select the feature that has maximum relevance with respect to the 
        target variable and minimum redundancy with respect to the features that have 
        been selected at previous iterations.
        
        General Concept:
        i - iteration              
                                    relevance(f | target)
        score_i(f) = -----------------------------------------------
                       redundancy(f | features selected until i - 1)
                       
        Actual Equation:
        corr - Pearson Correlation
                            F-statistic-test(F | target)
        score_i(f) = ------------------------------------------------
                       Σ(features selected until i - 1) | corr(f,s) 
                       --------------------------------------------
                                       i - 1
        
'''

import pandas as pd
from mrmr import mrmr_classif
from sklearn.datasets import make_classification

# create some data
X, y = make_classification(n_samples = 1000, n_features = 50, n_informative = 10, n_redundant = 40)
X = pd.DataFrame(X)
y = pd.Series(y)

# use mrmr classification
selected_features = mrmr_classif(X, y, K = 10)
print(selected_features)