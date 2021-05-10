'''
Created on May 10, 2021

@author: zollen
'''

import pandas as pd
from kedro.pipeline import node
from kedro.pipeline import Pipeline
from kedro.io import DataCatalog, MemoryDataSet
from kedro.runner import SequentialRunner
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyClassifier
import warnings


warnings.filterwarnings("ignore")


data = pd.read_csv("../data/iris.csv")
dataSource = MemoryDataSet(data=data)

data_catalog = DataCatalog({"input": dataSource})


def splitData(input):
    feature_df = input[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
    label_df = input['variety']

    X_train, X_test, Y_train, Y_test = train_test_split(feature_df, label_df, test_size=0.3, random_state=0)
    return X_train, X_test, Y_train, Y_test
    
splitData_node = node(splitData, inputs="input", outputs=["X_train", "X_test", "Y_train", "Y_test"])


def classifyData(X_train, X_test, Y_train, Y_test):
    clf = LazyClassifier(verbose=0,ignore_warnings=True)
    models, predictions = clf.fit(X_train, X_test, Y_train, Y_test)
    print(models)

classify_node = node(classifyData, inputs=["X_train", "X_test", "Y_train", "Y_test"], outputs=None)

# Assign "nodes" to a "pipeline"
pipeline = Pipeline([ splitData_node, classify_node ])


# Create a "runner" to run the "pipeline"
runner = SequentialRunner()

# Execute a pipeline
print(runner.run(pipeline, data_catalog))





