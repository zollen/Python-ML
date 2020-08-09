'''
Created on Aug. 3, 2020

@author: zollen
'''
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('C:\\Users\\zollen\\eclipse-workspace\\PythonExercise\\data\\iris.csv')

FEATURES = [ "sepal.length","sepal.width","petal.length","petal.width" ]

np.set_printoptions(precision=2)

print("==== MinMax Scaler ====")
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(df[FEATURES].values)
print(data[0:5])

print ("==== Binary Decider with thresold = 0.5 ====")
binarizer = Binarizer(threshold=0.5).fit(df[FEATURES].values)
data = binarizer.transform(df[FEATURES].values)
print(data[0:5])

print("== Gaussian Distribution Scaler ==")
scaler = StandardScaler().fit(df[FEATURES].values)
data = scaler.transform(df[FEATURES].values)
print(data[0:5])

print("=== Label Encoding ===")
input_labels = ['red','black','red','green','black','yellow','white']
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
print(encoder.transform(['green', 'red', 'black']))
print(encoder.inverse_transform([ 3, 0, 4, 1 ]))

aa = np.array([[  1.0, 2.0, 3.0], 
               [  5.0, 6.0, 7.0 ], 
               [  10.0, 6.0, 100.0 ], 
               [500.0, 2.0, 4.0]])
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
print(scaler.fit_transform(aa))

aa = [ 1.0, 5.0, 5.0, 1.0 ]
scaler = preprocessing.LabelBinarizer()
print(scaler.fit_transform(aa))

for i, j in zip([1, 2, 3], [4, 5, 6, 7]):
    print(i, j)