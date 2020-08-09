'''
Created on Jul. 19, 2020

@author: zollen
'''

import numpy as np
import pandas as pd
import tensorflow as tf

tbl={}
def normalize(name, col):
    if name + '.mean' not in tbl:
        tbl[name + '.mean'] = np.mean(col) 
    if name + '.std' not in tbl:
        tbl[name + '.std'] = np.std(col)
        
    return (col - tbl[name + '.mean']) / tbl[name + '.std']

def inflate(arr, size):
    buf = np.zeros((arr.size, size))
    for n in range(0, arr.size):
        buf[n, 0] = arr[n]
    return buf

features = ['x', 'y']

dataTrain = pd.read_csv('C:\\Users\\zollen\\eclipse-workspace\\PythonExercise\\data\\data1.csv')
dataEval = pd.read_csv('C:\\Users\\zollen\\eclipse-workspace\\PythonExercise\\data\\eval1.csv')

print(dataTrain.describe())

origX = np.copy(dataEval['x'])
origY = np.copy(dataEval['y'])
dataTrain['x'] = normalize('x', dataTrain['x'])
dataTrain['y'] = normalize('y', dataTrain['y'])
dataEval['x'] = normalize('x', dataEval['x'])
dataEval['y'] = normalize('y', dataEval['y'])

training_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.expand_dims(dataTrain[features], 1),
            tf.expand_dims(dataTrain['result'], 1) 
        )
    )
)

eval_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.expand_dims(dataEval[features], 1),
            tf.expand_dims(dataEval['result'], 1)
        )
    )
)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy'])


model.fit(training_dataset, epochs=10)

model.summary()

test_loss, test_accuracy = model.evaluate(eval_dataset)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

test_result = model.predict(eval_dataset)

for i in range(test_result.size):
    print("[{}, {}] ===> {}".format(origX[i], origY[i], test_result[i]))

