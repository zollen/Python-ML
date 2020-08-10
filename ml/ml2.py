
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

tbl={}
def normalize(name, col):
	if name + '.mean' not in tbl:
		tbl[name + '.mean'] = np.mean(col) 
	if name + '.min' not in tbl:
		tbl[name + '.min'] = np.min(col)
	if name + '.max' not in tbl:
		tbl[name + '.max'] = np.max(col)
	return (col - tbl[name + '.mean']) / (tbl[name + '.max'] - tbl[name + '.min'])


def read_file(file):
	data = pd.read_csv(file)
	print(data.dtypes)
	print(data.describe())
	data['sex'] = pd.Categorical(data['sex']).codes
	data['age'] = normalize('age', data['age'])
	data['n_siblings_spouses'] = normalize('n_siblings_spouses', data['n_siblings_spouses'])
	data['parch'] = normalize('parch', data['parch'])
	data['fare'] = normalize('fare', data['fare'])
	data['class'] = normalize('class', pd.Categorical(data['class']).codes)
	data['deck'] = normalize('deck', pd.Categorical(data['deck']).codes)
	data['embark_town'] = normalize('embark_town', pd.Categorical(data['embark_town']).codes)
	data['alone'] = pd.Categorical(data['alone']).codes
	
	data['survived'] = pd.Categorical(data['survived']).codes    
	label = data.pop('survived')
		
	return data, label


tf.keras.backend.set_floatx('float64')

model = tf.keras.Sequential([
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])


PROJECT_DIR=str(Path(__file__).parent.parent)  
dataTrain, labelTrain = read_file(os.path.join(PROJECT_DIR, 'data/train.csv'))
train_data = tf.data.Dataset.from_tensor_slices(( dataTrain, 
												 labelTrain ))

train_data = train_data.batch(1)


model.fit(train_data, epochs=35)
model.summary()

## The test data should be normalized against the training data, not the test data itself!!!
dataTest, labelTest = read_file(os.path.join(PROJECT_DIR, 'data/eval.csv'))
test_data = tf.data.Dataset.from_tensor_slices(( dataTest, 
												 labelTest ))
test_data = test_data.batch(1)
test_loss, test_accuracy = model.evaluate(test_data)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

test_result = model.predict(test_data)

count = 0
for i in range(test_result.size):
	pred = labelTest.iloc[i]
	act = 0 if test_result[i] < 0 else 1
	if pred == act:
		count = count + 1
#	print("Predict: ", pred, " ==> Actual: ", act)

print("TotaL: {}, Correct: {}, Percent: {}".format(test_result.size, count, count / test_result.size))

