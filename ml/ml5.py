'''
Created on Jul. 25, 2020

@author: zollen
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
bucket_column_names = ['age' ]
numeric_column_names = [ 'fare', 'n_siblings_spouses' ]
indicator_column_names = [ 'sex', 'parch', 'class', 'deck', 'embark_town', 'alone' ]
all_feauture_names = numeric_column_names + indicator_column_names + bucket_column_names
BATCH_SIZE = 30

tbl={}
def normalize(name, df):
    
    def norm(col):
        if name + '.mean' not in tbl:
            tbl[name + '.mean'] = np.mean(df[name]) 
        if name + '.std' not in tbl:
            tbl[name + '.std'] = np.std(df[name])
        
        return (col - tbl[name + '.mean']) / tbl[name + '.std']
    
    return norm

def read_file(file):
    data = pd.read_csv(file)
    data['survived'] = pd.Categorical(data['survived']).codes     
    data['fare'] = np.log(data['fare'] + 0.001)
    return data

def demo(column, batch):
    feature_layer = tf.keras.layers.DenseFeatures(column)
    tf.print(column.name)
    tf.print(feature_layer(batch))  
    

    
##tf.keras.backend.set_floatx('float64')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('max_colwidth', 15)
pd.set_option('precision', 2)
tf.random.set_seed(1)
np.random.seed(1)

dataTrain = read_file('C:\\Users\\zollen\\eclipse-workspace\\PythonExercise\\data\\train.csv')


print(dataTrain.dtypes)
print(dataTrain.head())
print(dataTrain.describe())

train_data = tf.data.Dataset.from_tensor_slices(( dict(dataTrain[all_feauture_names]),  
                                                  dataTrain['survived'] ))
train_data = train_data.batch(BATCH_SIZE)

example_batch = next(iter(train_data))[0]

tf.print(example_batch)


feature_columns = []    

for col_name in indicator_column_names:
    categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(col_name, dataTrain[col_name].unique()) 
    indicator_column = tf.feature_column.indicator_column(categorical_column)
    demo(indicator_column, example_batch)
    feature_columns.append(indicator_column)
    
for col_name in numeric_column_names:
    normalizer = normalize(col_name, dataTrain)
#    normalizer = None
    feature_columns.append(tf.feature_column.numeric_column(col_name, normalizer_fn = normalizer))

for col_name in bucket_column_names:
    col = tf.feature_column.numeric_column(col_name)
    feature_columns.append(tf.feature_column.bucketized_column(col, boundaries = [ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 ]))
    
age = tf.feature_column.numeric_column('age')
age_feature = tf.feature_column.bucketized_column(age, boundaries = [ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 ])
sex_feature = tf.feature_column.categorical_column_with_vocabulary_list('sex', dataTrain['sex'].unique())   
#feature_columns.append(tf.feature_column.indicator_column(tf.feature_column.crossed_column([ age_feature, sex_feature ], hash_bucket_size = 20)))

fare = tf.feature_column.numeric_column('fare')
fare_feature = tf.feature_column.bucketized_column(fare, boundaries = [ -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1 ])
class_feature = tf.feature_column.categorical_column_with_vocabulary_list('class', dataTrain['class'].unique())
#feature_columns.append(tf.feature_column.indicator_column(tf.feature_column.crossed_column([ fare_feature, class_feature ], hash_bucket_size = 50)))




METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]

model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=METRICS)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

history = model.fit(train_data, epochs=15, batch_size=BATCH_SIZE, callbacks = [early_stopping])

model.summary()


## The test data should be normalized against the training data, not the test data itself!!!
dataTest = read_file('C:\\Users\\zollen\\eclipse-workspace\\PythonExercise\\data\\eval.csv')
test_data = tf.data.Dataset.from_tensor_slices(( dict(dataTest[all_feauture_names]), 
                                                 dataTest['survived'] ))
test_data = test_data.batch(BATCH_SIZE)
#test_loss, test_accuracy = model.evaluate(test_data)
#print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

test_result = model.evaluate(test_data)
count = 0

for i in range(len(test_result)):
    pred = dataTest['survived'].iloc[i]
    act = 0 if test_result[i] < 0 else 1
    if pred == act:
        count = count + 1
#    print("Predict: ", pred, " ==> Actual: ", act)

print("TotaL: {}, Correct: {}, Percent: {}".format(len(test_result), count, count / len(test_result)))


