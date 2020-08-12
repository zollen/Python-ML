'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

BATCH_SIZE = 20
EPOCHS = 30
STEPS_PER_EPOCH = 1
SEED = 0

label_column = [ 'survived' ]
numeric_columns = [ 'age', 'n_siblings_spouses', 'parch', 'fare' ]
categorical_columns = [ 'sex', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns


tbl={}

def plot_metrics(history):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()

def normalize(name, df):
    
    def norm(col):
        if name + '.max' not in tbl:
            tbl[name + '.max'] = np.max(df[name]) 
        if name + '.min' not in tbl:
            tbl[name + '.min'] = np.min(df[name])
        
        return (col - tbl[name + '.min'] + 1) / (tbl[name + '.max'] - tbl[name + '.min'])
    
    return norm

def features_columns(df):
    feature_cols = []
    
    for name in numeric_columns: 
        if name == 'fare':
            df[name] = np.log(df[name].replace(0, 1) + 0.001)
        normalizer = normalize(name, df)
        feature_cols.append(tf.feature_column.numeric_column(name, normalizer_fn = normalizer))
                                                            
    for name in categorical_columns:
        feature_cols.append(tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(name, df[name].unique())))
        
    feature_cols.append(tf.feature_column.indicator_column(tf.feature_column.crossed_column([ 'parch', 'alone', 'n_siblings_spouses' ], hash_bucket_size = 20)))

        
    return feature_cols
    
def make_ds(data):     
    ds = tf.data.Dataset.from_tensor_slices(
            (dict(data[all_features_columns]), data[label_column]))
    ds = ds.batch(BATCH_SIZE)
    return ds
    
pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)
pd.set_option('max_colwidth', 15)
pd.set_option('precision', 2)
tf.random.set_seed(SEED)
np.random.seed(SEED)


PROJECT_DIR=str(Path(__file__).parent.parent)
data = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
eval_df = pd.read_csv(os.path.join(PROJECT_DIR , 'data/eval.csv'))    
train_df = data.copy()
test_df, val_df = train_test_split(eval_df, test_size=0.2)


print(train_df.info())
print(train_df.isnull().sum())
print(train_df.describe())


neg, pos = np.bincount(train_df['survived'])
total = neg + pos

weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print("Dead: ", neg, " Survived: ", pos, " Bias: ", np.log([pos/neg]))
print("Dead weight: ", weight_for_0, " Survived weight: ", weight_for_1)



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

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(features_columns(train_df)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid',
            bias_initializer=tf.keras.initializers.Constant(np.log([pos/neg])))
])

model.compile(
      optimizer=tf.keras.optimizers.Adam(lr=1e-3),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=METRICS)

history = model.fit(make_ds(train_df),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    steps_per_epoch=STEPS_PER_EPOCH,
    class_weight=class_weight,
    callbacks = [early_stopping],
    validation_data=make_ds(val_df))


model.summary()

plot_metrics(history)

plt.show()

test_ds = make_ds(test_df)

result = model.evaluate(test_ds, batch_size = BATCH_SIZE)

print(result)

predictions = model.predict(test_ds)

# Show some results
for prediction, survived in zip(predictions[:10], test_df['survived'][:10]):
    prediction = tf.sigmoid(prediction).numpy()
    print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))