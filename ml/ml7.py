'''
Created on Aug. 3, 2020

@author: zollen
'''
import tensorflow as tf

import seaborn as sb
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from  IPython import display
from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile

FEATURES = 28
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:],1)
    return features, label

logdir = pathlib.Path(tempfile.mkdtemp())
shutil.rmtree(logdir, ignore_errors = True)

gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

ds = tf.data.experimental.CsvDataset(gz,[tf.float32,]*(FEATURES+1), compression_type="GZIP")

packed_ds = ds.batch(10000).map(pack_row).unbatch()

#for features,label in packed_ds.batch(1000).take(1):
#    print(features[0])
#    plt.hist(features.numpy().flatten(), bins = 101)
    
#plt.show()
    
validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)

def get_callbacks(name):
    return [ 
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard(logdir/name),
        ]

#step = np.linspace(0,100000)
#lr = lr_schedule(step)
#plt.figure(figsize = (8,6))
#plt.plot(step/STEPS_PER_EPOCH, lr)
#plt.ylim([0,max(plt.ylim())])
#plt.xlabel('Epoch')
#_ = plt.ylabel('Learning Rate')

#plt.show()

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[
                  tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                  'accuracy'])

    model.summary()

    history = model.fit(
        train_ds,
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_data=validate_ds,
        callbacks=get_callbacks(name),
        verbose=0)
    return history

tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])

size_histories = {}

size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny', max_epochs=100)



sb.set_style('whitegrid')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(size_histories['Tiny'].history['loss'])
plt.plot(size_histories['Tiny'].history['val_loss'])
plt.legend(labels =['loss', 'val_loss'])

plt.show()

print(size_histories['Tiny'].history.keys())
