'''
Created on Jul. 20, 2020

@author: zollen
'''
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def demo(column, batch):
    feature_layer = layers.DenseFeatures(column)
    tf.print(feature_layer(batch).numpy())   
    
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

df = pd.read_csv('C:\\Users\\zollen\\eclipse-workspace\\PythonExercise\\data\\petfinder-mini.csv')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('max_colwidth', 15)
pd.set_option('precision', 2)

print(df.dtypes)
print(df.head())
print("======================================================================================")
print(df.describe())


df['target'] = np.where(df['AdoptionSpeed'] == 4, 0, 1)
df = df.drop(columns = ['AdoptionSpeed', 'Description'])
df['Fee'] = np.log(df['Fee'] + 0.001)


train, test = train_test_split(df, test_size = 0.2)
train, val = train_test_split(df, test_size = 0.2)

print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

batch_size = 30 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['Age'])
    print('A batch of targets:', label_batch )


example_batch = next(iter(train_ds))[0]
   
     
## Feature column of regular numberic column     
photo_count = feature_column.numeric_column('PhotoAmt')
print("Standard numeric columns")
demo(photo_count, example_batch)


## Feature column of bukertized column
age = feature_column.numeric_column('Age')
age_buckets = feature_column.bucketized_column(age, boundaries = [1, 2, 3, 4, 5])
print("Standard bucketized columns")
demo(age_buckets, example_batch)

## Feature column of cateogorical column
animal_type = feature_column.categorical_column_with_vocabulary_list(
      'Type', ['Cat', 'Dog'])

animal_type_one_hot = feature_column.indicator_column(animal_type)
print("Standard one hot categorical columns")
demo(animal_type_one_hot, example_batch)

# Feature column of cateeogical colum with too many categories
# condesnse all possible categories into small number of columns
breed1 = feature_column.categorical_column_with_vocabulary_list(
      'Breed1', df['Breed1'].unique())
breed1_embedding = feature_column.embedding_column(breed1, dimension=8)
print("Standard unlimited categorical columns by using embedding")
demo(breed1_embedding, example_batch)

# Feature column of categorical colume with too many categories
# by putting each value into a hash bucket
breed1_hashed = feature_column.categorical_column_with_hash_bucket(
      'Breed1', hash_bucket_size=10)
print("Standard unlimited categorical values columns by using hashed buckets")
demo(feature_column.indicator_column(breed1_hashed), example_batch)

## Feature column of crossed feature columns: combining multiple features
## into one feature
crossed_feature = feature_column.crossed_column([age_buckets, animal_type], hash_bucket_size=10)
#demo(feature_column.indicator_column(crossed_feature), example_batch)
#############################################################################################

feature_columns = []

tbl={}
def normalize(name):
    
    def norm(col):
        if name + '.mean' not in tbl:
            tbl[name + '.mean'] = np.mean(df[name]) 
        if name + '.std' not in tbl:
            tbl[name + '.std'] = np.std(df[name])
        
        return (col - tbl[name + '.mean']) / tbl[name + '.std']
    
    return norm

for header in ['PhotoAmt', 'Fee', 'Age']:
    normalizer = normalize(header)
    feature_columns.append(feature_column.numeric_column(header, normalizer_fn = normalizer))
    
age_buckets = feature_column.bucketized_column(age, boundaries = [1, 2, 3, 4, 5])
feature_columns.append(age_buckets)   

indicator_column_names = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                          'FurLength', 'Vaccinated', 'Sterilized', 'Health']
for col_name in indicator_column_names:
    categorical_column = feature_column.categorical_column_with_vocabulary_list(col_name, df[col_name].unique()) 
    indicator_column = feature_column.indicator_column(categorical_column)
    feature_columns.append(indicator_column)
    
feature_columns.append(breed1_embedding)

age_type_feature = feature_column.crossed_column([age_buckets, animal_type], hash_bucket_size=100)
feature_columns.append(feature_column.indicator_column(age_type_feature))   

model = tf.keras.Sequential([
  layers.DenseFeatures(feature_columns),
  layers.Dense(256, activation='relu'),
  layers.Dense(256, activation='relu'),
  layers.Dense(256, activation='relu'),
  layers.Dense(256, activation='relu'),
  layers.Dropout(.1),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=20)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy {:.4f}, Loss: {:.4f}".format(accuracy, loss))