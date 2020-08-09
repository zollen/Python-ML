'''
Created on Jul. 9, 2020

@author: zollen
'''

import collections
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep




nn = np.random.randint(1, 10, size=(1,10))
nn = np.array([[1., -1., 2.],[2., 0., 0.], [0., 1., -1.]], dtype='float32')
print(nn)
## data normalization
print((nn - np.min(nn)) / (np.max(nn) - np.min(nn)))
## data standardization
print((nn - np.average(nn)) / np.std(nn))

print("============ sklearn.preprocessing.normalize ================")
print(prep.normalize(nn))


print("============== tf.keras.utils.normalize =================")
print(tf.keras.utils.normalize(nn))

def pack(features, labels):
    for ele in features.values():
        print(">> ", type(ele), ele)
    for ele in labels:
        print("== ", type(ele), ele)
        
    return tf.stack(list(features.values()), axis=-1), labels

    
#dd = tf.data.Dataset.range(1, 6)

rec1 = collections.OrderedDict()
rec1['age'] = [ 28 ]
rec1['sib'] = [ 8.453 ]
rec1['parch'] = [ 0 ]
rec1['fare'] = [ 0 ]

rec1 = ( rec1, [ 0 ] )

rec2 = collections.OrderedDict()
rec2['age'] = [ 35 ]
rec2['sib'] = [ 53.1 ]
rec2['parch'] = [ 1 ]
rec2['fare'] = [ 0 ]

rec2 = ( rec2, [ 1 ] )


dataset = [ rec1, rec2 ]

for element in dataset:
    print(element)


print("===============================")
#packed_dataset = dataset.map(pack)
print("===============================")


aa = [  [1, 2, 3], 
        [3, 4, 4], 
        [5, 6, 7] ]

bb = tf.stack(aa, axis=-1)

print(bb)


t = [[1, 2, 3], 
     [4, 5, 6]]
print(tf.expand_dims(t, 1))

t = [[[1, 2, 3], [4, 5, 6 ]],
     [[7, 8, 9], [10, 11, 12]],
     [[13, 14, 15], [ 16, 17, 18]],
     [[19, 20, 21], [ 22, 23, 24]]
     ]


tf.print("SIZE: ", tf.size(t), " SIZE: ", tf.shape(t))
kk = tf.slice(t, [3, 0, 0], [1, tf.shape(t)[1], tf.shape(t)[2]])
tf.print("SHAPE[1, tf.shape(t)[1], tf.shape(t)[2]]: ", kk)
kk = tf.slice(t, [3, 0, 0], [1, 1, tf.shape(t)[2]])
tf.print("SHAPE[1, 1, tf.shape(t)[2]]: ", kk)
kk = tf.slice(t, [3, 0, 0], [1, 1, 1])
tf.print("SHAPE[1, 1, 1]: ", kk)

names = [ 'A', 'B', 'C' ]
features = { 'A': [1, 2, 3, 4 ], 'B': [ 5, 6, 7, 8 ], 'C': [ 9, 10, 11, 12 ] }
numeric_features = [features.pop(name) for name in names]
print(numeric_features)
numeric_features = tf.stack(numeric_features, axis = -1)
print(numeric_features)

aa = np.array([2, 4, 6, 8, 10, 12, 14])
ids = np.arange(len(aa))
np.random.shuffle(ids)
print("random.shuffle: ", ids, " ==> ", aa[ids])
choices = np.random.choice(aa, len(aa))
print("random.choice: ", choices)

print(np.log(500) + 0.001)
print(np.log(0) + 0.01)
print(np.log(1) )

aa = np.array([1, 0, 1, 1, 0, 1])
print(np.log(aa) + 0.001)
