'''
Created on Aug. 29, 2021

@author: zollen
@url: https://towardsdatascience.com/speed-up-your-pandas-processing-with-swifter-6aa314600a13
@desc: swifter with pandas for (pd.apply) can speed up operations for *only* vectorized function
'''

import pandas as pd
import numpy as np
import time
import swifter

df = pd.read_csv('../data/creditcard.csv')

print(df.head())
print(len(df))

start_t = time.time()
df['V1_1'] = df['V1'].apply(lambda x : x / 2 + 1)
end_t = time.time()
print("Standard Completed: ", end_t - start_t)

start_t = time.time()
df['V1_2'] = df['V1'].swifter.apply(lambda x : x / 2 + 1)
end_t = time.time()
print("Swifter  Completed: ", end_t - start_t)

# not support by swifter (swifter would implement disk parallel processing and 
# slow down the performance)
def non_vectorized_func(x):
    if x['Class'] == 0:
        return x['Class'] + 1
    else:
        return x['Class'] * 2
    
# supported by swifter
def vectorized_func(x):
    return np.where(x['Class'] == 0, x['Class'] + 1, x['Class'] * 2)

start_t = time.time()
df['Class_1'] = df.apply(non_vectorized_func, axis = 1)
end_t = time.time()
print("Non-Vectorized Completed: ", end_t - start_t)

start_t = time.time()
df['Class_1'] = df.apply(vectorized_func, axis = 1)
end_t = time.time()
print("Standard Completed     : ", end_t - start_t)

start_t = time.time()
df['Class_1'] = df.swifter.apply(vectorized_func, axis = 1)
end_t = time.time()
print("Swifter Completed     : ", end_t - start_t)


