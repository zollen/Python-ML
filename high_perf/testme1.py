'''
Created on Jun. 11, 2021

@author: zollen
'''

from numba import jit
import numpy as np
import time as time

x = np.arange(100000000).reshape(10000, 10000)

@jit(nopython=True) 
def go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace =+ np.tanh(a[i, i])
        
    return a + trace

# do not measure this, compilation time + run time
start_time = time.time()
n = go_fast(x)
end_time = time.time()

# measure this, only the runtime is being measured.
start_time = time.time()
n = go_fast(x)
end_time = time.time()

print("GPU computation Time: ", end_time - start_time)


