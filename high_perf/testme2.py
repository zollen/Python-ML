'''
Created on Jun. 11, 2021

@author: zollen
'''

import numpy as np
import time as time

x = np.arange(100000000).reshape(10000, 10000)


def go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace =+ np.tanh(a[i, i])
        
    return a + trace

# measure this, only the runtime is being measured.
start_time = time.time()
n = go_fast(x)
end_time = time.time()

print("CPU computation Time: ", end_time - start_time)


