'''
Created on Jun. 11, 2021

@author: zollen
'''

from numba import jit
import numpy as np
import time as time

x = np.arange(100).reshape(10, 10)

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

print("Time: ", end_time - start_time)
print(n)

'''
Time:  0.37746644020080566
[[  1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
 [ 11.  12.  13.  14.  15.  16.  17.  18.  19.  20.]
 [ 21.  22.  23.  24.  25.  26.  27.  28.  29.  30.]
 [ 31.  32.  33.  34.  35.  36.  37.  38.  39.  40.]
 [ 41.  42.  43.  44.  45.  46.  47.  48.  49.  50.]
 [ 51.  52.  53.  54.  55.  56.  57.  58.  59.  60.]
 [ 61.  62.  63.  64.  65.  66.  67.  68.  69.  70.]
 [ 71.  72.  73.  74.  75.  76.  77.  78.  79.  80.]
 [ 81.  82.  83.  84.  85.  86.  87.  88.  89.  90.]
 [ 91.  92.  93.  94.  95.  96.  97.  98.  99. 100.]]
'''