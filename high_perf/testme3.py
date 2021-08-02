'''
Created on Aug. 2, 2021

@author: zollen
@title: Performance comparsion of various numpy/array iteration
'''

import numpy as np
import time

k = np.random.random([14242872, 11])
k = k * 10
print(k.shape)


p = np.array([1.2, 2.4, 3.5, 4.7, 5.8, 2.3, 1.1, 5.6, 9.2, 10.2, 3.7, 4.2])

start_t = time.time()
'''
# TIME: 107
np.apply_along_axis(lambda x : x[0] + 2 * x[1] + 2 * x[2] + 2 * x[3] + 
                    2 * x[4] + 2 * x[5] + 2 * x[6] + 2 * x[7] + 2 * x[8] + 
                    2 * x[9] + 2 * x[10], 1, k)
'''
'''
# 74.72
[ x[0] + 2 * x[1] + 2 * x[2] + 2 * x[3] + \
                    2 * x[4] + 2 * x[5] + 2 * x[6] + 2 * x[7] + 2 * x[8] + \
                    2 * x[9] + 2 * x[10] for x in k ]
'''
'''
# TIME: 80
count = 0
for x in k:
    count = x[0] + 2 * x[1] + 2 * x[2] + 2 * x[3] + \
                    2 * x[4] + 2 * x[5] + 2 * x[6] + 2 * x[7] + 2 * x[8] + \
                    2 * x[9] + 2 * x[10]
'''

'''
# TIME: 21
kk = k.tolist()
count = sum( x[0] + 2 * x[1] + 2 * x[2] + 2 * x[3] + \
                    2 * x[4] + 2 * x[5] + 2 * x[6] + 2 * x[7] + 2 * x[8] + \
                    2 * x[9] + 2 * x[10] for x in kk )

'''
end_t = time.time()
print("TIME: ", end_t - start_t)
