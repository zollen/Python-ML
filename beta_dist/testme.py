'''
Created on Dec. 25, 2020

@author: zollen
@url: https://stats.stackexchange.com/questions/47771/what-is-the-intuition-behind-beta-distribution
'''

import numpy as np

'''
It return a random number based on the probabilty distrution of (a)lpha and (b)eta

The probabilty distrubtion would be concentrated mostly on alpha / (alpha + beta) 

For Example: If you know the average probabilty of outcomes is about 0.27
One might use alpha = 81 and beta = 270
so 81 / (81 + 270) = 0.27, 
Then majority of the probilities will be centered around 0.2 and 0.35

'''
for _ in range(0, 100):
    print(np.random.beta(81, 270))
