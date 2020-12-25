'''
Created on Dec. 25, 2020

@author: zollen
@url: https://stats.stackexchange.com/questions/47771/what-is-the-intuition-behind-beta-distribution
'''

import numpy as np

'''
Beta distribution is best used for probabilities

It return a random number based on the probabilty distrution of (a)lpha and (b)eta

The probabilty distrubtion would be concentrated mostly on alpha / (alpha + beta) 

For Example: If you know the average probabilty of outcomes is about 0.27
One might use alpha = 81 and beta = 270
so the *mean* is 81 / (81 + 219) = 0.27, 
Then majority of the probilities will be centered around 0.2 and 0.35

'''
for _ in range(0, 100):
    print(np.random.beta(81, 219))

'''
If an outcome is improved little (a player won a match..etc), then
Beta(alpha + 1, beta). 
The distriubtion barely change just because of the winning of one match
'''
    
'''
If an outcome is improved a lot(a plyer won 100 matches among 200 new matches... etc), then
Beta(alpha + 100, beta + 200) 
The distriubtion will re-oriented to the new average and a more *sharp*
'''
    
    