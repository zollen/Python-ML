'''
Created on Dec. 14, 2020

@author: zollen
'''
import numpy as np
import time
import traceback
import sys

np.random.seed(int(round(time.time())))

'''
https://towardsdatascience.com/mixture-transition-distribution-model-e48b106e9560
Borrowing the concept of generalized mixture transaction to capture sequence information
then flatten it into a pandas dataframe
'''


initials = np.array([1, 0, 0])
transitions = []

for index in range(0, 10):
    transitions.append(np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]
                    ]).astype('float64'))
                    




moves = [ 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2 ]