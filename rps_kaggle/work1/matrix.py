'''
Created on Dec. 13, 2020

@author: zollen
'''

import numpy as np
np.set_printoptions(precision=4)

SIGNS = [ 'ROCK', 'PAPER', 'SCOSSOR']


initials = np.array([1, 0, 0])
transitions = np.array([
                        [0, 0, 1],
                        [1, 0, 0],
                        [0, 1, 0]
                    ]).astype('float64')

for index in range(0, 10):
    probs = np.matmul(np.linalg.matrix_power(transitions, index), initials)
    print(probs, " ==> ", SIGNS[np.argmax(probs)])
    
exit()
    
A = np.array([
        [-5, -4, 2],
        [-2, -2, 2],
        [ 4,  2, 2]
    ])

print(A.transpose())
W, V = np.linalg.eig(A)
print(W)
print(V)
D = np.zeros((3, 3))
for index in [0, 1, 2]:
    D[index, index] = W[index]

print(np.matmul(np.matmul(V, D), np.linalg.inv(V)))
