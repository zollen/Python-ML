'''
Created on Apr 12, 2024

@author: STEPHEN
'''

import numpy as np
from bee_colony_optim.lib.bee_colony import Bees



# f(a, b, c) = 3 * cos(a)^4 + 4 * cos(b)^3 + 2 sin(c)^2 * cos(c)^2 + 5
# constraint1: a + b + c = 1
# constraint2: c <= 0.8 
def equation(x, y, z):
    "Objective function"
    return 3 * np.cos(x) ** 4 + 4 * np.cos(y) ** 3 + 2 * np.sin(z) ** 2 * np.cos(z) ** 2 + 5

def fitness(X):
    result = equation(X[:, 0], X[:, 1], X[:, 2])
    sc1 = np.abs(1 - (X[:,0] + X[:,1] + X[:,2])) * 1000
    sc2 = np.abs(X[:,2] - 0.8) * 1000

    return result - sc1 - sc2

def data(n):
    return np.random.rand(n, 3)


bees = Bees(fitness, data, 'max', 6)
bees.employedBee()
bees.onlookerBee()

'''
p = np.array([5,5,7,1,7,5,5])
eb = np.array([[0.5,0.4,0.2], 
               [0.3,0.1,0.6], 
               [0.8, 0.2, 0.3], 
               [0.2, 0.2, 0.3],
               [0.3, 0.2, 0.1],
               [0.1, 0.7, 0.1],
               [0.8, 0.9, 0.8]])

eb = np.hstack((np.expand_dims(range(eb.shape[0]), axis=1), eb))
while p.size > 0:
    res = np.unique(p, return_index=True) 
    p_tmp = res[0]
    eb_tmp = eb[res[1]]
    print(p_tmp)  
    print(eb_tmp) 
    p = np.delete(p, res[1])
    eb = np.delete(eb, res[1], axis=0)

'''

