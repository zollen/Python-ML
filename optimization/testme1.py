'''
Created on Apr 19, 2024

@author: STEPHEN

Global optimal [0.03937731 0.1612441  0.7991728 ] ==> 12.336679368657787
'''

import numpy as np
from optimization.lib.GreyWolfs import WolfPack


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


wolves = WolfPack(fitness, data, 'max', 100, obj_type = 'single', LB = -15, UB = 15)
best = wolves.start(30)
print("WolfPack optimal {} ==> {}".format(best, equation(best[:,0], best[:,1], best[:,2])))

'''
WolfPack optimal [[0.14025608 0.05956227 0.79964418]] ==> [12.36223935]
WolfPack optimal [[0.10031533 0.0995377  0.79963857]] ==> [12.38061563]
'''
