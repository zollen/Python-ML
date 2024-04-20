'''
Created on Apr 19, 2024

@author: STEPHEN
'''

import numpy as np
from optimization.lib.GreyWolfs import WolfPack


def equation(x, y, z):
    "Objective function"
    return 3 * x  + 4 * y +  5 * z + 1

def fitness(X):
    return equation(X[:, 0], X[:, 1], X[:, 2])

def data(n):
    return np.random.randint(0, 10, size=(n,3))


wolves = WolfPack(fitness, data, 'max', 100, obj_type = 'single', LB = -15, UB = 15)
best = wolves.start(30)
print("WolfPack optimal {} ==> {}".format(best, equation(best[0], best[1], best[2])))


