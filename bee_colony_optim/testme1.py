'''
Created on Apr 12, 2024

@author: STEPHEN
@best:
BeeColony optimal [0.13344278 0.06710421 0.7990396 ] ==> 12.367411787305533
BeeColony optimal [0.06392716 0.13783398 0.79929306] ==> 12.362444300354708
BeeColony optimal [0.08284015 0.11947901 0.79832788] ==> 12.373784393528066
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


bees = Bees(fitness, data, 'max', 1000)
best = bees.start(50)

print("BeeColony optimal {} ==> {}".format(best, equation(best[0], best[1], best[2])))

'''
Global optimal [0.03937731 0.1612441  0.7991728 ] ==> 12.336679368657787
'''
