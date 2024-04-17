'''
Created on Apr 15, 2024

@author: STEPHEN

Global optimal [0.03937731 0.1612441  0.7991728 ] ==> 12.336679368657787
'''

import numpy as np
from crow_optim.lib.crows import Crows

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


crows = Crows(fitness, data, 'max', 1000)    
best = crows.start(50)
    
print("Crows optimal {} ==> {}".format(best, equation(best[0], best[1], best[2])))

'''
Crows optimal [0.04397914 0.1555252  0.79965682] ==> 12.344912052322103
Crows optimal [0.12548645 0.07450495 0.7999926 ] ==> 12.37312578595073
Crows optimal [0.08087695 0.11913416 0.79998218] ==> 12.37608599536588
Crows optimal [0.10049555 0.09880521 0.79967978] ==> 12.381261523895574
'''

