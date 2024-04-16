'''
Created on Apr 15, 2024

@author: STEPHEN

Global optimal [0.03937731 0.1612441  0.7991728 ] ==> 12.336679368657787
'''

import numpy as np
from cheetah_optim.lib.cheetahs import Cheetahs

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


cheetahs = Cheetahs(fitness, data, 'max', 1000) 
best = cheetahs.start(50)
    
print("Cheetahs optimal {} ==> {}".format(best, equation(best[0], best[1], best[2])))

'''
Cheetahs optimal [0.11928574 0.08071479 0.79999966] ==> 12.376264027458726
Cheetahs optimal [0.068196   0.13180399 0.80000001] ==> 12.368594662753967
Cheetahs optimal [0.07112005 0.12887995 0.8       ] ==> 12.370653681403345
'''

