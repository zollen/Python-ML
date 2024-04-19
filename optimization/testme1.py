'''
Created on Apr 19, 2024

@author: STEPHEN
'''

import numpy as np
from optimization.lib.Optimization import Optimization


def equation(x, y, z):
    "Objective function"
    return 3 * x  + 4 * y +  5 * z + 1

def fitness(X):
    return equation(X[:, 0], X[:, 1], X[:, 2])

def data(n):
    return np.random.randint(0, 10, size=(n,3))


optimzer = Optimization(fitness, data, 'max', 10, obj_type = 'multiple', candidate_size=0.3)
scores = fitness(optimzer.population)
for i in range(optimzer.population.shape[0]):
    print(optimzer.population[i], " ==> ", scores[i])
print("=============================================")
best = optimzer.best(optimzer.population)
scores = fitness(best)
for i in range(best.shape[0]):
    print(best[i], " ==> ", scores[i])
