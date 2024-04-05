'''
Created on Apr 4, 2024

@author: STEPHEN
'''

from moth_swarn.lib.moths import *

# f(a, b, c) = 3 * cos(a)^4 + 4 * cos(b)^3 + 2 sin(c)^2 * cos(c)^2 + 5
# constraint1: a + b + c = 1
# constraint2: c <= 0.8 
def equation(x, y, z):
    "Objective function"
    return 3 * np.cos(x) ** 4 + 4 * np.cos(y) ** 3 + 2 * np.sin(z) ** 2 * np.cos(z) ** 2 + 5

def fitness(X):
    result = equation(X[:, 0], X[:, 1], X[:, 2])
    sc1 = np.abs(1 - (X[:,0] + X[:,1] + X[:,2])) * 10 + 100
    sc2 = np.abs(X[:,2] - 0.8) * 10 + 100

    return result - sc1 - sc2

def data(n):
    return np.random.rand(n, 3)


moths = MothsFlame(fitness, data, 'max', 1000)    
best = moths.start(50)
    
print("Global optimal {} ==> {}".format(best, equation(best[0], best[1], best[2])))

