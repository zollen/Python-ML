'''
Created on Mar 21, 2024

@author: STEPHEN
@url: https://www.baeldung.com/cs/grey-wolf-optimization

'''
import numpy as np
import sys
from grey_wolf.lib.grey_wolf import WolfPack

# f(a, b, c) = 3 * cos(a)^4 + 4 * cos(b)^3 + 2 sin(c)^2 * cos(c)^2 + 5
# constraint1: a + b + c = 1
# constraint2: c <= 0.8 
def myequation(X):
    "Objective function"
    return 3 * np.cos(X[:,0]) ** 4 + 4 * np.cos(X[:,1]) ** 3 + 2 * np.sin(X[:,2]) ** 2 * np.cos(X[:,2]) ** 2 + 5

def fitness(X):
    result = myequation(X)
    sc1 = np.abs(1 - (X[:,0] + X[:,1] + X[:,2])) * 10
    sc2 = np.abs(X[:,2] - 0.8) * 10
    
    score = result - sc1 - sc2
    
    ialpha = np.argmax(score)
    score[ialpha] = sys.maxsize * -1
    ibeta = np.argmax(score)
    score[ibeta] = sys.maxsize * -1
    igamma = np.argmax(score)
 
    return X[ialpha], X[ibeta], X[igamma]

def data(n):
    return np.random.rand(n, 3)
    
    
pack = WolfPack(fitness, data, 1000)    
alpha, beta, gamma = pack.hunt(100)
print("Global optimal at f({}) ==> {}".format(alpha, myequation(np.expand_dims(alpha, axis=0))))

'''
PSO Global optimal at f([a: 0.1003, b: 0.0997, c: 0.8000]) ==> score: 12.3804 

'''


