'''
Created on Mar 21, 2024

@author: STEPHEN
@url: https://www.baeldung.com/cs/grey-wolf-optimization
'''
import numpy as np
import sys
from grey_wolf.lib.grey_wolf import *

def myequation(X):
    "Objective function"
    return (X[:,0]-3.14)**2 + (X[:,1]-2.72)**2 + np.sin(3*X[:,0]+1.41) + np.sin(4*X[:,1]-1.73)

def fitness(X):
    result = myequation(X)
    ialpha = np.argmin(result)
    result[ialpha] = sys.maxsize
    ibeta = np.argmin(result)
    result[ibeta] = sys.maxsize
    igamma = np.argmin(result)
    
    return X[ialpha], X[ibeta], X[igamma]

def data(n):
    return np.random.rand(n, 2) * 5
    

res1 = []
for _ in range(50):     
    pack = WolfPack(fitness, data, 100)    
    alpha = pack.hunt(50)
    print("Global optimal at f({}) ==> {}".format(alpha, myequation(np.expand_dims(alpha, axis=0))))
    res1.append(myequation(np.expand_dims(alpha, axis=0)))
    
res2 = []
for _ in range(50):        
    pack = ImprovedWolfPack(myequation, fitness, data, 'min', 100)
    alpha = pack.hunt(100)
    print("Global optimal at f({}) ==> {}".format(alpha, myequation(np.expand_dims(alpha, axis=0))))
    res2.append(myequation(np.expand_dims(alpha, axis=0)))
   
print("Grey Wolf Avg: ", np.mean(res1))
print("Improved Grey Wolf Avg: ", np.mean(res2))

'''
PSO Global optimal at f([3.1818181818181817, 3.131313131313131])=-1.8082706615747688

'''


