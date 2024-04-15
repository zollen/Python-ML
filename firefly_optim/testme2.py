'''
Created on Apr 5, 2024

@author: STEPHEN
'''

import numpy as np
from firefly_optim.lib.firefly import Fireflies


def equation(x, y):
    "Objective function"
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)

def fitness(X):
    return equation(X[:,0], X[:,1])

def data(n):
    return np.random.rand(n, 2) * 5
    

firefiles = Fireflies(fitness, data, 'min', 1000)    
best = firefiles.start(50)
print("Global optimal at f({}) ==> {}".format(best, equation(best[0], best[1])))

'''
PSO Global optimal at f([3.1818181818181817, 3.131313131313131])=-1.8082706615747688

'''

