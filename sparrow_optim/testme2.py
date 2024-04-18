'''
Created on Apr 5, 2024

@author: STEPHEN
'''

import numpy as np
from sparrow_optim.lib.sparrows import Sparrows


def equation(x, y):
    "Objective function"
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)

def fitness(X):
    return equation(X[:,0], X[:,1])

def data(n):
    return np.random.rand(n, 2) * 5
    

sparrows = Sparrows(fitness, data, 'min', 100)    
best = sparrows.start(50)
print("Sparrows optimal at f({}) ==> {}".format(best, equation(best[0], best[1])))

'''
PSO Global optimal at f([3.1818181818181817, 3.131313131313131])=-1.8082706615747688
Sparrows optimal at f([3.18599974 3.13008594]) ==> -1.8083474083344968
Sparrows optimal at f([3.18468681 3.12768791]) ==> -1.8083113529146235
Sparrows optimal at f([3.18267871 3.12872155]) ==> -1.8083079927904904

'''

