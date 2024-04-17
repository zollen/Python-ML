'''
Created on Apr 5, 2024

@author: STEPHEN
'''

import numpy as np
from crow_optim.lib.crows import Crows


def equation(x, y):
    "Objective function"
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)

def fitness(X):
    return equation(X[:,0], X[:,1])

def data(n):
    return np.random.rand(n, 2) * 5
    

crows = Crows(fitness, data, 'min', 100)    
best = crows.start(50)
print("Crows optimal at f({}) ==> {}".format(best, equation(best[0], best[1])))

'''
PSO Global optimal at f([3.1818181818181817, 3.131313131313131])=-1.8082706615747688
Crows optimal at f([3.18264642 3.12869166]) ==> -1.8083065294451597
Crows optimal at f([3.18516362 3.1297968 ]) ==> -1.8083520352292988

'''

