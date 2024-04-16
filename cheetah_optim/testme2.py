'''
Created on Apr 5, 2024

@author: STEPHEN
'''

import numpy as np
from cheetah_optim.lib.cheetahs import Cheetahs


def equation(x, y):
    "Objective function"
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)

def fitness(X):
    return equation(X[:,0], X[:,1])

def data(n):
    return np.random.rand(n, 2) * 5
    

cheetahs = Cheetahs(fitness, data, 'min', 100)    
best = cheetahs.start(50)
print("Cheetahs optimal at f({}) ==> {}".format(best, equation(best[0], best[1])))

'''
PSO Global optimal at f([3.1818181818181817, 3.131313131313131])=-1.8082706615747688
Cheetahs optimal at f([3.18515538 3.12980282]) ==> -1.8083520359225962

'''

