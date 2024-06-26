'''
Created on Apr 16, 2024

@author: STEPHEN
@url: https://medium.com/analytics-vidhya/optimization-simplex-method-for-maximization-e117dfa38114

Maximize: p = 4x + 5y + 6z

Constrants: 2x + 3y + z <= 900
            3x + y + z <= 350
            4x + 2y + z <= 400
            x >= 0, y >= 0, z >= 0
            
The maximum optimal value is 2100 and found at (0,0, 350) of the objective function.

It is possible to perform integer base optimization with constraints.
                         
'''


import numpy as np

from grey_wolf.lib.grey_wolf import WolfPack
from whale_optim.lib.whales import Whales
from manta_rays_optim.lib.manta_rays import MantaRays
from moth_swarn.lib.moths import MothsFlame
from cheetah_optim.lib.cheetahs import Cheetahs
from crow_optim.lib.crows import Crows

def eq1(x, y, z):
    return  4 * np.floor(x) + 5 * np.floor(y) + 6 * np.floor(z)

def fitness(X):
    res = eq1(X[:,0], X[:,1], X[:,2])
    c1 = (900 - (2 * np.floor(X[:,0]) + 3 * np.floor(X[:,1]) + np.floor(X[:,2]))) 
    c2 = (350 - (3 * np.floor(X[:,0]) +     np.floor(X[:,1]) + np.floor(X[:,2]))) 
    c3 = (400 - (4 * np.floor(X[:,0]) + 2 * np.floor(X[:,1]) + np.floor(X[:,2]))) 
    penalty1 = np.where(X < 0, -5000000, 0)
    penalty2 = np.where(X > 500, -5000000, 0)
    penalty3 = np.where(c1 < 0, -5000000, 0)
    penalty4 = np.where(c2 < 0, -5000000, 0)
    penalty5 = np.where(c3 < 0, -5000000, 0)
    return res + penalty1[:,0] + penalty1[:,1] + penalty1[:,2] + penalty2[:, 0] + \
            penalty2[:, 1] + penalty2[:, 2] + penalty3 + penalty4 + penalty5

def data(n):
    return np.random.randint(0, 501, size=(n, 3))


pack = WolfPack(fitness, data, 'max', 10000)    
best = pack.hunt(100)
print("WolfPack optimal at f({}) ==> {}".format(np.floor(best), eq1(best[0], best[1], best[2])))

whales = Whales(fitness, data, 'max', 10000)    
best = whales.start(100)
    
print("Whales optimal f({}) ==> {}".format(np.floor(best), eq1(best[0], best[1], best[2])))

moths = MothsFlame(fitness, data, 'max', 10000)    
best = moths.start(100)
    
print("MothsFlame optimal f({}) ==> {}".format(np.floor(best), eq1(best[0], best[1], best[2])))

rays = MantaRays(fitness, data, 'max', 10000)    
best = rays.start(100)
    
print("MantaRays optimal f({}) ==> {}".format(np.floor(best), eq1(best[0], best[1], best[2])))

cheetahs = Cheetahs(fitness, data, 'max', 10000) 
best = cheetahs.start(100)
    
print("Cheetahs optimal f({}) ==> {}".format(np.floor(best), eq1(best[0], best[1], best[2])))

crows = Crows(fitness, data, 'max', 10000) 
best = crows.start(100)
    
print("Crows optimal f({}) ==> {}".format(np.floor(best), eq1(best[0], best[1], best[2])))