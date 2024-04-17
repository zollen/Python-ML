'''
Created on Apr 16, 2024

@author: STEPHEN
@url: https://medium.com/analytics-vidhya/optimization-simplex-method-for-maximization-e117dfa38114

Maximize: p = 4x + 5y + 6z

Constrants: 2x + 3y + z <= 900
            3x + y + z <= 350
            4x + 2y + z <= 400
            x >= 0, y >= 0, z >= 0
                         
'''


import numpy as np

from grey_wolf.lib.grey_wolf import WolfPack
from whale_optim.lib.whales import Whales
from manta_rays_optim.lib.manta_rays import MantaRays
from moth_swarn.lib.moths import MothsFlame
from cheetah_optim.lib.cheetahs import Cheetahs

def eq1(x, y, z):
    r = 4 * x + 5 * y + 6 * z
    c1 = (900 - 2 * x + 3 * y + z) * 5
    c2 = (350 - 3 * x +     y + z) * 5
    c3 = (400 - 4 * x + 2 * y + z) * 5
    return r + c1 + c2 + c3 

def equation(X):
    res = eq1(X[:,0], X[:,1], X[:,2])
    penalty1 = np.where(X < 0, -5000000, 0)
    penalty2 = np.where(X > 500, -5000000, 0)
    return res + penalty1[:,0] + penalty1[:,1] + penalty1[:,2] + penalty2[:, 0] + \
            penalty2[:, 1] + penalty2[:, 2]

def data(n):
    return np.random.randint(0, 500, size=(n, 3))


pack = WolfPack(equation, data, 'max', 1000)    
best = pack.hunt(70)
print("WolfPack optimal at f({}) ==> {}".format(best, eq1(best[0], best[1], best[2])))

whales = Whales(equation, data, 'max', 1000)    
best = whales.start(50)
    
print("Whales optimal f({}) ==> {}".format(best, eq1(best[0], best[1], best[2])))

moths = MothsFlame(equation, data, 'max', 1000)    
best = moths.start(100)
    
print("MothsFlame optimal f({}) ==> {}".format(best, eq1(best[0], best[1], best[2])))

rays = MantaRays(equation, data, 'max', 1000)    
best = rays.start(50)
    
print("MantaRays optimal f({}) ==> {}".format(best, eq1(best[0], best[1], best[2])))

cheetahs = Cheetahs(equation, data, 'max', 1000) 
best = cheetahs.start(70)
    
print("Cheetahs optimal f({}) ==> {}".format(best, eq1(best[0], best[1], best[2])))
