'''
Created on Apr 16, 2024

@author: STEPHEN
@ranking

standard optimization
---------------------
grey wolfs
whales
moths swarm
manta rays
cheetahs


Zakharov
--------
-10 <= x <= 10
f(x_best) = 0 at x_best = (0,0,0,0)

'''

import numpy as np
from pymoo.problems import get_problem

from grey_wolf.lib.grey_wolf import WolfPack
from whale_optim.lib.whales import Whales
from manta_rays_optim.lib.manta_rays import MantaRays
from moth_swarn.lib.moths import MothsFlame
from cheetah_optim.lib.cheetahs import Cheetahs
from crow_optim.lib.crows import Crows

problem = get_problem("zakharov", n_var=4)



def zakharov4d(X):
    return problem.evaluate(X).squeeze()

def fitness(X):
    return zakharov4d(X)

def data(n):
    return np.random.uniform(-10, 10, size=(n, 4))


pack = WolfPack(fitness, data, 'min', 100)    
best = pack.hunt(50)
print("WolfPack optimal at f({}) ==> {}".format(best, zakharov4d(best)))

whales = Whales(fitness, data, 'min', 1000)    
best = whales.start(50)
    
print("Whales optimal f({}) ==> {}".format(best, zakharov4d(best)))

moths = MothsFlame(fitness, data, 'min', 1000)    
best = moths.start(100)
    
print("MothsFlame optimal f({}) ==> {}".format(best, zakharov4d(best)))

rays = MantaRays(fitness, data, 'min', 1000)    
best = rays.start(50)
    
print("MantaRays optimal f({}) ==> {}".format(best, zakharov4d(best)))

cheetahs = Cheetahs(fitness, data, 'min', 1000) 
best = cheetahs.start(50)
    
print("Cheetahs optimal f({}) ==> {}".format(best, zakharov4d(best)))

crows = Crows(fitness, data, 'min', 1000) 
best = crows.start(50)
    
print("Crows optimal f({}) ==> {}".format(best, zakharov4d(best)))
