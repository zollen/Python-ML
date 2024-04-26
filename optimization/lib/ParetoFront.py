'''
Created on Apr 26, 2024

@author: STEPHEN
'''

import numpy as np
from optimization.lib.Optimization import Optimization

class ParetoFront(Optimization):
    
    def __init__(self, obj_func, data_func, enforcer_func, direction, population_size, 
                 LB = -50, UB = 50):
        super().__init__(obj_func, data_func, enforcer_func, direction, 
                         population_size, LB, UB)
             
    def move(self):
        r = np.random.rand(self.population.shape[0], self.population.shape[1])
        targets = np.tile(self.pareto_front, 
                        (int(np.ceil(self.population.shape[0] / self.pareto_front.shape[0])), 1))
        np.random.shuffle(targets)
        if self.population.shape[0] < targets.shape[0]:
            targets = targets[:-(targets.shape[0] - self.population.shape[0]),]
        self.population = self.enforcer_func(self.bound(self.bound(targets + 
                        np.abs(self.population - targets) * np.power(np.e, r) * 
                        np.cos(2 * np.pi * r))))
    
    def start(self, rounds):
        for _ in range(rounds):
            self.best()
            self.move()     
        return self.best()