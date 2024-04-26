'''
Created on Apr 26, 2024

@author: STEPHEN
'''

import numpy as np
from optimization.lib.Optimization import Optimization

class ParetoFront(Optimization):
    
    def __init__(self, obj_func, enforce_func, data_func, direction, population_size, 
                 obj_type = 'single', LB = -50, UB = 50, candidate_size = 0.01):
        super().__init__(obj_func, enforce_func, data_func, direction, 
                         population_size, obj_type, LB, UB, candidate_size)
             
    def move(self):
        r = np.random.rand(self.population.shape[0], self.population.shape[1])
        targets = np.tile(self.pareto_front, 
                        (int(np.ceil(self.population.shape[0] / self.population.shape[0])), 1))
        np.random.shuffle(targets)
        targets = targets[:-1,]
        self.population = targets + np.abs(self.population - targets) * \
                        np.power(np.e, r) * np.cos(2 * np.pi * r) 
    
    def start(self, rounds):
        for _ in range(rounds):
            self.best()
            self.move()     
        return self.best()