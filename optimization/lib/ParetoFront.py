'''
Created on Apr 26, 2024

@author: STEPHEN
'''

import numpy as np
from optimization.lib.Optimization import Optimization

class ParetoFront(Optimization):
    
    def __init__(self, obj_func, data_func, checker_func, enforcer_func, direction, population_size, 
                 LB = -50, UB = 50, candidate_size = 0.05, fitness_ratios = None):
        super().__init__(obj_func, data_func, checker_func, enforcer_func, direction, 
                         population_size, LB, UB, candidate_size, fitness_ratios)
             
    def move(self):
        pop = np.vstack((self.population, self.data_func(self.population_size * 50)))
        targets = np.tile(self.pareto_front, 
                        (int(np.ceil(pop.shape[0] / self.pareto_front.shape[0])), 1))
        np.random.shuffle(targets)
        if pop.shape[0] < targets.shape[0]:
            targets = targets[:-(targets.shape[0] - pop.shape[0]),]
            
        res = self.checker_func(pop)
        pop = pop[res == 6]
        targets = targets[res == 6]
        r = np.random.rand(self.population_size, pop.shape[1])
        d = np.random.choice([-1, 1], size=(self.population_size, pop.shape[1]))
        idx = np.array(range(pop.shape[0]))
        np.random.shuffle(idx)
        idx = idx[:self.population_size]
        pop = pop[idx]
        targets = targets[idx]
        self.population = self.bound(targets + 
                        np.abs(pop - targets) * np.power(np.e, r) * 
                        np.cos(2 * np.pi * r) * d)
    
    def start(self, rounds):
        for _ in range(rounds):
            self.best()
            self.move()     
        return self.best()