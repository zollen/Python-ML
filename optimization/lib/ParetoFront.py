'''
Created on Apr 26, 2024

@author: STEPHEN
'''

import numpy as np
from optimization.lib.Optimization import Optimization

class ParetoFront(Optimization):
    
    def __init__(self, obj_func, constriants, data_func, direction, population_size, 
                 obj_type = 'single', LB = -50, UB = 50, candidate_size = 0.01):
        super().__init__(obj_func, constriants, data_func, direction, 
                         population_size, obj_type, LB, UB, candidate_size)
        
    def best(self):
        pop = np.vstack((self.population, self.best_candidates, self.pareto_front))
        scores = self.fitness(pop)
        self.pareto_front = pop[self.is_pareto_efficient(scores, 'max', False)]
        return self.pareto_front
        
    def move(self):
        pass
    
    def start(self, rounds):
        for _ in range(rounds):
            self.best()
            self.move()     
        return self.best()