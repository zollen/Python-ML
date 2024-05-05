'''
Created on Apr 26, 2024

@author: STEPHEN
'''

import numpy as np
from optimization.lib.Optimization import Optimization

class ParetoFront(Optimization):
    
    def __init__(self, obj_func, data_func, checker_func, direction, population_size, 
                 ideal_scores, nadir_scores, LB = -50, UB = 50, candidate_size = 0.05):
        super().__init__(obj_func, data_func, checker_func, direction, 
                         population_size, ideal_scores, nadir_scores, LB, UB, candidate_size)
        
    def scale_up(self):
        return np.vstack((self.population, self.data_func(self.population_size * 50)))
    
    def scale_down(self, X):
        idx = np.array(range(X.shape[0]))
        np.random.shuffle(idx)
        idx = idx[:self.population_size]
        return X[idx]
          
    def move(self, X):
        res = self.consolidate(self.fitness(self.pareto_front))
        res += np.abs(np.min(res))
        ind = np.random.choice(np.array(range(res.size)), size=X.shape[0], p=(res / np.sum(res)))
        targets = self.pareto_front[ind]
        r = np.random.rand(X.shape[0], X.shape[1])
        b = np.random.choice([1, -1], size=(X.shape[0], X.shape[1]))
        X = self.bound(targets + 
                        np.abs(X - targets) * np.power(np.e, r) * 
                        np.cos(2 * np.pi * r) * b)
        
        res = self.checker_func(X)
        X = X[res > 0]
        return X
    
    def start(self, rounds):
        for _ in range(rounds):
            self.best()
            self.population = self.scale_down(self.move(self.scale_up()))   
        return self.best()