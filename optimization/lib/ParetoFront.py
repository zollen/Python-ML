'''
Created on Apr 26, 2024

@author: STEPHEN
'''

import numpy as np
from optimization.lib.Optimization import Optimization

class ParetoFront(Optimization):
    
    def __init__(self, obj_func, data_func, constraints_func, direction, population_size, 
                 ideal_scores, nadir_scores, LB = -50, UB = 50, candidate_size = 0.05):
        super().__init__(obj_func, data_func, constraints_func, direction, 
                         population_size, ideal_scores, nadir_scores, LB, UB, candidate_size)
        
    def scale_up(self):
        return np.vstack((self.population, self.data_func(self.population_size * 50)))
    
    def scale_down(self, X):
        idx = np.array(range(X.shape[0]))
        np.random.shuffle(idx)
        idx = idx[:self.population_size]
        return X[idx]
          
    def move(self, X):
        targets = np.tile(self.pareto_front,  
                        (int(np.ceil(X.shape[0] / self.pareto_front.shape[0])), 1)) 
        ind = np.array(range(targets.shape[0])) 
        np.random.shuffle(ind)
        targets = targets[ind]
        if X.shape[0] < targets.shape[0]: 
            targets = targets[:-(targets.shape[0] - X.shape[0]),] 
        r = np.random.rand(X.shape[0], X.shape[1])
        b = np.random.choice([1, -1], size=(X.shape[0], X.shape[1]))
        X = self.bound(targets + 
                        np.abs(X - targets) * np.power(np.e, r) * 
                        np.cos(2 * np.pi * r) * b)
        
        res = self.constraints_func(X)
        X = X[res > 0]
        return X
    
    def start(self, rounds):
        for _ in range(rounds):
            self.best()
            self.population = self.scale_down(self.move(self.scale_up()))   
        return self.best()