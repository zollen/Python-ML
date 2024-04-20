'''
Created on Apr 19, 2024

@author: STEPHEN
'''

import numpy as np

class Optimization:
    
    def __init__(self, obj_func, data_func, direction, population_size, 
                 obj_type = 'single', LB = -5, UB = 5, candidate_size = 0.05):
        self.obj_func = obj_func
        self.data_func = data_func
        self.direction = direction
        self.population_size = population_size
        self.obj_type = obj_type
        self.LB = LB
        self.UB = UB
        self.population = self.data_func(self.population_size)
        self.best_candidates = np.array([])
        self.best_scores = np.array([])
        self.best_positions = np.array([])
        if self.obj_type == 'single':
            self.candidate_size = 1
        else:
            self.candidate_size = int(self.population_size * candidate_size) + 1
    
    def fitness(self, X):
        if self.direction == 'max':
            return self.obj_func(X)
        else:
            return self.obj_func(X) * -1
    
    def best(self, X):
        all_pop = np.array(X)
        scores = self.fitness(all_pop)
        if self.best_scores.size > 0:
            all_scores =  np.where(scores[self.best_positions] < self.best_scores, 
                                self.best_scores, scores[self.best_positions])
            all_pop[self.best_positions] = np.where(np.expand_dims(scores[self.best_positions], axis=1) < 
                               np.expand_dims(self.best_scores, axis=1),
                                self.best_candidates, self.population[self.best_positions])
        else:
            all_scores = scores
            all_pop = self.population
        ind = np.argpartition(all_scores, -self.candidate_size)[-self.candidate_size:]
        self.best_scores = all_scores[ind]
        self.best_candidates = all_pop[ind]
        self.best_positions = ind
        return self.best_candidates
        
    def bound(self, X):
        X = np.where(X > self.LB, X, self.LB)
        X = np.where(X < self.UB, X, self.UB)
        return X
    
    def start(self, rounds):
        pass