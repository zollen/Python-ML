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
        if self.obj_type == 'single':
            self.candidate_size = 1
        else:
            self.candidate_size = int(self.population_size * candidate_size)
    
    def fitness(self, X):
        if self.direction == 'max':
            return self.obj_func(X)
        else:
            return self.obj_func(X) * -1
    
    def best(self, X):
        scores = self.fitness(X)
        if self.obj_type == 'single':
            ind = np.argmax(scores) 
            if self.best_scores.size <= 0 or self.best_scores[0] < scores[ind]:
                self.best_scores = scores[ind]
                self.best_candidates = X[ind] 
            return self.best_candidates
        else:
            all_scores = np.concatenate((scores, self.best_scores))
            all_pop = np.vstack((self.population, self.best_candidates))
            ind = np.argpartition(all_scores, -self.candidate_size)[-self.candidate_size:]
            self.best_scores = all_scores[ind]
            self.best_candidates = all_pop[ind]
            return self.best_candidates
        
    def bound(self, X):
        X = np.where(X > self.LB, X, self.LB)
        X = np.where(X < self.UB, X, self.UB)
        return X
    
    def start(self, rounds):
        pass