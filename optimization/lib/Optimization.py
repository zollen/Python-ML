'''
Created on Apr 19, 2024

@author: STEPHEN
'''

import numpy as np
from pygini import gini

class Optimization:
    
    def __init__(self, obj_func, data_func, direction, population_size, 
                 obj_type = 'single', LB = -50, UB = 50, candidate_size = 0.05, stop_criteria=0.03):
        self.obj_func = obj_func
        self.data_func = data_func
        self.direction = direction
        self.population_size = population_size
        self.obj_type = obj_type
        self.LB = LB
        self.UB = UB
        self.stop_criteria = stop_criteria
        self.population = self.bound(self.data_func(self.population_size))
        self.best_candidates = np.array([])
        self.best_scores = np.array([])
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
        scores = self.fitness(X)
        ind = np.argmax(scores)
        if self.best_scores.size == 0 or self.best_scores[0] < scores[ind]:
            self.best_scores = scores[ind]
            self.best_candidates = self.population[ind]
        return self.best_candidates
    
    def final(self, pool_size = 1):
        scores = np.concatenate((self.fitness(self.population), self.best_scores))
        pop = np.vstack((self.population, self.best_candidates))
        ind = np.argpartition(scores, -pool_size)[-pool_size:]
        return pop[ind]
        
    def bound(self, X):
        X = np.where(X > self.LB, X, self.LB)
        X = np.where(X < self.UB, X, self.UB)
        return X
    
    def gini(self, X):
        return np.mean(gini(X, axis=0))
    
    def start(self, rounds):
        pass