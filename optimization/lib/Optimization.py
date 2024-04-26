'''
Created on Apr 19, 2024

@author: STEPHEN
'''

import numpy as np
from pygini import gini

class Optimization:
    
    def __init__(self, obj_func, enforce_func, data_func, direction, population_size, 
                 obj_type = 'single', LB = -50, UB = 50, candidate_size = 0.05, stop_criteria=0.03):
        self.obj_func = obj_func
        self.enforce_func = enforce_func
        self.data_func = data_func
        self.direction = direction
        self.population_size = population_size
        self.obj_type = obj_type
        self.LB = LB
        self.UB = UB
        self.stop_criteria = stop_criteria
        self.population = self.enforce_func(self.bound(self.data_func(self.population_size)))
        self.pareto_front = [ self.population[0] ]
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
       
    def is_pareto_efficient(self, costs, direction = 'max', return_mask = True):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            if direction == 'max':
                nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            else:
                nondominated_point_mask = np.any(costs > costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient
    
    def best(self):
        pop = np.vstack((self.population, self.best_candidates, self.pareto_front))
        scores = self.fitness(pop)
        self.pareto_front = pop[self.is_pareto_efficient(scores, 'max', False)]
        return self.pareto_front
        
    def bound(self, X):
        X = np.where(X > self.LB, X, self.LB)
        X = np.where(X < self.UB, X, self.UB)
        return X
    
    def gini(self, X):
        return np.mean(gini(np.array(X), axis=0))
    
    def start(self, rounds):
        pass