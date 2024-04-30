'''
Created on Apr 19, 2024

@author: STEPHEN
'''

import numpy as np
from pygini import gini

class Optimization:
    
    def __init__(self, obj_func, data_func, checker_func, enforcer_func, direction, population_size, 
                 LB, UB, candidate_size, fitness_ratios):
        self.obj_func = obj_func
        self.checker_func = checker_func
        self.enforcer_func = enforcer_func
        self.data_func = data_func
        self.direction = direction
        self.population_size = population_size
        self.fitness_ratios = fitness_ratios
        self.LB = LB
        self.UB = UB
        self.candidate_size = int(np.ceil(population_size * candidate_size))
        self.population = self.bound(self.data_func(self.population_size))
        self.pareto_front = [ self.population[0] ]
        self.best_candidates = np.array([])
        self.best_scores = np.array([])
    
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
        
    def consolidate(self, X):
        if self.fitness_ratios == None:
            self.fitness_ratios = 1 / X.shape[1]           
        points = 0
        for col in range(X.shape[1]):
            points += X[:,col] * self.fitness_ratios[col]
        return points
          
    def best(self):
        if self.best_candidates.size == 0:
            pop = self.population
        else:
            pop = np.vstack((self.pareto_front, self.best_candidates, self.population))
        results = self.checker_func(pop)
        scores = self.fitness(pop)
        points = self.consolidate(scores)
        pop = pop[results == 6]
        scores = scores[results == 6]
        points = points[results == 6]
        size = int(scores.shape[0] * 0.1)
        if size < 5:
            size = 5
        ind = np.argpartition(points, -size)[-size:]
        self.best_candidates = pop[ind]
        self.pareto_front = pop[self.is_pareto_efficient(scores, self.direction, False)]
        return self.pareto_front
        
    def bound(self, X):
        X = np.where(X >= self.LB, X, self.LB)
        X = np.where(X <= self.UB, X, self.UB)
        return X
    
    def gini(self, X):
        return np.mean(gini(np.array(X), axis=0))
    
    def start(self, rounds):
        pass