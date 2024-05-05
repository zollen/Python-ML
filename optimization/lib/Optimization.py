'''
Created on Apr 19, 2024

@author: STEPHEN
'''

import numpy as np
from pygini import gini

class Optimization:
    
    def __init__(self, obj_func, data_func, checker_func, direction, population_size, 
                 ideal_scores, nadir_scores, LB, UB, candidate_size):
        self.obj_func = obj_func
        self.checker_func = checker_func
        self.data_func = data_func
        self.direction = direction
        self.population_size = population_size
        self.LB = LB
        self.UB = UB
        self.candidate_size = int(np.ceil(population_size * candidate_size))
        self.ideal_scores = ideal_scores
        self.nadir_scores = nadir_scores
        self.population = self.bound(self.data_func(self.population_size))
        self.pareto_front = [ self.population[0] ]
    
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
        
    def vikor(self, X):
        '''
        @url(VIKOR): https://www.youtube.com/watch?v=WMX3SVnvRls
        step1: normalize the matrix 
            fmax = max(X, axis=0), fmin = min(X, axis=0)
            x = (fmax - X) / (fmax - fmin)
        step2: calculating S and R
            S = sum(x, axis=1)
            R = max(x, axis=1)
            Smax = max(S)
            Smin = min(S)
            Rmax = max(R)
            Rmin = min(R)
        step3: calculating Q
            Q = 0.5 * ((S - Smin) / (Smax - Smin)) + 0.5 * ((R - Rmin) / (Rmax - Rmin))
            Q is the ranking
        step4: Acceptance of Rank choice
            C1 = acceptable advantages
                C1 = Q(second_best) - Q(best) >= DQ 
                however if failed, Then second_best and best are in the compromise group
                Then we pick the best candidate for above check 
                    C1 = Q(third_best) - Q(best) >= DQ ,
            C2 = acceptable stability in decision making
                DQ = 1 / (j - 1) where j is number of alternatives (size of dataset)
                alternatives must also be the best ranked by either R value or S values.
                
            Condition
                1. Alternative best and second_best, if condition is A2 not satisifed
                2. Alternative best, second, third...., if condition c1 is not satifisifed
                    a(th-best) is determined by the relation Q(a-th)- Q1 < DQ for maximum M 
                    (the position of these alternaives are in closeness)
        '''
        minn = np.min(X, axis=0)
        maxx = np.max(X, axis=0)
        x = (X - minn) / (maxx - minn)
        S = np.sum(x, axis=1)
        R = np.max(x, axis=1)
        Smax = np.max(S)
        Smin = np.min(S)
        Rmax = np.max(R)
        Rmin = np.min(R)
        return 0.5 * ((S - Smin) / (Smax - Smin)) + 0.5 * ((R - Rmin) / (Rmax - Rmin))
        
    def consolidate(self, X):
        return self.vikor(X)
          
    def best(self):
        pop = np.vstack((self.pareto_front, self.population))
        results = self.checker_func(pop)
        scores = self.fitness(pop)
        points = self.consolidate(scores)
        pop = pop[results > 0]
        scores = scores[results > 0]
        points = points[results > 0]
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