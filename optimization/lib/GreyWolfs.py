'''
Created on Apr 19, 2024

@author: STEPHEN
@url: https://www.baeldung.com/cs/grey-wolf-optimization
@url: https://www.nature.com/articles/s41598-019-43546-3   (improved grey wolf)
@description:

t = current iteration
X_prey = presumed prey position
X = position of a wolf 
r1 and r2 are random vectors with values between [0,1]
A vector controls the trade off between exploration and exploitation. For divergence, set A with random 
values > 1 or < -1 to oblige the search agent to diverge from the prey.
C vector always add some degree of randomness between [0, 2] because our agents can get stuck at a 
local optima (Exploration)
C is *not* linearly decreased in contrast to A. C must be random values at all time in order to 
emphasize exploration.

D = | C * X_prey(t) - X(t) |
X(t+1) = X_prey(t) - A * D

The fluctuation of A is also decreased by a. In other word A is a random value in the interval between 
[-2a, 2a] where a is linearly decreased from 2 to 0 over the course of the iterations.
|A| < 1 force the wolves to attack the prey (exploitation)
|A| > 1 force the wolves to diverge from the prey in the hope of finding a better prey
C < 1 deemphasize the attack
C > 1 emphasize the attack

a = 2 - t (2 / Max_t)     % a decrease linearly from 2 to 0,  t <- round
A = 2 * a * r1 - a
C = 2 * r2


We don't know the real position of the prey, so we use the best 3 solutions for updating each agent(wolf)
X is the current position of an agent

D_alpha = | C1 * X_alpha - X(t) |  # distance between all wolves and alpha
D_beta =  | C2 * X_beta - X(t)  |  # distance between all wolves and beta
D_gamma = | C3 * X_gamma - X(t) |  # distance between all wolves and gamma
X1 = X_alpha - A1 * D_alpha        # all wolves following the alpha
X2 = X_beta - A2 * D_beta          # all wolves following the beta
X3 = X_gamma - A3 * D_gamma        # all wolves following the gamma

X(t+1) = (X1 + X2 + X3) / 3        # average out

 



=================================================================
Initialize the grey wolf population X(i), i = 1...n
Initialize a, A and C
Calculate the fitness of each search agent
X_alpha = the best search agent
X_beta = the second best search agent
X_gamma = the third best search agent

while t < max_number_of_iteration do
    for each search agent do
        Update the position of the current search agent by the equation above.
    Update a, A and C
    Calculate the fitness of all search agents
    Update X_alpha, X_beta and X_gamma
    t = t + 1
    
return X_alpha
'''


import numpy as np
from optimization.lib.Optimization import Optimization

    
class WolfPack(Optimization):
    
    def __init__(self, obj_func, data_func, direction, num_wolves, obj_type = 'single',
                 LB = -5, UB = 5, candidate_size = 0.05):
        super().__init__(obj_func, data_func, direction, num_wolves, obj_type, LB, UB, candidate_size)
        if self.obj_type == 'single':
            self.candidate_size = 3
         
    def cofficients(self, a):
        r1 = np.random.rand(self.population_size, self.population[0].size)
        r2 = np.random.rand(1)
        A = 2 * a * r1 - a
        C = 2 * r2
        return A, C
    
    def chase(self, a, alpha, beta, gamma):
        A1, C1 = self.cofficients(a)
        A2, C2 = self.cofficients(a)
        A3, C3 = self.cofficients(a)
        
        D1 = np.abs( C1 * alpha - self.population )
        X1 = self.bound( alpha - A1 * D1 )
        D2 = np.abs( C2 * beta - self.population )
        X2 = self.bound( beta - A2 * D2 )
        D3 = np.abs( C3 * gamma - self.population )
        X3 = self.bound( gamma - A3 * D3 )
        return (X1 + X2 + X3) / 3
   
    def start(self, rounds = 30):
        a = np.linspace(2, 0, rounds)
      
        for rnd in range(rounds):
            best = self.best(self.population)
            self.population = self.chase(a[rnd], best[0], best[1], best[2])     
        return self.best(self.population, 1)