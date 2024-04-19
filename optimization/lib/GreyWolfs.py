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

    
class WolfPack:
    
    def __init__(self, obj_func, data_func, direction, numOfWolves):
        self.numOfWolves = numOfWolves 
        self.direction = direction
        self.obj_func = obj_func
        self.data_func = data_func
        self.X = self.data_func(self.numOfWolves)
         
    def cofficients(self, a, n):
        r1 = np.random.rand(n, self.X[0].size)
        r2 = np.random.rand(1)
        A = 2 * a * r1 - a
        C = 2 * r2
        return A, C
    
    def fitness(self, X):
        if self.direction == 'max':
            return self.obj_func(X)
        else:
            return self.obj_func(X) * -1
    
    def best(self):
        scores = self.fitness(self.X)
        ind = np.argpartition(scores, -3)[-3:] 
        tt = [ scores[ind[0]], scores[ind[1]], scores[ind[2]] ]
        ii = np.argmax(tt)
        ialpha = ind[ii]
        tt = np.delete(tt, ii)
        ind = np.delete(ind, ii)
        ii = np.argmax(tt)
        ibeta = ind[ii]
        tt = np.delete(tt, ii)
        ind = np.delete(ind, ii)
        igamma = ind[0]
        return ialpha, ibeta, igamma
        
    def chase(self, a, alpha, beta, gamma):
        A1, C1 = self.cofficients(a, self.numOfWolves)
        A2, C2 = self.cofficients(a, self.numOfWolves)
        A3, C3 = self.cofficients(a, self.numOfWolves)
        
        D1 = np.abs( C1 * alpha - self.X )
        X1 = alpha - A1 * D1
        D2 = np.abs( C2 * beta - self.X )
        X2 = beta - A2 * D2
        D3 = np.abs( C3 * gamma - self.X )
        X3 = gamma - A3 * D3
        return (X1 + X2 + X3) / 3
   
    def hunt(self, rounds = 30):
        a = np.linspace(2, 0, rounds)
      
        for rnd in range(rounds):
            alpha, beta, gamma = self.best()
            self.X = self.chase(a[rnd], self.X[alpha], self.X[beta], self.X[gamma])     
        return self.X[alpha]