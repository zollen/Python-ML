'''
Created on Mar 21, 2024

@author: STEPHEN
@url: https://www.baeldung.com/cs/grey-wolf-optimization
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
import sys

def myequation(X):
    "Objective function"
    return (X[:,0]-3.14)**2 + (X[:,1]-2.72)**2 + np.sin(3*X[:,0]+1.41) + np.sin(4*X[:,1]-1.73)

def fitness(X):
    DD = myequation(X)
    ialpha = np.argmin(DD)
    DD[ialpha] = sys.maxsize
    ibeta = np.argmin(DD)
    DD[ibeta] = sys.maxsize
    igamma = np.argmin(DD)
    
    return X[ialpha], X[ibeta], X[igamma]

def data(n):
    return np.random.rand(n, 2) * 5
    
    
    
class WolfPack:
    
    def __init__(self, fitness, data_func, numOfWolves):
        self.numOfWolves = numOfWolves 
        self.fitness = fitness
        self.data_func = data_func
        self.X = self.data_func(self.numOfWolves)
         
    def cofficients(self, a, n):
        r1 = np.random.rand(n, self.X[0].size)
        r2 = np.random.rand(1)
        A = 2 * a * r1 - a
        C = 2 * r2
        return A, C
    
    def best(self):
        return self.fitness(self.X)
        
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
            self.X = self.chase(a[rnd], alpha, beta, gamma)
            print("Round: {} at f({}) ==> {}".format(rnd + 1, 
                                alpha, myequation(np.expand_dims(alpha, axis=0))))
            
        return alpha, beta, gamma
           

    
pack = WolfPack(fitness, data, 100)    
alpha, beta, gamma = pack.hunt(50)
np.printoptions(precision=4)
print("Global optimal at f({}) ==> {}".format(alpha, myequation(np.expand_dims(alpha, axis=0))))

'''
Global optimal at f([3.1818181818181817, 3.131313131313131])=-1.8082706615747688

'''


