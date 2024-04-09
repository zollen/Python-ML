'''
Created on Apr 9, 2024

@author: STEPHEN
@url: https://www.nature.com/articles/s41598-023-51135-8#:~:text=The%20Whale%20Optimization%20Algorithm%20(WOA)23%20is%20a%20novel%20intelligent,whales%20in%20the%20natural%20world.
@url: https://www.youtube.com/watch?v=f7hvvDkLoHs
@desc: Combination of grey wolf and moths

Position update
----------------
A = 2 * a * r1 - a
C = 2 * r2
a = 2 - (2t / t_max)   t - current iteration, t_max - max iteration
X*(t) is the position of the prey

D = | C * X*(t) - X(t) |


l is a random number between -1 and 1
b is a constant used to define the logarithmic spiral shape

There are two methods of position updated.
    The contraction boundary Mechanism
    Spiral Updating position
    
X(t+1) = X*(t) - A * D
X(t+1) = D * e^(bl) * cos(2 pi * l) + X*(t)


Algorithm
=========
Initialization of agents
Calculate the fitness of each agent
x* = best agent

while curr_iter < max_iter:
    for each agent:
        update a,A,C,L and p
        if p < 0.5:
            if |A| < 1:
                update the position of each agent using D = | C * X*(t) - X(t) |
            else if |A| > 1:
                Select a random agent (X_rand)
                Update the position of the agent using X(t+1) = X_rand(t) - A * D
        else if p > 0.5:
            Update the position with spiral using X(t+1) = D * e^(bl) * cos(2 pi * l) + X*(t)
    
    Calculate the fitness of each agent
    Update x* if the method can detect a better solution
    
return x*        
    

'''

import numpy as np


class Whales:
    
    def __init__(self, fitness, data_func, direction, numOfWhales, spiral = 1):
        self.fitness = fitness
        self.data_func = data_func
        self.direction = direction
        self.numOfWhales = numOfWhales
        self.spiral = spiral
        self.whales = self.data_func(self.numOfWhales)
    
    def cofficients(self, rnd, rounds):
        a = 2 - (2 * rnd / rounds) 
        r1 = np.random.rand(self.whales.shape[0], self.whales.shape[1])
        r2 = np.random.rand(1)
        A = 2 * a * r1 - a
        C = 2 * r2
        p = np.expand_dims(np.random.rand(self.whales.shape[0]), axis=1)
        l = np.random.uniform(-1, 1, size=(self.whales.shape[0], self.whales.shape[1]))
        k = np.random.randint(0, self.numOfWhales, size=self.whales.shape[0])
        g = np.expand_dims(A[:,0], axis=1)
        return A, C, p, l, k, g
    
    def best(self):
        score = self.fitness(self.whales)
        if self.direction == 'max':
            ibest = np.argmax(score)
        else:
            ibest = np.argmin(score)
        return self.whales[ibest]
    
    def hunt(self, A, D, l, k, best):
        X1 = best - A * D                # good
        X2 = self.whales[k] - A * D
        X3 = D * np.power(np.e, self.spiral * l) * np.cos(2 * np.pi * l) + best
        return X1, X2, X3
    
    def start(self, rounds):
        for rnd in range(rounds):
            A, C, p, l, k, g = self.cofficients(rnd, rounds)
            best = self.best()
            D = np.abs(C * best - self.whales) 
            X1, X2, X3 = self.hunt(A, D, l, k, best)
            self.whales = np.where(p < 0.5, np.where(np.abs(g) < 1, X1, X2), X3)
                
        return self.best()        
                

