'''
Created on Apr 17, 2024

@author: STEPHEN
@url: https://www.youtube.com/watch?v=O6sxRAvycxk
@url: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10526407/#:~:text=The%20Crow%20Search%20Algorithm%20(CSA,limiting%20the%20algorithm%20solving%20ability.

Crow updates its position by selecting a random other crow i.e. x{j} and following it to known
m{j}. Then new x{j} is calculated as follows:

    Updating crow move
    ------------------
    AP{j,t}: crow j awareness probability
    r{i}, r{j}: random numbers
    fl{i,t}: crow i flight length to crow j memory


               x{i,t} + r{i} * fl{i,t} * (m{j,t} - x{i,t})  r{j} >= AP{j,t}
    x{i,t+1} = 
                a random position                             otherwise
                
                
    Updating crow memory
    --------------------
                x{i,t+1}    f(x{i},t+1) <= f(m{i},t)
    m{i,t+1} = 
                m{i,t}      otherwise
                
Steps
-----
Initialize position of crows and Initialize crow's memory
while cur_iter < max_iter:
    for crow in (1...crows}:
        choose a random crow
        determine a value of awareness probability AP and update X{p,t+1}
    Check solution boundaries
    Calculate the fitness of each crow and update crow's memory
    
'''

import numpy as np

class Crows:
    
    def __init__(self, obj_func, data_func, direction, numOfCrows, AP = 0.3, FL = 2):
        self.obj_func = obj_func
        self.data_func = data_func
        self.direction = direction
        self.numOfCrows = numOfCrows
        self.AP = AP
        self.FL = FL
        self.crows = self.data_func(self.numOfCrows)
        self.memory = np.array(self.crows)
        self.best_crow = None
        self.best_score = None
    
    def fitness(self, X):
        if self.direction == 'max':
            return self.obj_func(X)
        else:
            return self.obj_func(X) * -1
    
    def best(self):
        scores = self.fitness(self.crows)
        ind = np.argmax(scores)
        if self.best_score == None or self.best_score < scores[ind]:
            self.best_crow = self.crows[ind]
            self.best_score = scores[ind]
    
    def move(self, rounds, rnd):
        r = np.random.rand(self.crows.shape[0], self.crows.shape[1])
        d = np.expand_dims(np.random.rand(self.crows.shape[0]), axis=1)
        j = np.array(range(self.numOfCrows))
        np.random.shuffle(j)
        move1 = self.crows + r * self.FL * (self.memory[j] - self.crows)
        move2 = self.data_func(self.numOfCrows)
        self.crows = np.where(d > self.AP, move1, move2)
    
    def update(self):
        fit1 = np.expand_dims(self.fitness(self.crows), axis=1)
        fit2 = np.expand_dims(self.fitness(self.memory), axis=1)
        self.memory = np.where(fit1 > fit2, self.crows, self.memory)
    
    def start(self, rounds):
        for rnd in range(rounds):
            self.move(rounds, rnd)
            self.update()
            self.best()
            
        return self.best_crow
