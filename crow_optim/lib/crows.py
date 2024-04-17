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
    
    def __init__(self, obj_func, data_func, direction, numOfCrows, AP = 0.5):
        self.obj_func = obj_func
        self.data_func = data_func
        self.direction = direction
        self.numOfCrows = numOfCrows
        self.AP = AP
        self.crows = self.data_func(self.numOfCrows)
        self.best_crow = None
        self.best_score = None
    
    def fitness(self, X):
        if self.direction == 'max':
            return self.obj_func(X)
        else:
            return self.obj_func(X) * -1
    
    def best(self):
        scores = self.fitness(self.crows)
        id = np.argmax(scores)
        if self.best_score == None or self.best_score < scores[id]:
            self.best_crow = self.crows[id]
            self.best_score = scores[id]
    
    def move(self):
        pass
    
    def update(self):
        pass
    
    def start(self, rounds):
        pass
