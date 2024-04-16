'''
Created on Apr 16, 2024

@author: STEPHEN
@url: https://www.youtube.com/watch?v=USD2IzkIb2M
@url: https://www.codeproject.com/Articles/5369245/Cheetah-Optimizer-Python-Implementation

Cheetahs behave one of the four ways

1. Attack
    Rushing: When the cheetah decides to attack, they rush toward the prey with maximum speed.
    Capturing: The cheetah used speed and flexibility to capture the prey by approaching the prey.

    r{i,j}: random number between [0,1]
    β{t,i,j}: e^(abs(r{i,j}) * sin(2 * pi * r{i,j})
    
    X{t+1,i,j} = X{t,b,j} + r{i,j} * β{t,i,j}
    
2. Sit and Wait
    After the prey is detected, but the situation is not proper, cheetahs may sit and wait 
    for the prey to come nearer or for the position to be better

    X{t+1,i,j} = X{t,i,j}

3. Search
    Cheetahs need to search, including scanning or active search, in their territories 
    (search space) or the surrounding area to find their prey.

    r{i,j}: random number between [0,1]
    t: current hunting time - upper limit
    T: max hunting time - lower limit
    α{t,i,j}: step length of cheetah i.  α{t,i,j} = 0.001 * t / T
    
    X{t+1,i,j} = X{t,i,j} + r{i,j} * α{t,i,j}

4. Leave the prey and go home
    Two situations are considered for this strategy. (1) If the cheetah is unsuccessful
    in hunting the prey, it should change its position or return to its territory. (2) In 
    cases with no successful hunting action in some time interval, it may change its position 
    to the last prey detected and searched around it.
'''

import numpy as np

class Cheetahs:
    
    def __init__(self, obj_func, data_func, direction, numOfCheetahs, 
                 alpha = 0.5, beta = 0.5, delta = 0.5):
        self.obj_func = obj_func
        self.data_func = data_func
        self.direction = direction
        self.numOfCheetahs = numOfCheetahs
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.cheetahs = self.data_func(self.numOfCheetahs)
        self.best_cheetah = None
        self.best_score = None
        
    def fitness(self, X):
        if self.direction == 'max':
            return self.obj_func(X)
        else:
            return self.obj_func(X) * -1
        
    def best(self):
        scores = self.fitness(self.cheetahs)
        ind = np.argmax(scores)
        if self.best_score == None or self.best_score < scores[ind]:
            self.best_cheetah = self.cheetahs[ind]
            self.best_score = scores[ind] 
    
    def attack(self):
        rn = np.random.rand(self.cheetahs.shape[0], self.cheetahs.shape[1])
        return self.cheetahs + rn * (self.best_cheetah - self.cheetahs)
    
    def wait(self):
        return self.cheetahs + np.random.uniform(-0.01, 0.01, 
                                        size=(self.cheetahs.shape[0], self.cheetahs.shape[1]))
    
    def search(self, rounds, rnd):
        choices = np.random.choice([0, 1, 2], 
                                    size=(self.cheetahs.shape[0], self.cheetahs.shape[1]))
        ones = np.ones((self.cheetahs.shape[0], self.cheetahs.shape[1]))
        cos_che = ones * np.cos((rnd + 1) / rounds)
        sin_che = ones * np.sin((rnd + 1) / rounds)
        tan_che = ones * np.tan((rnd + 1) / rounds)
        alpha = np.where(choices == 0, 
                          cos_che,
                          np.where(choices == 1, sin_che, tan_che))
        return self.best_cheetah + alpha * (self.best_cheetah - self.cheetahs)
    
    def leave(self):
        home = np.tile(self.cheetahs[0], (self.numOfCheetahs, 1))
        return home + np.random.uniform(-self.delta, self.delta, 
                                    size=(self.cheetahs.shape[0], self.cheetahs.shape[1]))
    
    def start(self, rounds):
        self.best()
        for rnd in range(rounds):
            r = np.random.rand()
            s = np.random.rand()
            if r < self.alpha:
                if s < 0.5:
                    self.cheetahs = self.search(rounds, rnd)
                else:
                    self.cheetahs = self.wait()
            elif r < self.alpha + self.beta:
                self.cheetahs = self.attack()
            else:
                self.cheetahs = self.leave()            
            self.best()
        return self.best_cheetah