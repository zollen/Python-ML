'''
Created on Apr 10, 2024

@author: STEPHEN
@url: https://www.baeldung.com/cs/firefly-algorithm
@url: https://en.wikipedia.org/wiki/Firefly_algorithm
@desc

Generation an initial population of fireflies x = {1...n}
Formulate light intensity I so that it is assoicated with f(x)
Define absorption coefficient y

r = distance between firefiles[i] and firefiles[j]
e = gaussion or other distribution zae
x{t+1} = x{t} + beta * e^(-gramma * r{i,j}^2) * (firefiles[j] - firefiles[i]) + alpha{t} * e{t}

while curr_iter < max_iter:
    for i in {1..n}:
        for j in {1..n}:
            if I(i) < I(j):
                fireflies[i] move forward to fireflies[j]
                Evaluate new solutions and update light intensity
                
Fine the best solution
    
'''

import numpy as np


class Fireflies:
    
    def __init__(self, fitness, data_func, direction, numOfFireflies, alpha=2, beta=2, gamma=0.97, LB = -5, UB = 5):
        self.fitness = fitness
        self.data_func = data_func
        self.direction = direction
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.LB = LB
        self.UB = UB
        self.numOfFireFlies = numOfFireflies
        self.fireflies = self.data_func(self.numOfFireFlies)
        
    def best(self):
        scores = self.fitness(self.fireflies)
        if self.direction == 'max':
            return np.argmax(scores)
        else:
            return np.argmin(scores)
        
    def randomwalk(self):
        return np.random.uniform(self.LB, self.UB, size=(self.numOfFireFlies, self.fireflies.shape[1]))
    
    def moveforward(self, target):
        distance = np.linalg.norm(self.fireflies - target)
        moves =  self.fireflies + self.beta * np.exp(-self.gamma*(distance**2)) * (target - self.fireflies) + self.alpha * (np.random.uniform(0, 1) - 0.5)
        moves = np.where(moves > self.LB, self.fireflies, self.LB)
        moves = np.where(moves < self.UB, self.fireflies, self.UB)
        return moves
    
    def start(self, rounds):
        for _ in range(rounds):
            for i in range(self.numOfFireFlies):
                scores = np.expand_dims(self.fitness(self.fireflies), axis=1)
                target = self.fireflies[i]
                X1 = self.moveforward(self.fireflies[i])
                X2 = self.randomwalk()
                self.fireflies = np.where(scores < scores[i], X1, self.fireflies)
                self.fireflies = np.where(scores != scores[i], self.fireflies, X2)
                self.fireflies[i] = target
                
        return self.fireflies[self.best()]        
        
