'''
Created on Apr 2, 2024

@author: STEPHEN
@url: https://www.youtube.com/watch?v=2by9CN8QIpw
@title: Moth Flame Optimization 

M(i): ith Moth
F(j): jth flame

Spiral function: S(M(i), F(j)) = D(i) * e^(b * t) * cos(2 * pi * t) + F(j)

Flame = best moth position
D(i) = | F(j) - M(i) |           <-- distance between flame and moth
b is a constant for defining shape of logarithmic sprial
t is a random number between -1 and 1


Number of flames during each iteration is defined as:
-----------------------------------------------------
N = max number of flames
T = max number of Iterations
x = current iteration
number of flames = round( N - x * (N - 1) / T )


# First moth: always update its position w.r.t best flame. After updating list of flames,
# the flames are sorted based on their fitness values. So that moth can update their position
# w.r.t corresponding flame
# Last moth: always update its position w.r.t Worst Flame.
# Specific flames are provides to each moth to prevent local stability
    
Algorithm#1
===========
Initialize all important parameters
Initialize moths positions randomly in the search place

while iteration < max_iteration:

    Calculate fitness values for each moths
    Calculate the number of flames and update flames
    
    For i in {1..max_moths}
        for j in {1..max_flames}
            Update r and t
            Calculate distance for corresponding moths - D(i)
            Update moths position                      - S(M(i), F(j))
        End
    End
Display best solution


Algorithm#2
===========
1. Generating the initial population of moths    - (ub(i) - lb(j)) * rand() + lb(i)
2. Updating the moth positions                   - S(M(i), F(j))
3. Updating the flame number                     - round( N - 1 * (N - x) / T )

'''

import numpy as np


class MothsFlame:
    
    def __init__(self, fitness, data_func, direction, numOfMoths, maxFlames = 0.4, spiral = 1):
        self.fitness = fitness
        self.data_func = data_func
        self.numOfMoths = numOfMoths
        self.direction = direction
        self.spiral = spiral
        self.numOfFlames = 0
        self.maxFlames = int(self.numOfMoths * maxFlames)
        self.moths = self.data_func(self.numOfMoths)
     
    def updatePositions(self, t, flames):
        self.moths = np.abs(flames - self.moths) * \
                        np.power(np.e, self.spiral * t) * \
                        np.cos(2 * np.pi * t) + flames
    
    def calculateFlames(self, rounds, rnd):
        self.numOfFlames = int(np.round(self.maxFlames - rnd * (self.maxFlames - 1) / rounds))
        if self.direction == 'max':
            ind = np.argpartition(self.fitness(self.moths), -self.numOfFlames)[-self.numOfFlames:]
        else:
            ind = np.argpartition(self.fitness(self.moths), self.numOfFlames)[:self.numOfFlames]
        flames = np.tile(self.moths[ind], 
                 (int(np.ceil(self.numOfMoths / self.numOfFlames)), 1))[0:self.numOfMoths]
        flames[ind] = self.moths[ind]
        return flames
    
    def start(self, rounds):
        for rnd in range(rounds):
            r = 1 - rnd * (0.99) / rounds 
            t = np.random.uniform(-r, r, (self.numOfMoths, self.moths[0].size))
            flames = self.calculateFlames(rounds, rnd)
            self.updatePositions(t, flames)

        return self.moths[np.argmax(self.fitness(self.moths))]
            


