'''
Created on Jan. 16, 2021

@author: zollen
'''

import numpy as np
import time
import math
import random


class Flatter:
    
    def __init__(self, states = 3, flatten = 0.7, offset = 2.0, halfLife = 100.0):
        self.countInc = 1e-30
        self.countOp  = self.countInc * np.ones((states, states, states))
        self.countAg  = self.countInc * np.ones((states, states, states))
        self.histAgent = []    # Agent history
        self.histOpponent = [] # Opponent history
        self.nwin = 0
        self.states = states
        self.pFlatten = flatten
        self.offset = offset
        self.halfLife = halfLife
        self.countPow = math.exp(math.log(2) / halfLife)
        self.reward  = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
        self.rnd = 0
        
    def __str__(self):
        return "Flatter"
    
    def add(self, token):
        self.histOpponent.append(token)
        self.nwin += self.reward[self.histAgent[-1], token]
        
    def decide(self):
        
        if self.rnd == 0:
            dist = np.ones(self.states)
        else:
            if self.rnd > 1:
                self.countOp[self.histOpponent[-2], self.histAgent[-2], self.histOpponent[-1]] += self.countInc
                self.countAg[self.histOpponent[-2], self.histAgent[-2], self.histAgent[-1]] += self.countInc
                        
            # decide on what strategy to play
            if len(self.histOpponent) < 2:
                dist = np.ones(self.states)
            else:
                if random.random() < self.pFlatten:
                    # stochastically flatten the distribution
                    count = self.countAg[self.histOpponent[-1], self.histAgent[-1]]
                    dist  = (self.offset + 1) * count.max() - self.offset * count.min() - count
                else:
                    # simple prediction of opponent
                    count = self.countOp[self.histOpponent[-1], self.histAgent[-1]]
                    gain  = np.dot(self.reward, count)
                    dist  = gain + gain.min()
              
        agentAction = random.choices(range(self.states), weights=dist)[0]
        self.histAgent.append(agentAction)
        self.countInc *= self.countPow
        self.rnd += 1
        
        return(agentAction)