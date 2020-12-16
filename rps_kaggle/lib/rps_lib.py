'''
Created on Dec. 15, 2020

@author: zollen
'''
import numpy as np
import time

class NOrderMarkov:
    
    def __init__(self, states, power = 1):
        np.random.seed(int(round(time.time())))
        self.moves = np.array([])
        self.states = states
        self.power = power
        self.dimen = np.power(self.states, self.power)

        
    def positions(self, vals):
        
        total = 0

        for index in range(0, self.power):
            total = total + vals[index] * np.power(self.states, self.power - index - 1)

        return int(total.item())
    
    def add(self, token):
        self.moves = np.append(self.moves, token)
        
    def __str__(self):
        return np.array2string(self.moves)
        
    def predict(self):
           
        totalMoves = len(self.moves)
        
        if totalMoves <= self.power:
            return np.random.randint(0, self.states)
        
        initials = np.zeros(self.dimen).astype('float64')
        transitions = np.zeros((self.dimen, self.dimen)).astype('float64')
        
        initials[self.positions(self.moves[0:self.power])] = 1
        for index in range(self.power, totalMoves):  
            dest = self.positions(self.moves[index - self.power + 1:index + 1])
            src = self.positions(self.moves[index - self.power:index])
            transitions[dest, src] = transitions[dest, src] + 1
        
        for col in range(0, self.dimen):
            transitions[:, col] = 0 if transitions[:, col].sum() == 0 else transitions[:, col] / transitions[:, col].sum()
            
        probs = np.matmul(np.linalg.matrix_power(transitions, totalMoves - self.power + 1), initials)    
         
        res = np.argwhere(probs == np.amax(probs)).ravel()
        
        return np.random.choice(res).item() % self.states

    
    


