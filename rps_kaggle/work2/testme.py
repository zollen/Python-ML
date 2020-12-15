'''
Created on Dec. 14, 2020

@author: zollen
'''
import numpy as np
import time
import traceback
import sys

np.random.seed(int(round(time.time())))

'''
https://towardsdatascience.com/mixture-transition-distribution-model-e48b106e9560
Borrowing the concept of generalized mixture transaction to capture sequence information
then flatten it into a pandas dataframe
'''


class GMarkov:
    
    DEFALT_MIN_MOVES = 3
    
    def __init__(self, states, num_of_moves = DEFALT_MIN_MOVES):
        np.random.seed(int(round(time.time())))
        self.moves = np.array([])
        self.states = states
        self.dimen = np.power(self.states, 1)
        self.numMoves = num_of_moves
        self.lambdas = np.array([1.0, 1.0, 1.0, 1.0])
        self.transitions = []

    def add(self, token):
        self.moves = np.append(self.moves, token)
        
    def __str__(self):
        return np.array2string(self.moves)
    
    def predict(self):
        
        totalMoves = len(self.moves)
        
        if totalMoves <= self.DEFALT_MIN_MOVES:
            return np.random.randint(0, self.states)
        
        for _ in range(0, self.numMoves - 1):
            self.transitions.append(np.zeros((self.dimen, self.dimen)).astype('float64'))
        
        for index in range(0, totalMoves - self.numMoves + 1):
            submoves = self.moves[index:index + self.numMoves]
            for subindex in range(0, self.numMoves - 1):
                dest = int(submoves[self.numMoves - 1])
                src = int(submoves[subindex])
                self.transitions[subindex][src, dest] = self.transitions[subindex][src, dest] + 1
         
        for subindex in range(0, self.numMoves):
            for row in range(0, self.dimen):
                self.transitions[subindex][row, :] = 0 if self.transitions[subindex][row, :].sum() == 0 else self.transitions[subindex][row, :] / self.transitions[subindex][row, :].sum()
               
        prob = 0.0
        submoves = self.moves[totalMoves - self.numMoves:totalMoves]
        print(submoves)
        for subindex in range(0, self.numMoves - 1):
            dest = int(submoves[self.numMoves - 1])
            src = int(submoves[subindex])
            prob += self.transitions[subindex][src, dest] * self.lambdas[subindex] 
            
        return prob
        
markov = GMarkov(3, 4)

markov.add([ 1, 2, 0, 1, 0, 2, 2, 1, 2, 0, 0, 1, 1 ])

print(markov.predict())