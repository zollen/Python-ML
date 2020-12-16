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

https://math.stackexchange.com/questions/362412/generating-a-monotonically-decreasing-sequence-that-adds-to-1-for-any-length
for generating decreasing sequence that adds up to 1
'''


class GMarkov:
    
    DEFALT_MIN_MOVES = 3
    
    def __init__(self, states, num_of_moves = DEFALT_MIN_MOVES):
        np.random.seed(int(round(time.time())))
        self.moves = np.array([])
        self.states = states
        self.dimen = np.power(self.states, 1)
        self.numMoves = num_of_moves
        self.lambdas = self.priors()
        self.transitions = []
        
    def priors(self):
        
        seq = []
        for index in range(self.numMoves, 0, -1):
            seq.append((2 * index - 1) / (self.numMoves * self.numMoves))
            
        return seq
    
    def add(self, token):
        self.moves = np.append(self.moves, token)
        
    def __str__(self):
        return np.array2string(self.moves)
    
    def predict(self):
        
        totalMoves = len(self.moves)
        
        if totalMoves <= self.numMoves:
            return np.random.randint(0, self.states)
        
        for _ in range(0, self.numMoves - 1):
            self.transitions.append(np.zeros((self.dimen, self.dimen)).astype('float64'))
        
        for index in range(0, totalMoves - self.numMoves + 1):
            submoves = self.moves[index:index + self.numMoves]
            for subindex in range(0, self.numMoves - 1):
                dest = int(submoves[self.numMoves - 1])
                src = int(submoves[subindex])
                self.transitions[subindex][src, dest] = self.transitions[subindex][src, dest] + 1
         
        for subindex in range(0, self.numMoves - 1):
            for row in range(0, self.dimen):
                self.transitions[subindex][row, :] = 0 if self.transitions[subindex][row, :].sum() == 0 else self.transitions[subindex][row, :] / self.transitions[subindex][row, :].sum()
               
   
        submoves = self.moves[totalMoves - self.numMoves + 1:totalMoves]

        best_score = -99
        best_move = 0
        for target in range(0, self.states):
            prob = 0.0
            for subindex in range(0, len(submoves)):
                dest = target
                src = int(submoves[subindex])
                prob += self.transitions[subindex][src, dest] * self.lambdas[subindex] 
 
            if prob > best_score:
                best_score = prob
                best_move = target
            
        return best_move
        
markov = GMarkov(3, 6)

markov.add([ 2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0])

SIGNS = [ 'ROCK', 'PAPER', 'SCISSOR']

print(markov.lambdas)
t_start = time.perf_counter_ns()
nextMove = markov.predict()
t_end = time.perf_counter_ns()

print("PREDICTED MOVE: [%s] ==> %d ns" % (SIGNS[nextMove], t_end - t_start))