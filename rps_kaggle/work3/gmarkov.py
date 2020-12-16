'''
Created on Dec. 16, 2020

@author: zollen
'''
import numpy as np
import time

class GMarkov:
    
    DEFALT_MIN_MOVES = 3
    
    def __init__(self, states, num_of_moves = DEFALT_MIN_MOVES):
        np.random.seed(int(round(time.time())))
        self.moves = np.array([])
        self.mines = np.array([])
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
        
    def submit(self, token):
        self.mines = np.append(self.mines, token)
        return token
        
    def __str__(self):
        return "GMarkov(" + str(self.numMoves) + ")"
    
    def predict(self):
        
        totalMoves = len(self.moves)
        
        if totalMoves <= self.numMoves:
            return self.submit(np.random.randint(0, self.states))

        for _ in range(0, self.numMoves):
            self.transitions.append(np.zeros((self.dimen, self.dimen)).astype('float64'))
        
        for index in range(0, totalMoves - self.numMoves):
            submoves = self.moves[index:index + self.numMoves + 1]
            submines = self.mines[index:index + self.numMoves + 1]
       
            length = len(submoves)

            for subindex in range(0, length - 1):
                dest = int(submoves[-1])
                src = int(submoves[subindex])
                self.transitions[subindex][src, dest] = self.transitions[subindex][src, dest] + 1
      
                res = submines[subindex] - submoves[subindex]
                if res == 1 or res == -2:
                    self.transitions[subindex][src, dest] = self.transitions[subindex][src, dest] + 1
            
                
         
        for subindex in range(0, self.numMoves):
            for row in range(0, self.dimen):
                self.transitions[subindex][row, :] = 0 if self.transitions[subindex][row, :].sum() == 0 else self.transitions[subindex][row, :] / self.transitions[subindex][row, :].sum()
                   
        
        submoves = self.moves[totalMoves - self.numMoves:]

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
            
        return self.submit(best_move)
    
    
markov = GMarkov(3, 2)


for token in [ 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2 ]:
    markov.add(token)
    markov.predict()

SIGNS = [ 'ROCK', 'PAPER', 'SCISSOR']


t_start = time.perf_counter_ns()
nextMove = markov.predict()

t_end = time.perf_counter_ns()

print("PREDICTED MOVE: [%s] ==> %d ns" % (SIGNS[nextMove], t_end - t_start))