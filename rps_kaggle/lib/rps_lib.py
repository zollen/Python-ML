'''
Created on Dec. 15, 2020

@author: zollen
'''
import numpy as np
import time


class Randomer:
    
    def __init__(self):
        np.random.seed(int(round(time.time())))
        pass
    
    def add(self, token):
        pass
        
    def __str__(self):
        return "Randomer()"
    
    def predict(self):
        return np.random.randint(0, 3)
        
    
    
    
    
'''
High order Markov. It holds a sequence of transitions (as oppose to just a single transition 
in the transition matrix
'''
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
        return "NOrderMarkov(" + str(self.power) + ")"
        
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
        return "GMarkov(" + str(self.numMoves) + ")"
    
    def predict(self):
        
        totalMoves = len(self.moves)
        
        if totalMoves <= self.numMoves:
            return np.random.randint(0, self.states)

        for _ in range(0, self.numMoves):
            self.transitions.append(np.zeros((self.dimen, self.dimen)).astype('float64'))
        
        for index in range(0, totalMoves - self.numMoves - 1):
            submoves = self.moves[index:index + self.numMoves + 1]
            length = len(submoves)
            for subindex in range(0, length - 1):
                dest = int(submoves[-1])
                src = int(submoves[subindex])
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
            
        return best_move
    
    


