'''
Created on Dec. 11, 2020

@author: zollen
'''

import numpy as np
import time
import traceback
import sys

np.random.seed(int(round(time.time())))
np.set_printoptions(precision=6)


SIGNS = [ 'ROCK', 'PAPER', 'SCOSSOR']
ROCK = 0
PAPER = 1
SCISSOR = 2


moves = []

def markov_move(observation, configuration):

    global moves
    
    try:

        if observation.step > 0:
            moves.append(observation.lastOpponentAction)
 
        if len(moves) <= 2:
            return np.random.randint(0, configuration.signs)
    
        if len(moves) > 600:
            moves.pop(0)
            
        numMoves = len(moves)

        initials = np.zeros(configuration.signs).astype('float64')
        transitions = np.zeros((configuration.signs, configuration.signs)).astype('float64')
    
        initials[moves[0]] = 1
    
        for index in range(1, numMoves):  
            transitions[moves[index], moves[index - 1]] = transitions[moves[index], moves[index - 1]] + 1

        for col in range(0, configuration.signs):
            transitions[:, col] = 0 if transitions[:, col].sum() == 0 else transitions[:, col] / transitions[:, col].sum()
        
        probs = np.matmul(np.linalg.matrix_power(transitions, numMoves), initials)    

        res = np.argwhere(probs == np.amax(probs)).ravel()
        
        '''
        unpack numpy.int64 back to int64
        '''
        nextMove = np.random.choice(res).item()
        
        '''
        Infinite/Trend: UD^nU^(-1) * P0
        A^n * P0 = Pn
        p = np.matmul(np.matmul(np.matmul(V, D), np.linalg.inv(V), initials)
        '''
   
        return (nextMove + 1) % configuration.signs

    except:
        traceback.print_exc(file=sys.stderr)
    
    

TOTAL = 16
for index in range(0, TOTAL):
    moves.append(np.random.randint(0, 3))

    print([ SIGNS[x] for x in moves])

    class observationCls:
        step = index + 1
        lastOpponentAction = np.random.randint(0, 3)
    class configurationCls:
        signs = 3
       

    observation = observationCls()
    configuration = configurationCls()
    print("MY NEXT MOVE: [%s]" % SIGNS[markov_move(observation, configuration)])   

   
    