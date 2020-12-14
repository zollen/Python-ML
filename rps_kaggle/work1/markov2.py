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


SIGNS = [ 'ROCK', 'PAPER', 'SCISSOR']
ROCK = 0
PAPER = 1
SCISSOR = 2

def position(x0, x1, x2, x3, x4, x5):
    return x0 * 243 + x1 * 81 + x2 * 27 + x3 * 9 + x4 * 3 + x5

moves = []


def markov_move(observation, configuration):

    global moves
    
    try:

        if observation.step > 0:
            moves.append(observation.lastOpponentAction)
            
        print("CURRENT MOVES: ", moves)
 
        if len(moves) <= 6:
            return np.random.randint(0, configuration.signs)
    
        if len(moves) > 500:
            moves.pop(0)
            
        numMoves = len(moves)

        initials = np.zeros(2187).astype('float64')
        transitions = np.zeros((2187, 2187)).astype('float64')
    
        initials[position(moves[0], moves[1], moves[2], moves[3], moves[4], 
                          moves[5])] = 1
    
        for index in range(6, numMoves):  
            dest = position(
                            moves[index - 5], moves[index - 4], moves[index - 3], 
                            moves[index - 2], moves[index - 1], moves[index])
            src = position( 
                            moves[index - 6], moves[index - 5], moves[index - 4], 
                            moves[index - 3], moves[index - 2], moves[index - 1])
            transitions[dest, src] = transitions[dest, src] + 1

        for col in range(0, 2187):
            transitions[:, col] = 0 if transitions[:, col].sum() == 0 else transitions[:, col] / transitions[:, col].sum()
        
        '''
        Critical bug fix numMoves - 5, not numMoves - 6
        '''
        probs = np.matmul(np.linalg.matrix_power(transitions, numMoves - 5), initials)    
         
        res = np.argwhere(probs == np.amax(probs)).ravel()
        
        nextMove = np.random.choice(res).item() % configuration.signs
        
        print("PREDICT MOVE: [%s]" % SIGNS[nextMove])
        
        return (nextMove + 1) % configuration.signs

    except:
        traceback.print_exc(file=sys.stderr)
    
    

moves = [ 2, 2, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0, 2 ]


class observationCls:
    step = len(moves) + 1
    lastOpponentAction = 2
class configurationCls:
    signs = 3
   

observation = observationCls()
configuration = configurationCls()
t_start = time.perf_counter_ns()
result = markov_move(observation, configuration)
t_end = time.perf_counter_ns()
print("MY NEXT MOVE: [%s] ==> %d ns" % (SIGNS[result], t_end - t_start))   


   
    