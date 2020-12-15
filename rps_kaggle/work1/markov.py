'''
Created on Dec. 11, 2020

@author: zollen
'''

import rps_kaggle.lib.rps_lib as rps
import numpy as np
import time


np.random.seed(int(round(time.time())))



SIGNS = [ 'ROCK', 'PAPER', 'SCISSOR']


'''
3 states - 0, 1, 2
6 orders - keeping track of the transition probabilties of six moves
'''
markov = rps.NPowerMarkov(3, 6)

def markov_move(observation, configuration):

    global markov
    
    if observation.step > 0:
        markov.add(observation.lastOpponentAction)
        
    return markov.predict()
    
    


class observationCls:
    step = 100
    lastOpponentAction = [ 2, 2, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0, 2, 2 ]
class configurationCls:
    signs = 3
   

observation = observationCls()
configuration = configurationCls()
t_start = time.perf_counter_ns()
result = markov_move(observation, configuration)
t_end = time.perf_counter_ns()
print(markov)
print("MY NEXT MOVE: [%s] ==> %d ns" % (SIGNS[result], t_end - t_start))   


   
    