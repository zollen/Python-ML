'''
Created on Dec. 14, 2020

@author: zollen
'''

import rps_kaggle.lib.rps_lib as rps
import time




'''
3 states - 0, 1, 2
6 orders - keeping track of the transition probabilties of six moves
'''
markov = rps.GMarkov(3, 6)

markov.add([ 2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0 ])

SIGNS = [ 'ROCK', 'PAPER', 'SCISSOR']

print(markov.lambdas)
t_start = time.perf_counter_ns()
nextMove = markov.predict()
t_end = time.perf_counter_ns()

print("PREDICTED MOVE: [%s] ==> %d ns" % (SIGNS[nextMove], t_end - t_start))