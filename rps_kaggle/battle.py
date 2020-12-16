'''
Created on Dec. 16, 2020

@author: zollen
'''

import rps_kaggle.lib.rps_lib as rps
import time

SIGNS = ['ROCK', 'PAPER', 'SCISSORS']


player1 = rps.GMarkov(3, 6)
player2 = rps.NOrderMarkov(3, 6)

results = []
for _ in (0, 1000):
    move1 = player1.predict()
    move2 = player2.predict()
    
    res = move1 - move2
    res1 = abs(res)
    
    if res1 == 0:
        winner = "TIE"
    elif res1 ==  1:
        if res > 0:
            winner = "GMarkov(3, 6)"
    elif res1 == 2:
        pass

