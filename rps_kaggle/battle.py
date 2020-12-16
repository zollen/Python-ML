'''
Created on Dec. 16, 2020

@author: zollen
'''

import rps_kaggle.lib.rps_lib as rps
import pprint
import time

SIGNS = ['ROCK', 'PAPER', 'SCISSORS']
pp = pprint.PrettyPrinter(indent=3) 

player1 = rps.GMarkov(3, 6)
player2 = rps.NOrderMarkov(3, 6)

results = []
win1 = 0
win2 = 0

for _ in range(0, 1000):

    move1 = player1.predict()
    move1 = (move1 + 1) % 3
    move2 = player2.predict()
    move2 = (move2 + 1) % 3
    
    player1.add(move2)
    player2.add(move1)
    
    res = move1 - move2
    
    winner = "TIE"
    
    if res ==  1 or res == -1:
        if res > 0:
            winner = player1
        else:
            winner = player2
    elif res == 2 or res == -2:
        if res > 0:
            winner = player2
        else:
            winner = player1
            
    if winner == player1:
        win1 = win1 + 1
    elif winner == player2:
        win2 = win2 + 1

    
    msg = "{:>8} | {:<8} => {:^20}"
    print(msg.format(SIGNS[move1], SIGNS[move2], winner.__str__()))
    results.append(msg.format(SIGNS[move1], SIGNS[move2], winner.__str__()))

if win1 == win2:
    print("BOTH TIE!!")
elif win1 > win2:
    print(player1.__str__() + " WON!!")
else:
    print(player2.__str__() + " WON!!")
