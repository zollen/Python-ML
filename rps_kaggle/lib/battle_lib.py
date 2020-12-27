'''
Created on Dec. 27, 2020

@author: zollen
'''

import time


SIGNS = ['ROCK', 'PAPER', 'SCISSORS']

def battleground(player1, player2, rnd = 1, verbose = True):
    
    win1 = 0
    win2 = 0
    
    for rnd in range(0, 1000):
        
        t_start = time.perf_counter_ns()
        move1 = player1.decide()
        t_end = time.perf_counter_ns()
        
        move2 = player2.decide()
        
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
    
        if verbose:
            msg = "[{:<4}]   {:>8} | {:<8} => {:>40}  {:>4}|{:<4} {}"
            print(msg.format(rnd + 1, SIGNS[move1], SIGNS[move2], winner.__str__()[-40:], win1, win2, t_end - t_start))
    
    
    if verbose:
        print("====================================================================")
        if win1 == win2:
            print("BOTH TIE!!")
        elif win1 > win2:
            print("PLAYER1 {}, PLAYER2 {}  RATIO {:2.4f}".format(win1, win2, win1 / win2))
            print("Player1: [{}] WON!!!!!".format(player1))
        else:
            print("PLAYER1 {}, PLAYER2 {} RATIO {:2.4f}".format(win1, win2, win1 / win2))
            print("Player1: [{}] LOST!!!!!".format(player1))
        
    return win1, win2
