'''
Created on Jan. 20, 2021

@author: zollen
'''

import numpy as np
import itertools
import pprint


pp = pprint.PrettyPrinter(indent=3) 

a = [ x for x in range(4) ]

table = []
for ll in range(0, len(a)):
    for subset in itertools.combinations(a, ll):
        if len(subset) > 0:
            table.append(subset)
            
table.append(tuple(a))
            
# 4 scorers
# 2 agents

predictions = [
        0,  # agent1 decided 2(SCISSORS)    <- lost
        2,  # agent2 decided 0(ROCK)        <- won
        1   # agent3 decided 1(PAPER)       <- event
    ]

scorers = [
        [ 0.1, 0.2, 0.3 ],  # scorer1 chose agent3 <-- even
        [ 0.4, 0.3, 0.1 ],  # scorer2 chose agent1 <-- lost
        [ 0.4, 0.5, 0.4 ],  # scorer3 chose agent2 <-- won
        [ 0.7, 0.8, 0.5 ]   # scorer4 chose agent2 <-- won
    ]

mines = [ 2 ]
opponent = [ 1 ]
outcome = [ -1 if (x - y) % 3 == 2 else x - y for x, y in zip(mines, opponent) ]

result1 = {}
result2 = {}
for combo in table:
    scores = [ 0 ] * len(predictions)
    winOnly = True
    winAndEvenOnly = True
    for entry in combo:
        flag = (predictions[np.argmax(scorers[entry])] - opponent[-1]) % 3
        if flag == 0:
            winOnly = False
        if flag == 2:
            winOnly = False
            winAndEvenOnly = False
            break
        scores = [ x + (y * flag) for x, y in zip(scores, scorers[entry]) ]
    if winOnly == True:
        result1[combo] = scores
    if winAndEvenOnly == True:
        result2[combo] = scores
        
pp.pprint(result1)
print("=======================")
pp.pprint(result2)

best_score = -1
best_combo = None        
for combo, scores in result1.items():
    score = np.std(scores)
    if score > best_score:
        best_score = score
        best_combo = combo
    


print("LAST MINE ", mines[-1])
print("LAST OPP ", opponent[-1])
print("LAST OUTCOME ", outcome[-1])
print("BEST COMBO: ", best_combo, " ==> ", result1[best_combo])