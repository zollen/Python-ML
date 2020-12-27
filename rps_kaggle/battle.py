'''
Created on Dec. 16, 2020

@author: zollen
'''

import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import rps_kaggle.lib.rps_lib as rps
import warnings


warnings.filterwarnings('ignore')

SIGNS = ['ROCK', 'PAPER', 'SCISSORS']


class ShellClassifier:
    
    def __init__(self, classifier, states = 3, beat = 1):
        self.classifier = classifier
        self.mines = 0
        self.opponent = 0
        self.states = states
        self.beat = beat
        
    def __str__(self):
        return "<" +self.classifier.__str__() + ">{" + self.beat + "}"
    
    def reset(self):
        self.classifier.reset()
    
    def deposit(self, token):
        self.mines = np.append(self.mines, token)
        self.classifier.deposit(token)
        
    def add(self, token):
        self.opponent += 1
        if len(self.classifier.opponent) < self.opponent:
            self.classifier.add(token)
    
    def decide(self):
        if self.classifier.last != None:
            return (self.classifier.last + self.beat) % self.states
       
        return (self.classifier.decide(False) + self.beat) % self.states
    
    






exit()

forest1 = rps.Classifier(RandomForestClassifier(random_state = 23, n_estimators = 10), window = 10)
xgb1 = rps.Classifier(XGBClassifier(random_state = 26, n_estimators = 10, eval_metric = 'logloss'), window = 10)
forest2 = rps.Classifier(RandomForestClassifier(random_state = 37, n_estimators = 10), window = 10, beat = 2)
xgb2 = rps.Classifier(XGBClassifier(random_state = 43, n_estimators = 10, eval_metric = 'logloss'), window = 10, beat = 2)
forest3 = rps.Classifier(RandomForestClassifier(random_state = 51, n_estimators = 10), window = 10, beat = 0)
xgb3 = rps.Classifier(XGBClassifier(random_state = 53, n_estimators = 10, eval_metric = 'logloss'), window = 10, beat = 0)

agents = [
            [ rps.Randomer(), [1, 1]                      ],
            [ rps.MirrorOpponentDecider(beat = 0), [1, 1] ],
            [ rps.MirrorOpponentDecider(beat = 1), [1, 1] ],
            [ rps.MirrorOpponentDecider(beat = 2), [1, 1] ],
            [ rps.MirrorSelfDecider(beat = 0), [1, 1]     ],
            [ rps.MirrorSelfDecider(beat = 1), [1, 1]     ],
            [ rps.MirrorSelfDecider(beat = 2), [1, 1]     ],
            [ forest1, [1, 1]                             ],
            [ xgb1, [1, 1]                                ],
            [ forest2, [1, 1]                             ],
            [ xgb2, [1, 1]                                ],
            [ forest3, [1, 1]                             ],
            [ xgb3, [1, 1]                                ]
        ]
    

'''
OClassifier(XGBClassifier) window = 14
'''
'''
WIN : 0
LOST: 0
'''
player1 = rps.BetaAgency(agents)
player2 = rps.Classifier(XGBClassifier(random_state = 17, n_estimators = 10, eval_metric = 'logloss'), 
                         window = 10)


#player2 = rps.MirrorSelfDecider(ahead = 2)
#player2 = rps.NMarkov(3, 6)
#player2 = rps.Randomer()

results = []
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


    msg = "[{:<4}]   {:>8} | {:<8} => {:>50}  {:>4}|{:<4} {}"
    print(msg.format(rnd + 1, SIGNS[move1], SIGNS[move2], winner.__str__(), win1, win2, t_end - t_start))


print("====================================================================")
if win1 == win2:
    print("BOTH TIE!!")
elif win1 > win2:
    print("PLAYER1 {}, PLAYER2 {}  RATIO {:2.4f}".format(win1, win2, win1 / win2))
    print("Player1: [{}] WON!!!!!".format(player1))
else:
    print("PLAYER1 {}, PLAYER2 {} RATIO {:2.4f}".format(win1, win2, win1 / win2))
    print("Player1: [{}] LOST!!!!!".format(player1))
