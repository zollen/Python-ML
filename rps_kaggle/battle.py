'''
Created on Dec. 16, 2020

@author: zollen
'''

import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import rps_kaggle.lib.rps_lib as rps
import warnings
from rps_kaggle.lib.rps_lib import StandardCounterMover





warnings.filterwarnings('ignore')

SIGNS = ['ROCK', 'PAPER', 'SCISSORS']

print("forest1")
forest1 = rps.Classifier(RandomForestClassifier(random_state = 23, n_estimators = 10), window = 10)
print("xgb1")
xgb1 = rps.Classifier(XGBClassifier(random_state = 26, n_estimators = 10, eval_metric = 'logloss'), window = 10)
print("forest2")
forest2 = rps.Classifier(RandomForestClassifier(random_state = 37, n_estimators = 10), window = 10, beat = 2)
print("xgb2")
xgb2 = rps.Classifier(XGBClassifier(random_state = 43, n_estimators = 10, eval_metric = 'logloss'), window = 10, beat = 2)
print("forest3")
forest3 = rps.Classifier(RandomForestClassifier(random_state = 51, n_estimators = 10), window = 10, beat = 0)
print("xgb3")
xgb3 = rps.Classifier(XGBClassifier(random_state = 53, n_estimators = 10, eval_metric = 'logloss'), window = 10, beat = 0)

agents = {
    rps.Randomer(): [1, 1],
    rps.MirrorOpponentDecider(beat = 0): [1, 1],
    rps.MirrorOpponentDecider(beat = 1): [1, 1],
    rps.MirrorOpponentDecider(beat = 2): [1, 1],
    rps.MirrorSelfDecider(beat = 0): [1, 1],
    rps.MirrorSelfDecider(beat = 1): [1, 1],
    rps.MirrorSelfDecider(beat = 2): [1, 1],
    forest1: [1, 1],
    xgb1: [1, 1],
    forest2: [1, 1],
    xgb2: [1, 1],
    forest3: [1, 1],
    xgb3: [1, 1]
    }

class BetaAgency:
    
    def __init__(self, agents):
        print("constructor with agents", agents)
        self.agents = agents
        self.mines = []
        self.opponent = []
        self.executor = None
        print("done constructor")
    
    def __str__(self):
        return "Agency(" + self.executor.__str__() + ")"
    
    def add(self, token):
        for agent, _ in self.agents.items():
            print("adding token: ", agent.__class__.__name__, token)
            agent.add(token)
            
    def submit(self, token):
        for agent, _ in self.agents.items():
            print("submit token: ", agent.__class__.__name__, token)
            if self.executor != agent:
                agent.submit(token)
            
        return token
    
    def lastgame(self, agent):
        
        print("last game: ",agent.__class__.__name__)
        if len(agent.mines) <= 0 or len(agent.opponent) <= 0:
            print("last game: ",agent.__class__.__name__, " return 0")
            return 0
        
        res = (agent.mines[-1] - agent.opponent[-1]) % 3
        if res == 1:
            print("last game: ",agent.__class__.__name__, " return 1")
            return 1
        elif res == 2:
            print("last game: ",agent.__class__.__name__, " return -1")
            return -1
        
        print("last game: ",agent.__class__.__name__, " return 0")
        return 0
        
          
    def decide(self):
        
        for agent, scores in self.agents.items():
            print(agent.__class__.__name__, " => ", scores)
            scores[0] = (scores[0] - 1) / 1.05 + 1
            scores[1] = (scores[1] - 1) / 1.05 + 1
            
            outcome = self.lastgame(agent)
            print(agent.__class__.__name__, " outcome: ", outcome)
            if outcome > 0:
                scores[0] += 3
            elif outcome < 0:
                scores[1] += 3
            else:
                scores[0] = scores[0] + 3/2
                scores[1] = scores[1] + 3/2
                
            print(agent.__class__.__name__, " scores: ", scores)
        
        
        best_prob = -1
        best_agent = None
        best_move = None
        for agent, scores in self.agents.items():
            print(agent.__class__.__name__, " before beta: ", scores)
            prob = np.random.beta(scores[0], scores[1])
            print(agent.__class__.__name__, " after beta: ", prob)
            move = agent.decide()
            print(agent.__class__.__name__, " after decide: ", move)
            if prob > best_prob:
                best_prob = prob
                best_agent = agent
                best_move = move
        
        self.executor = best_agent
        print(self.executor, " best agent: ", self.executor.__class__.__name__)
                
        return best_move
    

'''
OClassifier(XGBClassifier) window = 14
'''
'''
WIN : 0
LOST: 0
'''
player1 = BetaAgency(agents)
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
