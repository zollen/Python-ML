'''
Created on Dec. 16, 2020

@author: zollen
'''

import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import rps_kaggle.lib.rps_lib as rps
import warnings
from rps_kaggle.lib.rps_lib import StandardCounterMover
from sklearn.linear_model.tests.test_passive_aggressive import MyPassiveAggressive



warnings.filterwarnings('ignore')

SIGNS = ['ROCK', 'PAPER', 'SCISSORS']


class Balancer:
    
    def __init__(self, agents, duration):
        self.agents = agents
        self.duration = duration
        self.interval = 0
        self.target = None
        
    def __str__(self):
        return self.target.__class__.__name__
        
    def add(self, token): 
        for agent in agents:
            agent.add(token)
    
    def submit(self, token):
        for agent in agents:
            agent.submit(token)             
        return token
    
    def reset(self):        
        for agent in agents:
            agent.reset()           
        self.target = None
        self.interval = 0
        
    def decide(self):
        if self.interval <= 0:
            self.interval = self.duration
            
            targets = self.agents.copy()
            if self.target != None:
                targets.remove(self.target)
                
            self.target = self.agents[np.random.randint(0, len(targets))]
            
        self.interval = self.interval - 1
        
        return self.submit(self.target.decide())


        
class Distractor:   
    
    def __init__(self, primary, best, distractors, delta = 5):
        self.primary = primary
        self.best = best
        self.distractors = distractors
        self.delta = delta
        self.mines = []
        self.opponent = []
        self.totalMatches = 0
        self.totalWin = 0
        self.totalLoss = 0
        self.executor = None
        
    def __str__(self):
        return self.executor.__str__()
        
    def add(self, token): 
        self.distractors.add(token)
        self.primary.add(token)
        self.best.add(token)
        
        self.opponent.append(token)
            
    def submit(self, token):
        if self.executor != self.distractors:
            self.distractors.submit(token)
        if self.executor != self.primary:
            self.primary.submit(token)
        if self.executor != self.best:
            self.best.submit(token)
            
        self.mines.append(token)
            
        return token
    
    def reset(self):
        self.distractors.reset()
        self.primary.reset()
        self.best.reset()
        
    def won(self, index):
        
        if self.totalMatches == 0:
            return 0
        
        res = self.mines[index] - self.opponent[index]
        if res == 1 or res == -2:
            return 1
        elif res == -1 or res == 2:
            return -1
        return 0
        
    def decide(self):
        
        if self.totalMatches > 0 and self.won(-1) > 0:
            self.totalWin = self.totalWin + 1
        elif self.totalMatches > 0 and self.won(-1) < 0:
            self.totalLoss = self.totalLoss + 1
            
        self.totalMatches = self.totalMatches + 1
            
        ratio = (self.totalWin - self.totalLoss)
        
        if ratio > self.delta:
            self.executor = self.distractors
        elif ratio <= self.delta and ratio > -self.delta:
            self.executor = self.primary
        else:
            self.executor = self.best
        
        return self.submit(self.executor.decide())
                
                
        
 
       
    
    

agents = [
    rps.Randomer(),
    rps.MirrorOpponentDecider(ahead = 0),
    rps.MirrorSelfDecider(ahead = 0),
    rps.MirrorOpponentDecider(ahead = 1),
    rps.MirrorSelfDecider(ahead = 1),
    rps.MirrorOpponentDecider(ahead = 2),
    rps.MirrorSelfDecider(ahead = 2),
    ]

balancer = Balancer(agents, 3)
primray = rps.Classifier(KNeighborsClassifier(), window = 10)
best = rps.OClassifier(
                rps.ClassifierHolder([
                    AdaBoostClassifier(random_state = 17, n_estimators = 10),
                    KNeighborsClassifier(),
                    SVC(kernel = 'rbf')
                ]), window = 10)

distractor = Distractor(
    primary = primray,
    best = best,
    distractors = balancer,
    delta = 4
    )






'''
OClassifier(XGBClassifier) window = 14
'''
'''
WIN : 0
LOST: 0
'''
player1 = distractor
player2 = rps.Classifier(XGBClassifier(random_state = 17, n_estimators = 10, eval_metric = 'logloss'), 
                         window = 10)


#player2 = rps.MirrorSelfDecider(ahead = 2)
#player2 = rps.NMarkov(3, 6)
#player2 = rps.Randomer()

results = []
win1 = 0
win2 = 0

for rnd in range(0, 1000):

    move1 = player1.decide()
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


    msg = "[{:<4}]   {:>8} | {:<8} => {:^50}  {:>4}|{:<4}"
    print(msg.format(rnd + 1, SIGNS[move1], SIGNS[move2], winner.__str__(), win1, win2))


print("====================================================================")
if win1 == win2:
    print("BOTH TIE!!")
elif win1 > win2:
    print("PLAYER1 {}, PLAYER2 {}  RATIO {:2.4f}".format(win1, win2, win1 / win2))
    print("Player1: [{}] WON!!!!!".format(player1))
else:
    print("PLAYER1 {}, PLAYER2 {} RATIO {:2.4f}".format(win1, win2, win1 / win2))
    print("Player1: [{}] LOST!!!!!".format(player1))
