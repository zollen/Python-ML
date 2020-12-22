'''
Created on Dec. 16, 2020

@author: zollen
'''

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import rps_kaggle.lib.rps_lib as rps
import warnings

warnings.filterwarnings('ignore')

SIGNS = ['ROCK', 'PAPER', 'SCISSORS']
'''
WON : 2
LOST: 1
'''   
player1 = rps.SClassifier(XGBClassifier(random_state = 17, n_estimators = 10, eval_metric = 'logloss'), 
                         window = 6)
player2 = rps.Classifier(XGBClassifier(random_state = 17, n_estimators = 10, eval_metric = 'logloss'), 
                         window = 6)


#player2 = rps.NMarkov(3, 6)
#player2 = rps.Randomer()

results = []
win1 = 0
win2 = 0

for rnd in range(0, 1000):

    move1 = player1.predict()
    move2 = player2.predict()
    
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


    msg = "[{:<4}]   {:>8} | {:<8} => {:^20}"
    print(msg.format(rnd + 1, SIGNS[move1], SIGNS[move2], winner.__str__()))


print("====================================================================")
if win1 == win2:
    print("BOTH TIE!!")
elif win1 > win2:
    print("PLAYER1 {}, PLAYER2 {}  RATIO {:2.4f}".format(win1, win2, win1 / win2))
    print("Player1: [{}] WON!!!!!".format(player1))
else:
    print("PLAYER1 {}, PLAYER2 {} RATIO {:2.4f}".format(win1, win2, win1 / win2))
    print("Player1: [{}] LOST!!!!!".format(player1))
