'''
Created on Dec. 16, 2020

@author: zollen
'''

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import rps_kaggle.lib.rps_lib as rps
import rps_kaggle.lib.battle_lib as bat
import warnings


warnings.filterwarnings('ignore')


def setup():
    '''
    
    xgb15 Classifier
    WON [8 ], LOST [12] EVEN [0 ] WINNING RATIO [1.0713]
    
    xgb10 Classifier
    WON [9 ], LOST [11] EVEN [0 ] WINNING RATIO [1.0367] (random_state = 26)
    WON [10], LOST [10] EVEN [0 ] WINNING RATIO [1.0440] (with everything)
    WON [9 ], LOST [11] EVEN [0 ] WINNING RATIO [1.0601] (with no Most and Least)
    
    
    xgb6 Classifier
  
    
    xgb8 Classifier
   
    
    forest10, xgb10 Classifier
    WON [13], LOST [6 ] EVEN [1 ] WINNING RATIO [1.0836]
    WON [10], LOST [10] EVEN [0 ] WINNING RATIO [1.0754]
    
    '''
    
    forest1 = rps.Classifier(RandomForestClassifier(n_estimators = 10), window = 10)
    forest2 = rps.Sharer(forest1, beat = 1)
    forest3 = rps.Sharer(forest1, beat = 2)
    xgb1 = rps.Classifier(XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), window = 10)
    xgb2 = rps.Sharer(xgb1, beat = 1)
    xgb3 = rps.Sharer(xgb1, beat = 2)
    
    agents = [
                [ rps.MirrorOpponentDecider(beat = 0), [1, 1] ],
                [ rps.MirrorOpponentDecider(beat = 1), [1, 1] ],
                [ rps.MirrorOpponentDecider(beat = 2), [1, 1] ],
                [ rps.MirrorSelfDecider(beat = 0),     [1, 1] ],
                [ rps.MirrorSelfDecider(beat = 1),     [1, 1] ],
                [ rps.MirrorSelfDecider(beat = 2),     [1, 1] ],
                [ forest1,                             [1, 1] ],
                [ xgb1,                                [1, 1] ],
                [ forest2,                             [1, 1] ],
                [ xgb2,                                [1, 1] ],
                [ forest3,                             [1, 1] ],
                [ xgb3,                                [1, 1] ]
            ]
        
    player1 = rps.BetaAgency(agents)
    player2 = rps.Classifier(XGBClassifier(random_state = 17, n_estimators = 10, eval_metric = 'logloss'), 
                             window = 10)
    
    return player1, player2




#player2 = rps.MirrorSelfDecider(ahead = 2)
#player2 = rps.NMarkov(3, 6)
#player2 = rps.Randomer()

totalwin = 0
totalloss = 0
totaleven = 0
totalratio = 0.0
for rnd in range(20):
    
    player1, player2 = setup()
   
    win1, win2 = bat.battleground(player1, player2, verbose = False)
    if win1 > win2:
        totalwin += 1
        totalratio += win1 / win2
    elif win1 < win2:
        totalloss += 1
    else:
        totaleven += 1   
     
    print("Match [{:>2}] WON [{}]  LOST [{}] RATIO [{:2.4f}]".format(rnd + 1, win1, win2, win1 / win2))
 
print("=================== TOTAL =======================")    
print("WON [{:<2}], LOST [{:<2}] EVEN [{:<2}] WINNING RATIO [{:2.4f}]".format(totalwin, totalloss, totaleven, 0 if totalwin == 0 else totalratio / totalwin))
