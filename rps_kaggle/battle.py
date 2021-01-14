'''
Created on Dec. 16, 2020

@author: zollen
'''

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import rps_kaggle.lib.rps_lib as rps
import rps_kaggle.lib.battle_lib as bat
import warnings


warnings.filterwarnings('ignore')


def setup():

    xgb1 = rps.Classifier(XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), window = 10)
    xgb2 = rps.Sharer(xgb1, ahead = 1)
    xgb3 = rps.Sharer(xgb1, ahead = 2)
    managers = [
                [ XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), [0, 0], [0]],
                [ RandomForestClassifier(n_estimators = 10),                 [0, 0], [0]],
                [ AdaBoostClassifier(n_estimators = 10),                     [0, 0], [0]],
                [ KNeighborsClassifier(),                                    [0, 0], [0]],
                [ SVC(kernel = 'rbf'),                                       [0, 0], [0]]
                ]
    
    agents = [ xgb1, xgb2, xgb3 ]
        
    player1 = rps.MetaAgency(managers, agents, window = 25, history = 50, random_threshold = -40, randomness = 0.1)
   
    import rps_kaggle.lib.enemy4_lib as enm4
    player2 = enm4.Iocaine2(num_predictor = 120)
    
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
