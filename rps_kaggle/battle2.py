'''
Created on Dec. 16, 2020

@author: zollen
'''

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import rps_kaggle.lib.rps_lib as rps
import rps_kaggle.lib.battle_lib as bat
import rps_kaggle.lib.enemy_lib as enm
import rps_kaggle.lib.enemy2_lib as enm2
import rps_kaggle.lib.enemy3_lib as enm3
import rps_kaggle.lib.enemy4_lib as enm4
import warnings


warnings.filterwarnings('ignore')

'''
XGB and FOREST
PLAYER1 336, PLAYER2 335  RATIO 1.0030



'''

def setup():
    
    if True:
        
        markovChain = enm2.MarkovChain(4, 0.9)
        iocaine2 = enm4.Iocaine2(num_predictor = 100)
        xgb15 = rps.Classifier(XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), window = 15)
        
        agents = [
                    [ markovChain,            [0, 0], [0]],
                    [ iocaine2,               [0, 0], [0]],
                    [ xgb15,                  [0, 0], [0]]
            ]
        
        player1 = rps.BetaAgency(agents, decay = 1.1)
    
    
    if False:
        xgb1 = rps.Classifier(XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), window = 15)
        xgb2 = rps.Sharer(xgb1, ahead = 1)
        xgb3 = rps.Sharer(xgb1, ahead = 2)
        managers = [
                         [ XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), [0, 0], [0] ],
                         [ RandomForestClassifier(n_estimators = 10),                 [0, 0], [0] ],
                         [ KNeighborsClassifier(),                                    [0, 0], [0] ],
                         [ SVC(kernel = 'rbf'),                                       [0, 0], [0] ]
                     ]
        
        agents = [ xgb1, xgb2, xgb3 ]
             
        player2 = rps.MetaAgency(managers, agents, window = 20, history = 50, random_threshold = -10, randomness = 0.1)


    if True:
        player2 = enm.MultiArmsBandit()
    if False:
        player2 = enm2.MarkovChain(4, 0.9)
    if False:
        player2 = enm3.Iocaine()
    if False:
        player2 = enm4.Iocaine2(num_predictor = 100)


   
    
    return player1, player2




    
player1, player2 = setup()
bat.battleground(player1, player2)
   