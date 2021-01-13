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
     
    player1 = rps.MetaAgency(managers, agents, window = 20, history = 50, random_threshold = -10, randomness = 0.1)

#    player2 = enm.MutliArmAgent()
#    player2 = enm2.MarkovChain(2, 0.9)
#    player2 = enm3.Iocaine()
    player2 = enm4.Iocaine2()

   
    
    return player1, player2



player1, player2 = setup()
bat.battleground(player1, player2)

