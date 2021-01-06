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
import warnings


warnings.filterwarnings('ignore')

'''
XGB and FOREST
PLAYER1 336, PLAYER2 335  RATIO 1.0030



'''

def setup():
   
    xgb1 = rps.Classifier(XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), history = 300, window = 15)
    forest = rps.Classifier(RandomForestClassifier(n_estimators = 10), history = 300, window = 15)
    ada = rps.Classifier(AdaBoostClassifier(n_estimators = 10), history = 300, window = 15)
    knn = rps.Classifier(KNeighborsClassifier(), history = 300, window = 15)
    svm = rps.Classifier(SVC(kernel = 'rbf'), history = 300, window = 15)

    agents = [
                [ xgb1,                [0, 0], [0]],
                [ forest,              [0, 0], [0]],
                [ ada,                 [0, 0], [0]],
                [ knn,                 [0, 0], [0]],
                [ svm,                 [0, 0], [0]]
            ]
            

    player1 = rps.VoteAgency(agents, randomness = 0.1)
    player2 = enm.MutliArmAgent()
    
    return player1, player2



player1, player2 = setup()
bat.battleground(player1, player2)

