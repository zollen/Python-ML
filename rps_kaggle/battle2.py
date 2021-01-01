'''
Created on Dec. 16, 2020

@author: zollen
'''

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import rps_kaggle.lib.rps_lib as rps
import rps_kaggle.lib.battle_lib as bat
import rps_kaggle.lib.enemy_lib as enm
import warnings


warnings.filterwarnings('ignore')


def setup():
    
    forest1 = rps.Classifier(RandomForestClassifier(n_estimators = 10), window = 10)
    forest2 = rps.Sharer(forest1, ahead = 1)
    forest3 = rps.Sharer(forest1, ahead = 2)
    
    xgb1 = rps.Classifier(XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), window = 10)
    xgb2 = rps.Sharer(xgb1, ahead = 1)
    xgb3 = rps.Sharer(xgb1, ahead = 2)
    manager = XGBClassifier(n_estimators = 10, eval_metric = 'logloss')
    
    agents = [ xgb1, xgb2, xgb3 ]
        
    player1 = rps.MetaAgency(manager, agents, window = 12)
   
    player2 = enm.MutliArmAgent()
    
    return player1, player2



player1, player2 = setup()
bat.battleground(player1, player2)