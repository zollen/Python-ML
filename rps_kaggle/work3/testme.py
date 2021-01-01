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


xgb1 = rps.Classifier(XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), window = 15)
xgb2 = rps.Sharer(xgb1, ahead = 1)
xgb3 = rps.Sharer(xgb1, ahead = 2)
manager = XGBClassifier(n_estimators = 10, eval_metric = 'logloss')

agents = [ xgb1, xgb2, xgb3 ]
    
agency = rps.MetaAgency(manager, agents, window = 20, history = 50)



def classifier_move(observation, configuration):

    global agency
    
    if observation.step > 0:
        agency.add(observation.lastOpponentAction)
        
    return agency.decide()


class observationCls:
    step = 0
    lastOpponentAction = 0
class configurationCls:
    signs = 3
    
observation = observationCls()
configuration = configurationCls()

for rnd in range(0, 1000):
    
    choice = None
    observation.step = rnd
    observation.lastOpponentAction = np.random.randint(3)

    t_start = time.perf_counter_ns()
    choice = classifier_move(observation, configuration)
    t_end = time.perf_counter_ns()
    
    print("Round {} Choice: {}, Elapse Time: {}".format(rnd + 1, choice, t_end - t_start))

