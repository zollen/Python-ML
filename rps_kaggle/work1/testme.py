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
import rps_kaggle.lib.enemy2_lib as enm2
import rps_kaggle.lib.enemy3_lib as enm3
import rps_kaggle.lib.enemy4_lib as enm4
import warnings


warnings.filterwarnings('ignore')


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
     
agency = rps.MetaAgency(managers, agents, window = 20, history = 50, random_threshold = -10, randomness = 0.1)

agency = enm2.MarkovChain(3, 0.9)

agency = enm4.Iocaine(num_predictor = 160)

agency = enm3.MemoryPatterns(warmup=20, min_memory=60, max_memory=120, verbose = False)


def agency_move(observation, configuration):

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

for rnd in range(0, 100):
    
    choice = None
    observation.step = rnd
    observation.lastOpponentAction = np.random.randint(3)

    t_start = time.perf_counter_ns()
    choice = agency_move(observation, configuration)
    t_end = time.perf_counter_ns()
    
    print("Round {} Choice: {}, Elapse Time: {}".format(rnd + 1, choice, t_end - t_start))

