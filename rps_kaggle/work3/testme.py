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
from sklearn.ensemble._weight_boosting import AdaBoostRegressor


warnings.filterwarnings('ignore')


xgb1 = rps.Classifier(XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), history = 200, window = 15)
forest = rps.Classifier(RandomForestClassifier(n_estimators = 10), history = 200, window = 15)
ada = rps.Classifier(AdaBoostClassifier(n_estimators = 10), history = 200, window = 15)
knn = rps.Classifier(KNeighborsClassifier(), history = 200, window = 15)
svm = rps.Classifier(SVC(kernel = 'rbf'), history = 200, window = 15)

agents = [
                [ xgb1,                [0, 0], [0]],
                [ forest,              [0, 0], [0]],
                [ ada,                 [0, 0], [0]],
                [ knn,                 [0, 0], [0]],
                [ svm,                 [0, 0], [0]]
            ]
            

agency = rps.VoteAgency(agents, randomness = 0.1)



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

