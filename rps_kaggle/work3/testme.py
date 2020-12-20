'''
Created on Dec. 16, 2020

@author: zollen
'''
import numpy as np
import time
from xgboost import XGBClassifier
import rps_kaggle.lib.rps_lib as rps
import warnings

warnings.filterwarnings('ignore')



clr = rps.Classifier(XGBClassifier(random_state = 17, n_estimators = 10, eval_metric = 'logloss'), window = 3)




def classifier_move(observation, configuration):

    global clr
    
    if observation.step > 0:
        clr.add(observation.lastOpponentAction)
        
    return clr.predict()


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
    observation.lastOpponentAction = np.random.randint(0, 3)
    
    t_start = time.perf_counter_ns()
    choice = classifier_move(observation, configuration)
    t_end = time.perf_counter_ns()
    
    print("Round {} Choice: {}, Elapse Time: {}".format(rnd + 1, choice, t_end - t_start))


'''
TO DO LIST
build matrix level information (failed, performed poorer)
retry DecisionMaker with +2 step ahead
'''  
