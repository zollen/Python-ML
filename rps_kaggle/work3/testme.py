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


class CounterMover:
    
    def __init__(self, agent, states = 3, interval = 5):
        self.states = states
        self.interval = interval
        self.agent = agent
        self.enable = 0
        
    def won(self, index):
        res = self.agent.mines[index] - self.agent.opponent[index]  
        if res == 0:
            return 0
        elif res == 1:
            return 1
        elif res == -1:
            return -1
        elif res == 2:
            return -1
        else:             
            return 1
        
    def predict(self, token):
        
        print("Enable: ", self.enable)
        if self.enable <= 0 and self.won(-1) < 0 and self.won(-2) < 0:
            self.enable = np.random.randint(1, self.interval + 1)
            print("NEW ENABLE: ", self.enable)
                
        if self.enable > 0:
            print("Counter Move: ", (token + 2) % self.states)
            self.enable = self.enable - 1
            return (token + 2) % self.states
        print("Original Choice: ", token )
        return token





clr = rps.Classifier(XGBClassifier(random_state = 17, n_estimators = 10, eval_metric = 'logloss'), window = 3)

class Dummy:
    
    def __init__(self):
        self.mines = [0, 1]
        self.opponent = [1, 2]
    
    def predict(self):
        return np.random.randint(0, 3)

agent = Dummy()
mm = CounterMover(agent = agent)    
for _ in range(0, 5):
    print("MOVE: ", mm.predict(1))
    agent.mines.append(1)
    agent.opponent.append(0)


exit()


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
