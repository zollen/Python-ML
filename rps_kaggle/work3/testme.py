'''
Created on Dec. 16, 2020

@author: zollen
'''
import numpy as np
import time
from lightgbm import LGBMClassifier 
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

class BaseAgent:
    def __init__(self, states = 3, window = 3):
        np.random.seed(int(round(time.time())))
        self.mines = np.array([])
        self.opponent = np.array([])
        self.results = np.array([])
        self.states = states
        self.window = window
        
    def add(self, token):
        self.opponent = np.append(self.opponent, token)
        
    def submit(self, token):
        self.mines = np.append(self.mines, token)
        return token
    
    def random(self):
        return np.random.randint(0, self.states)
        
    def __str__(self):
        return "BaseAgent(" + str(self.window) + ")"
    
    def predict(self):
        pass


class Classifier(BaseAgent):
    
    def __init__(self, classifier, states = 3, window = 3, delay_process = 5):
        super().__init__(states, window)
        self.classifier = classifier
        self.delayProcess = delay_process
        self.row = 0
        self.data = np.zeros(shape = (1100, self.window * 2))
      
   
    def add(self, token):
        BaseAgent.add(self, token)
        
        if len(self.opponent) >= self.window + 1: 
            self.buildrow() 
    
    def __str__(self):
        return self.classifier.__class__.__name__
    
    def buildrow(self):
        self.data[self.row] = self.mines[self.row:self.row+self.window].tolist() + self.opponent[self.row:self.row+self.window].tolist()
        self.results = np.append(self.results, self.opponent[-1])    
        self.row = self.row + 1

    
    def predict(self):
     
        if len(self.opponent) > self.window + self.delayProcess + 1:
            self.classifier.fit(self.data[:self.row], self.results)  
            test = np.array(self.mines[-self.window:].tolist() + self.opponent[-self.window:].tolist()).reshape(1, -1)   
            return self.submit((int(self.classifier.predict(test).item()) + 1) % self.states)
            
        return self.submit(self.random())
        


#clr = Classifier(XGBClassifier(random_state = 17, n_estimators = 10, eval_metric = 'logloss'), window = 6)
clr = Classifier(LGBMClassifier(random_state = 17, n_estimators = 10), window = 6)


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


    
