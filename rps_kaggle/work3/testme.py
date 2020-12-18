'''
Created on Dec. 16, 2020

@author: zollen
'''
import numpy as np
import time
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

class Classifier:
    
    def __init__(self, classifier, window = 3, delay_process = 5):
        np.random.seed(int(round(time.time())))
        self.classifier = classifier
        self.window = window
        self.opponent = np.array([])
        self.mines = np.array([])
        self.results = np.array([])
        self.delayProcess = delay_process
        self.row = 0
        self.data = np.zeros(shape = (1100, self.window * 2))
      
   
    def add(self, token):
        self.opponent = np.append(self.opponent, token)
    
    def submit(self, token):
        self.mines = np.append(self.mines, token)
        return token
        
    def value(self, src, dest):
        return self.values[str(src) + str(dest)]
    
    def random(self):
        return np.random.randint(0, 3)
    
    def __str__(self):
        return self.classifier.__class__.__name__
    
    def buildrow(self):
        self.data[self.row] = self.mines[self.row:self.row+self.window].tolist() + self.opponent[self.row:self.row+self.window].tolist()
        self.results = np.append(self.results, self.opponent[-1])    
        self.row = self.row + 1

    
    def predict(self):

        length = len(self.opponent)
        
        if length < self.window:    
            return self.submit(self.random())
        
        if length >= self.window + 1: 
            self.buildrow() 
     
        if length > self.window + self.delayProcess + 1:
            self.classifier.fit(self.data[:self.row], self.results)  
            test = np.array(self.mines[-self.window:].tolist() + self.opponent[-self.window:].tolist()).reshape(1, -1)   
            return self.submit(int(self.classifier.predict(test).item()))
            
        return self.submit(self.random())
        
    
    

clr = Classifier(XGBClassifier(random_state = 17, n_estimators = 10, eval_metric = 'logloss'), window = 6)

def classifier_move(observation, configuration):

    global clr
    
    if observation.step > 0:
        clr.add(observation.lastOpponentAction)
        
    return (clr.predict() + 1) % configuration.signs


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


    
