'''
Created on Dec. 16, 2020

@author: zollen
'''
import numpy as np
import pandas as pd
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
        self.data = np.zeros(shape = (1000, self.window * 2))
        self.values = {'00': 0, '01': 1, '02': 2, 
                       '10': 3, '11': 4, '12': 5,
                       '20': 6, '21': 7, '22': 8
            }
   
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
        return "XGBoost()"
    
    def buildrow(self):
        print("UDPATE: ", self.mines[self.row:self.row+self.window].tolist() + self.opponent[self.row:self.row+self.window].tolist())
        self.data[self.row] = self.mines[self.row:self.row+self.window].tolist() + self.opponent[self.row:self.row+self.window].tolist()
        self.results = np.append(self.results, self.opponent[-1])    
        self.row = self.row + 1

    
    def predict(self):

        length = len(self.opponent)
        
        if length < self.window:    
            return self.submit(self.random())
        
        if length >= self.window + 1: 
            self.buildrow() 
        
        print(self.mines, self.opponent, self.results)
        print("DATA: ", self.data[:self.row])
        
        if length > self.window + self.delayProcess + 1:
            self.classifier.fit(self.data[:self.row], self.results)  
            test = np.array(self.mines[-self.window:].tolist() + self.opponent[-self.window:].tolist()).reshape(1, -1)   
            return self.submit(int(self.classifier.predict(test).item()))
        
        
        return self.submit(self.random())
    
moves = []
for _ in range(0, 1000):
    moves.append(np.random.randint(0, 3))

xgb = Classifier(XGBClassifier(random_state = 17, eval_metric = 'logloss'))
for index in range(0, len(moves)):
    print("MY PREDICTED MOVE: ", xgb.predict()) 
    xgb.add(moves[index])
    