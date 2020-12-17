'''
Created on Dec. 16, 2020

@author: zollen
'''
import numpy as np
import pandas as pd
import time


class XGBoost:
    
    def __init__(self, window):
        np.random.seed(int(round(time.time())))
        self.window = window
        self.opponent = []
        self.mines = []
        self.data = pd.DataFrame()
   
    def add(self, token):
        self.opponent = np.append(self.opponent, token)
    
    def submit(self, token):
        self.opponent = np.append(self.mines, token)
        
    def __str__(self):
        return "XGBoost()"
    
    def predict(self):
        
        if len(self.mines) <= self.window:
            return self.submit(np.random.randint(0, 3))
        
        
    


