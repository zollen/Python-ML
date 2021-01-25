'''
Created on Jan. 13, 2021

@author: zollen
'''

import random
from operator import itemgetter
import rps_kaggle.lib.rps_lib as rps


class MarkovChain(rps.BaseAgent):
    
    def __init__(self, states = 3, window = 3, ahead = 1, counter = None, order = 3, decay=1.0):
        super().__init__(states, window, ahead, counter)
        self.decay = decay
        self.order = order
        self.matrix = {}
        self.SYMBOLS = ['R', 'P', 'S']
        self.RSYMBOLS = { 'R': 0, 'P': 1, 'S': 2 }
        self.beat = {0: 1, 1: 2, 2: 0}
        self.best_move = ''
        self.pair_diff2 = ''
        self.pair_diff1 = ''
        self.last = None
        
    @staticmethod
    def init_matrix():
        return {
                'R': {  'prob' : 1 / 3,
                        'n_obs' : 0
                    },
                'P': {  'prob' : 1 / 3,
                        'n_obs' : 0
                    },
                'S': {  'prob' : 1 / 3,
                        'n_obs' : 0
                    }
                }
        
    def reset(self):
        self.last = None
        super().reset()
    
    def __str__(self):
        return "MarkovChain(" + str(self.order) + ")"
    
    def encode(self):
        
        mymoves, opmoves = self.mines[-1 * self.order:], self.opponent[-1 * self.order:]
        
        bothmoves = ''
        for mymove, opmove in zip(mymoves, opmoves):
            bothmoves += self.SYMBOLS[mymove] + self.SYMBOLS[opmove]
          
        return bothmoves
    
    def deposit(self, token):
        self.last = token
        self.best_move = token
        super().deposit(token)
    
    def add(self, token):
        
        super().add(token)
        
        self.pair_diff2 = self.pair_diff1
        self.pair_diff1 = self.encode()

        if len(self.pair_diff2) == self.order * 2:
            self.update_matrix(self.pair_diff2, self.SYMBOLS[token])

    def update_matrix(self, pair, input):
          
        if pair not in self.matrix:
            self.matrix[pair] = self.init_matrix()
        
        for i in self.matrix[pair]:
            self.matrix[pair][i]['n_obs'] = self.decay * self.matrix[pair][i]['n_obs']

        self.matrix[pair][input]['n_obs'] = self.matrix[pair][input]['n_obs'] + 1
        
        n_total = 0
        for i in self.matrix[pair]:
            n_total += self.matrix[pair][i]['n_obs']
            
        for i in self.matrix[pair]:
            self.matrix[pair][i]['prob'] = self.matrix[pair][i]['n_obs'] / n_total  
            
    def decide(self):
        return self.predict(self.pair_diff1)       
    
    def estimate(self): 
        self.record = False
        return self.predict(self.pair_diff1)  

    def predict(self, pair):
        
        probs = self.init_matrix()

        if len(pair) == self.order * 2 and pair in self.matrix:
            probs = self.matrix[pair]
            
        if self.best_move == '' or max(probs.values(), key=itemgetter('prob')) == min(probs.values(), key=itemgetter('prob')):
            self.best_move = random.choice([ 0, 1, 2 ])
        else: 
            best_prob = -1
            best_move = 0
            for move, item in probs.items():
                if item['prob'] > best_prob:
                    best_prob = item['prob']
                    best_move = move
            self.best_move = self.beat[self.RSYMBOLS[best_move]]
            
        self.last = self.best_move
            
        return self.submit(self.best_move)
    
    
