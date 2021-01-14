'''
Created on Jan. 13, 2021

@author: zollen
'''

import random
import itertools
from operator import itemgetter


class MarkovChain:
    
    def __init__(self, order, decay=1.0):
        self.decay = decay
        self.order = order
        self.mines = []
        self.opponents = []
        self.matrix = self.create_matrix(order)
        self.SYMBOLS = ['R', 'P', 'S']
        self.RSYMBOLS = { 'R': 0, 'P': 1, 'S': 2 }
        self.beat = {0: 1, 1: 2, 2: 0}
        self.best_move = ''
        self.pair_diff2 = ''
        self.pair_diff1 = ''

    @staticmethod
    def create_matrix(order):

        def create_keys(order):            

            keys = ['R', 'P', 'S']

            for i in range((order * 2 - 1)):
                key_len = len(keys)
                for i in itertools.product(keys, ''.join(keys)):
                    keys.append(''.join(i))
                keys = keys[key_len:]

            return keys

        keys = create_keys(order)

        matrix = {}
        for key in keys:
            matrix[key] = {'R': {'prob' : 1 / 3,
                                 'n_obs' : 0
                                },
                           'P': {'prob' : 1 / 3,
                                 'n_obs' : 0
                                },
                           'S': {'prob' : 1 / 3,
                                 'n_obs' : 0
                                }
                          }

        return matrix
    
    def __str__(self):
        return "MarkovChain(" + str(self.order) + ")"
    
    def submit(self, token):
        self.mines.append(token)
        return token
        
    def encode(self):
        
        mymoves, opmoves = self.mines[-1 * self.order:], self.opponents[-1 * self.order:]
        
        bothmoves = ''
        for mymove, opmove in zip(mymoves, opmoves):
            bothmoves += self.SYMBOLS[mymove] + self.SYMBOLS[opmove]
          
        return bothmoves
    
    def add(self, token):
        
        self.opponents.append(token)
        
        self.pair_diff2 = self.pair_diff1
        self.pair_diff1 = self.encode()

        if len(self.pair_diff2) == self.order * 2:
            self.update_matrix(self.pair_diff2, self.SYMBOLS[token])

    def update_matrix(self, pair, input):
                
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

    def predict(self, pair):
        
        if pair in self.matrix:
            probs = self.matrix[pair]
                      
        if self.best_move == '' or len(self.pair_diff1) < self.order * 2 or len(self.pair_diff2) < self.order * 2 or max(probs.values(), key=itemgetter('prob')) == min(probs.values(), key=itemgetter('prob')):
            self.best_move = random.choice([ 0, 1, 2 ])
        else: 
            best_prob = -1
            best_move = 0
            for move, item in probs.items():
                if item['prob'] > best_prob:
                    best_prob = item['prob']
                    best_move = move
            self.best_move = self.beat[self.RSYMBOLS[best_move]]
            
        return self.submit(self.best_move)
    
    
    

class MarkovChain2:
    
    def __init__(self, order, decay=1.0):
        self.decay = decay
        self.order = order
        self.mines = []
        self.opponents = []
        self.matrix = {}
        self.SYMBOLS = ['R', 'P', 'S']
        self.RSYMBOLS = { 'R': 0, 'P': 1, 'S': 2 }
        self.beat = {0: 1, 1: 2, 2: 0}
        self.best_move = ''
        self.pair_diff2 = ''
        self.pair_diff1 = ''
        
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
    
    def __str__(self):
        return "MarkovChain2(" + str(self.order) + ")"
    
    def encode(self):
        
        mymoves, opmoves = self.mines[-1 * self.order:], self.opponents[-1 * self.order:]
        
        bothmoves = ''
        for mymove, opmove in zip(mymoves, opmoves):
            bothmoves += self.SYMBOLS[mymove] + self.SYMBOLS[opmove]
          
        return bothmoves
    
    def submit(self, token):
        self.mines.append(token)
        return token
    
    def add(self, token):
        
        self.opponents.append(token)
        
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
            
        return self.submit(self.best_move)
    
    
