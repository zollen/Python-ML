'''
Created on Jan. 13, 2021

@author: zollen
'''
import random
import math
import numpy as np
from collections import defaultdict 
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
    
    
class MarkovNet(rps.BaseAgent):
    
    FTRANSLATE = { 
              "01": 0, "12": 1, "20": 2,
              "00": 3, "11": 4, "22": 5,
              "10": 6, "21": 7, "02": 8
            }

    RTRANSLATE = {
                0: '01', 1: '12', 2: '20',
                3: '00', 4: '11', 5: '22',
                6: '10', 7: '21', 8: '02'
            }
    
    def __init__(self, states = 3, opp_predicted = True, min_len = 3, max_len = 7):
        super().__init__(states, 0, 0, None)
        self.almoves = []
        self.minLength = min_len
        self.maxLength = max_len
        self.tokens = defaultdict(lambda: [0] * self.states)
        self.currLength = 0
        self.last = None
        
        if opp_predicted == True:
            self.identity = 1
        else:
            self.identity = 0
        
    def __str__(self):
        return "MarkovNet(" + str(self.minLength) + ", " + str(self.maxLength) + ")"
    
    def normalize(self, scores):
        total = np.sum(scores)
        if total <= 0:
            return [ 0 ] * len(scores)
        return [ x / total for x in scores ]
    
    def decide(self):
        
        if self.currLength < self.mines.size and self.currLength < self.opponent.size:
          
            self.almoves.append(self.FTRANSLATE[str(self.mines[-1]) + str(self.opponent[-1])])
            self.currLength += 1
            
            if self.currLength > self.minLength:
                
                for key, val in self.tokens.items():
                    self.tokens[key] = [ x * 0.8 for x in val ]
                
                for window in range(self.minLength, self.maxLength + 1):
                    predicted = int(self.RTRANSLATE[self.almoves[-1]][self.identity])
                    for action in range(self.states):
                        if action == predicted:
                            self.tokens[tuple(self.almoves[-window-1:-1])][action] += 1
                        elif action == (predicted + 2) % self.states:
                            self.tokens[tuple(self.almoves[-window-1:-1])][action] += 0.4
                        else:
                            self.tokens[tuple(self.almoves[-window-1:-1])][action] *= 0.5
                      
            
        final_scores = [0] * self.states
        for window, p in zip(range(self.maxLength, self.minLength - 1, -1), range(100)):
            final_scores = [ x + (y * math.pow(0.9, p)) for x, y in zip(final_scores, self.normalize(self.tokens[tuple(self.almoves[-window:])])) ]
        
        if all(x == final_scores[0] for x in final_scores):
            return self.submit(np.random.randint(self.states))
        
        return self.submit((np.argmax(final_scores).item() + 1) % self.states) 
    
    

