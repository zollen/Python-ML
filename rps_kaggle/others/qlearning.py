'''
Created on Feb. 3, 2021

@author: zollen
@review: check rps_kaggle/lib/enemy6_lib.py (using q-learning)
'''

import numpy as np
import time
import random


class QLeaner:
    
    def __init__(self):
        self.Q = np.zeros((9, 3))
        self.alpha = 0.7004648211071717 
        self.alpha_decay = 1
        self.discount = 0.31635680630654883
        self.epsilon = 0.8254597834978199
        self.epsilon_decay = 0.9999999912474902

        self.STATES = {(0, 0): 0,
                      (0, 1): 1,
                      (0, 2): 2,
                      (1, 0): 3,
                      (1, 1): 4,
                      (1, 2): 5,
                      (2, 0): 6,
                      (2, 1): 7,
                      (2, 2): 8}
        
        self.current_state = 0
        self.current_action = 0
        self.current_opponent = 0

    def determine_winner(self, you, opponent):
        winning_situations = [[0,2],[2,1],[1,0]]
        if [you,opponent] in winning_situations:
            return 1
        elif you == opponent:
            return 0
        else:
            return -1
    
    def __str__(self):
        return "QLearner"
    
    def add(self, token):
        self.current_opponent = token
        self.current_state = self.STATES[(self.current_action, token)]
        
    
    def decide(self):
        
        if observation.step == 0:
            self.current_action = int(np.random.randint(0,3))
            return self.current_action
        elif observation.step == 1:
           
            if self.epsilon > random.uniform(0,1):
                self.current_action = int(np.random.randint(0,3))
                return self.current_action
            else:
                self.current_action = int(self.Q[self.current_state,:].argmax())
                return self.current_action
            
            return self.current_action 
        else:
            reward = self.determine_winner(self.current_action, self.current_opponent)
            next_state = self.STATES[(self.current_action, self.current_opponent)]
            
            discounted_next_state = self.alpha*(reward+
                                           self.discount*self.Q[next_state,self.Q[next_state,:].argmax()] - 
                                           self.Q[self.current_state, self.current_action])
            
            self.Q[self.current_state, self.current_action] = self.Q[self.current_state, self.current_action] + discounted_next_state
            self.current_state = self.STATES[(self.current_action, self.current_action)]
            
            
            
            if self.epsilon > random.uniform(0,1):
                self.current_action = int(np.random.randint(0,3))
            else:
                self.current_action = int(self.Q[self.current_state,:].argmax())
             
            self.alpha *= self.alpha_decay
            self.epsilon *= self.epsilon_decay
            
            return self.current_action
        


class observationCls:
    step = 0
    lastOpponentAction = 0
class configurationCls:
    signs = 3
    
observation = observationCls()
configuration = configurationCls()

agent = QLeaner()

for rnd in range(0, 100):
    
    choice = None
    observation.step = rnd
    observation.lastOpponentAction = np.random.randint(3)

    t_start = time.perf_counter_ns()
    choice = agent.decide()
    t_end = time.perf_counter_ns()
    
    agent.add(np.random.randint(3))
    
    print("Round {} Choice: {}, Elapse Time: {}".format(rnd + 1, choice, t_end - t_start))