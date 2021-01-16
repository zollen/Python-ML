'''
Created on Dec. 16, 2020

@author: zollen
'''

import numpy as np 
import time
import rps_kaggle.lib.enemy3_lib as enm3

            
mp = enm3.MemoryPatterns()
def kaggle_agent(observation, configuration):
    if observation.step > 0:
        mp.add(observation.lastOpponentAction)
        
    return mp()

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
    observation.lastOpponentAction = np.random.randint(3)

    t_start = time.perf_counter_ns()
    choice = kaggle_agent(observation, configuration)
    t_end = time.perf_counter_ns()
    
    print("Round {} Choice: {}, Elapse Time: {}".format(rnd + 1, choice, t_end - t_start))

