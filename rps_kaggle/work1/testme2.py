'''
Created on Jan. 18, 2021

@author: zollen

12,8
'''

import numpy as np
import rps_kaggle.lib.enemy2_lib as enm2




markov = enm2.MarkovNet(min_len = 3, max_len = 4)
 
for _ in range(500):
    
    print("MOVE: ", markov.decide())
    markov.add(np.random.randint(3))