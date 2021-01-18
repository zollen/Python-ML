'''
Created on Jan. 18, 2021

@author: zollen
'''
import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import rps_kaggle.lib.rps_lib as rps
import rps_kaggle.lib.battle_lib as bat
import rps_kaggle.lib.enemy_lib as enm
import rps_kaggle.lib.enemy2_lib as enm2
import rps_kaggle.lib.enemy3_lib as enm3
import rps_kaggle.lib.enemy4_lib as enm4
import rps_kaggle.lib.enemy5_lib as enm5


    
agents = [
            [ enm2.MarkovChain(4, 0.9),                                               [], [] ],
            [ enm3.MemoryPatterns(min_memory=50, max_memory=120, warmup=20),          [], [] ],
            [ enm3.MemoryPatterns(min_memory=60, max_memory=120, warmup=20),          [], [] ],
            [ enm4.Iocaine(num_predictor = 100),                                      [], [] ]
    ]

calculators = [
            rps.PopularityManager(agents),
            rps.OutComeManager(agents),
            rps.Last5RoundsManager(agents),
            rps.BetaManager(agents)
        ]

agency = rps.StatsAgency(calculators, agents, randomness = 0.1, random_threshold = -10)



for _ in range(5):
    print("MOVE: ", agency.decide())
    agency.add(np.random.randint(3))