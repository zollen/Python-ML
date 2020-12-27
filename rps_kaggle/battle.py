'''
Created on Dec. 16, 2020

@author: zollen
'''

import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import rps_kaggle.lib.rps_lib as rps
import rps_kaggle.lib.battle_lib as bat
import warnings


warnings.filterwarnings('ignore')

SIGNS = ['ROCK', 'PAPER', 'SCISSORS']


forest1 = rps.Classifier(RandomForestClassifier(random_state = 23, n_estimators = 10), window = 10)
forest2 = rps.ShareClassifier(forest1, beat = 1)
forest3 = rps.ShareClassifier(forest1, beat = 2)
xgb1 = rps.Classifier(XGBClassifier(random_state = 26, n_estimators = 10, eval_metric = 'logloss'), window = 10)
xgb2 = rps.ShareClassifier(xgb1, beat = 1)
xgb3 = rps.ShareClassifier(xgb1, beat = 2)

agents = [
            [ rps.Randomer(), [1, 1]                      ],
            [ rps.MirrorOpponentDecider(beat = 0), [1, 1] ],
            [ rps.MirrorOpponentDecider(beat = 1), [1, 1] ],
            [ rps.MirrorOpponentDecider(beat = 2), [1, 1] ],
            [ rps.MirrorSelfDecider(beat = 0), [1, 1]     ],
            [ rps.MirrorSelfDecider(beat = 1), [1, 1]     ],
            [ rps.MirrorSelfDecider(beat = 2), [1, 1]     ],
            [ forest1, [1, 1]                             ],
            [ xgb1, [1, 1]                                ],
            [ forest2, [1, 1]                             ],
            [ xgb2, [1, 1]                                ],
            [ forest3, [1, 1]                             ],
            [ xgb3, [1, 1]                                ]
        ]
    

'''
OClassifier(XGBClassifier) window = 14
'''
'''
WIN : 0
LOST: 0
'''
player1 = rps.BetaAgency(agents)
player2 = rps.Classifier(XGBClassifier(random_state = 17, n_estimators = 10, eval_metric = 'logloss'), 
                         window = 10)


#player2 = rps.MirrorSelfDecider(ahead = 2)
#player2 = rps.NMarkov(3, 6)
#player2 = rps.Randomer()

bat.battleground(player1, player2)
