'''
Created on Dec. 16, 2020

@author: zollen
'''

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
import warnings


warnings.filterwarnings('ignore')



def setup():
    
    if True:
        xgb15 = rps.Classifier(XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), window = 15)
        agents = [
            [ enm2.MarkovChain(3, 0.9),                                               [], [] ],
            [ enm3.MemoryPatterns(min_memory=40, max_memory=140, warmup=20),          [], [] ],
            [ enm3.MemoryPatterns(min_memory=50, max_memory=140, warmup=20),          [], [] ],
            [ enm4.Iocaine(num_predictor = 160),                                      [], [] ],
            [ xgb15,                                                                  [], [] ]
        ]

        calculators = [
           rps.PopularityScorer(agents),    # 8,11  14,6 10,9 10,10           12,7
           rps.OutComeScorer(agents),       # 7,11  14,6            12,8      12,7
           rps.Last5RoundsScorer(agents),   # 10,10      10,9            8,12 12,7
        #    rps.BetaScorer(agents)         # 9,11            10,10 12,8 8,12
        ]

        player1 = rps.StatsAgency(calculators, agents, random_threshold = -10)
    
   
    if False:
        
        markovChain = enm2.MarkovChain(3, 0.9)
        iocaine2 = enm4.Iocaine(num_predictor = 140)
        xgb15 = rps.Classifier(XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), window = 10)
        
        agents1 = [
                    [ markovChain,            [0, 0], [0] ],
                    [ iocaine2,               [0, 0], [0] ],
                    [ xgb15,                  [0, 0], [0] ]
            ]
        
        player2 = rps.BetaAgency(agents1, decay = 1.1)
        
    if True:
        xgb1 = rps.Classifier(XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), window = 15)
        xgb2 = rps.Sharer(xgb1, ahead = 1)
        xgb3 = rps.Sharer(xgb1, ahead = 2)
        managers1 = [
                         [ XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), [0, 0], [0] ],
                         [ RandomForestClassifier(n_estimators = 10),                 [0, 0], [0] ],
                         [ KNeighborsClassifier(),                                    [0, 0], [0] ],
                         [ SVC(kernel = 'rbf'),                                       [0, 0], [0] ]
                     ]
        
        agents1 = [ xgb1, xgb2, xgb3 ]
             
        player2 = rps.MetaAgency(managers1, agents1, window = 20, history = 50, random_threshold = -10, randomness = 0.1)


    if False:
        player2 = enm.MultiArmsBandit()
    if False:
        player2 = enm2.MarkovChain(3, 0.9)
    if False:
        player2 = enm4.Iocaine(num_predictor = 160)
    if False:
        player1 = enm3.MemoryPatterns(min_memory=40, max_memory=140, warmup=20)
    if False:
        player1 = enm5.GreenBerb()
    if False:
        player2 = rps.Classifier(XGBClassifier(n_estimators = 10, eval_metric = 'logloss'), window = 15)
    

   
    
    return player1, player2




if False:    
    player1, player2 = setup()
    bat.battleground(player1, player2)
else:   
    totalwin = 0
    totalloss = 0
    totaleven = 0
    totalratio = 0.0
    for rnd in range(20):
        
        player1, player2 = setup()
       
        win1, win2 = bat.battleground(player1, player2, verbose = False)
        if win1 > win2:
            totalwin += 1
            totalratio += win1 / win2
        elif win1 < win2:
            totalloss += 1
        else:
            totaleven += 1   
              
        print("Match [{:>2}] WON [{}]  LOST [{}] RATIO [{:2.4f}]".format(rnd + 1, win1, win2, win1 / win2))
     
    print("=================== TOTAL =======================")    
    print("WON [{:<2}], LOST [{:<2}] EVEN [{:<2}] WINNING RATIO [{:2.4f}]".format(totalwin, totalloss, totaleven, 0 if totalwin == 0 else totalratio / totalwin))
