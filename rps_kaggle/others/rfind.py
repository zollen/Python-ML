'''
Created on Dec. 21, 2020

@author: zollen
'''

import rps_kaggle.lib.battle_lib as bat
import warnings


warnings.filterwarnings('ignore')


max_limit = 23  # can be modified
add_rotations = True

# number of predictors
numPre = 6
if add_rotations:
    numPre *= 3

# number of meta-predictors
numMeta = 4
if add_rotations:
    numMeta *= 3

# saves history
moves = ['', '', '']

beat = {'R':'P', 'P':'S', 'S':'R'}
dna =  {'RP':0, 'PS':1, 'SR':2,
        'PR':3, 'SP':4, 'RS':5,
        'RR':6, 'PP':7, 'SS':8}

p = ["P"]*numPre
m = ["P"]*numMeta
pScore = [[0]*numPre for i in range(8)]
mScore = [0]*numMeta

length = 0
threat = 0
output = "P"


def myagent(observation, configuration):    
    global max_limit, add_rotations, \
        numPre, numMeta, moves, beat, dna, \
        p, m, pScore, mScore, length, threat, output

    if observation.step < 2:
        output = beat[output]
        return {'R':0, 'P':1, 'S':2}[output]

    # - - - -

    input = "RPS"[observation.lastOpponentAction]

    # threat of opponent
    '''
    outcome = (we won) - (we lost)
    outcome = 1 if we won
    outcome = -1 if we lost
    '''
    outcome = (beat[input]==output) - (input==beat[output])
    threat = 0.9*threat - 0.1*outcome
    
    # refresh pScore
    for i in range(numPre):
        pp = p[i]        ## P, P, P ... (initially)
        bpp = beat[pp]   ## S, S, S ... (initially)
        bbpp = beat[bpp] ## R, R, R ... (initially)
        
        '''
        input <- enemy last move
        if enemy move = predefined move then +1
        if enemy move = predefined counter move then -1
        if enemy move = neither then 0
        '''
        pScore[0][i] = 0.9*pScore[0][i] + 0.1*((input==pp)-(input==bbpp))
        
        '''
        output <- my last move
        if my move = predefined move then +1
        if my move = predefined counter move then -1
        if my move = neither then 0
        '''
        pScore[1][i] = 0.9*pScore[1][i] + 0.1*((output==pp)-(output==bbpp))
        
        
        pScore[2][i] = 0.8*pScore[2][i] + 0.3*((input==pp)-(input==bbpp)) + \
                        0.1*(length % 3 - 1)
        pScore[3][i] = 0.8*pScore[3][i] + 0.3*((output==pp)-(output==bbpp)) + \
                        0.1*(length % 3 - 1)

    # refresh mScore
    for i in range(numMeta):
        mScore[i] = 0.9*mScore[i] + 0.1*((input==m[i])-(input==beat[beat[m[i]]])) + \
                    0.05*(length % 5 - 2)

    # refresh moves
    moves[0] += str(dna[input+output])
    moves[1] += input
    moves[2] += output

    # refresh length
    length += 1

    # new predictors
    limit = min([length,max_limit])
    for y in range(3):    # my moves, his, and both
        j = limit
        while j>=1 and not moves[y][length-j:length] in moves[y][0:length-1]:
            j-=1
        if j>=1:
            i = moves[y].rfind(moves[y][length-j:length],0,length-1)
            p[0+2*y] = moves[1][j+i] 
            p[1+2*y] = beat[moves[2][j+i]]

    # rotations of predictors
    if add_rotations:
        for i in range(int(numPre/3),numPre):
            p[i]=beat[beat[p[i-int(numPre/3)]]]

    # new meta
    for i in range(0,4,2):
        m[i] = p[pScore[i].index(max(pScore[i]))]
        m[i+1] = beat[p[pScore[i+1].index(max(pScore[i+1]))]]

    # rotations of meta
    if add_rotations:
        for i in range(4,12):
            m[i]=beat[beat[m[i-4]]]
    
    # - - -
    
    output = beat[m[mScore.index(max(mScore))]]

    if threat > 0.4:
        # ah take this!
        output = beat[beat[output]]

    return {'R':0, 'P':1, 'S':2}[output]



class Observation:
    def __init__(self):
        self.step = 0
        self.lastOpponentAction = 0
        
class Configuration:
    def __init__(self):
        self.signs = 3
        
observation = Observation()
configuration = Configuration()

'''
for rnd in range(0, 10):
    observation.step = rnd
    observation.lastOpponentAction = np.random.randint(0, 3)  
    print("Round [%d] MY PREDICTED MOVE: [%d]" %(rnd + 1, myagent(observation, configuration)))
'''    


import time
from xgboost import XGBClassifier
import rps_kaggle.lib.rps_lib as rps

SIGNS = ['ROCK', 'PAPER', 'SCISSORS']

class HighPerformer:
    
    def __str__(self):
        return "HighPerformer"
    
    def add(self, move):
        observation.step += 1
        observation.lastOpponentAction = move 
      
    def decide(self):  
        return myagent(observation, configuration)



def setup():           
    player1 = HighPerformer()
    player2 = rps.Classifier(XGBClassifier(random_state = 17, n_estimators = 10, eval_metric = 'logloss'), window = 10)
    return player1, player2



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
