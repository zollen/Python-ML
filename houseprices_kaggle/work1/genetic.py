'''
Created on Dec. 10, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
import math
import operator
from deap import base
from deap import creator
from deap import tools
from sklearn.metrics import mean_squared_error
import warnings

SEED = 87

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
np.random.seed(SEED)



FILES = [ 
            'cat.csv', 
            'xgb.csv', 
            'lasso.csv', 
            'eleasticnet.csv', 
            'linear.csv',
            'svm.csv'
        ]

pp = pprint.PrettyPrinter(indent=3) 

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train_data.csv'))

result_df = pd.DataFrame()

repository = {}
for name in FILES:
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/models/', name))
    df = df.loc[df['Id'] < 1461, ['Id', 'SalePrice']]
    repository[name] = df




def evaluate(individual):
    res_df = pd.DataFrame()
    
    for index, name in zip([1, 2, 3, 4, 5, 6], FILES): 
        if index == 1:
            res_df['Id'] = repository[name]['Id']
            res_df['SalePrice'] = (5000 * individual[0]) + (individual[index] * repository[name]['SalePrice'])
        else:
            res_df['SalePrice'] = res_df['SalePrice'] + (individual[index] * repository[name]['SalePrice'])
            
    score = np.sqrt(mean_squared_error(train_df['SalePrice'], res_df['SalePrice']))
    return score, 
    
def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(np.random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [np.random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def update(part, best, phi1, phi2):
    u1 = (np.random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (np.random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))
    
    
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, 
    smin=None, smax=None, best=None)

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=7, pmin=-1.0, pmax=1.0, smin=-0.2, smax=0.2)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", update, phi1=0.2, phi2=0.2)
toolbox.register("evaluate", evaluate)

def main():
    pop = toolbox.population(n=10000)

    GEN = 120
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
            
        for part in pop:
            toolbox.update(part, best)

    print("=== BEST ===")
    pp.pprint(np.round(best, 4))
    print("BEST RMSE: %0.4f" % evaluate(best))
    
    return pop, best

'''
array([ 0.1355,  1.3787, -0.3565, -0.136 ,  0.0214,  0.0776,  0.0113])
BEST RMSE: 6649.9119
'''
if __name__ == "__main__":
    main()    

