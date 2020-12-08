'''
Created on Dec. 8, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
from deap import creator
from deap import base
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
            'cat.csv', 'xgb.csv', 'lasso.csv', 
            'eleasticnet.csv', 'linear.csv', 'svm.csv'
        ]

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
train_df = train_df[['Id', 'SalePrice']]

result_df = pd.DataFrame()

all_df = {}
for name in FILES:
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/models/', name))
    df = df.loc[df['Id'] < 1461, ['Id', 'SalePrice']]
    all_df[name] = df
 
POPLUATION_SIZE = 2000
TOURNAMENT_SIZE = 800
TOTAL_GENERATIONS = 50
TOTAL_PARAMS = 7
BEST = 10
SEED = 23

def evaluate(individual):
    test_df = pd.DataFrame()
    
    for index, name in zip(range(1, len(FILES)), FILES):
        if index == 1:
            test_df['SalePrice'] = individual[0] + (all_df[name]['SalePrice'] * individual[index])
        else:
            test_df['SalePrice'] = test_df['SalePrice'] + (all_df[name]['SalePrice'] * individual[index])
    
    return np.sqrt(mean_squared_error(train_df['SalePrice'], test_df['SalePrice'])), 


toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox.register("attribute", np.random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=TOTAL_PARAMS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.02, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=POPLUATION_SIZE)

toolbox.register("evaluate", evaluate)


def main():
    pop = toolbox.population(n=POPLUATION_SIZE)
    avgfit = POPLUATION_SIZE
    CXPB, MUTPB, NGEN = 0.8, 0.2, TOTAL_GENERATIONS

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for gen in range(NGEN):
        
        print("Generation: %d Fitness: %0.2f" % (gen, avgfit / POPLUATION_SIZE))
        # Select the next generation individuals
        offsprings = toolbox.select(pop, len(pop))
        # Clone the selected individuals      
        offsprings = [toolbox.clone(ind) for ind in offsprings]

        # Apply crossover and mutation on the offspring
        for i in range(1, POPLUATION_SIZE, 2):
            if np.random.random() < CXPB:
                toolbox.mate(offsprings[i - 1], offsprings[i])
                del offsprings[i - 1].fitness.values, offsprings[i].fitness.values

        for mutant in offsprings:
            if np.random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        avgfit = 0
        invalid_ind = [ind for ind in offsprings if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            avgfit += fit[0]

        # The population is entirely replaced by the offspring
        pop[:] = offsprings

    return tools.selBest(pop, k=BEST)

champions = main()

for champ in champions:
    print(np.round(champ, 4), " ==> ", evaluate(champ))