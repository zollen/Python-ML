'''
Created on Sep. 7, 2020

@author: zollen
'''

import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
from deap import creator
from deap import base
from deap import tools
from sklearn.impute import KNNImputer
import titanic_kaggle.second.titanic_lib as tb

POPLUATION_SIZE = 1000
TOURNAMENT_SIZE = 400
GUESTS_RATIO = 0.50
NGEN=200
TEST_MIN_VALUE = 2
TEST_MAX_VALUE = 79
total_boundaries = 5
SEED = 87


pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(SEED)

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))

tb.reeigneeringTitle(train_df)
ddff = tb.normalize({}, train_df, ['Title', 'Sex', 'Embarked', 'Pclass' ])
columns = [ 'Age', 'Title', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Survived' ]
imputer = KNNImputer(n_neighbors=13)
imputer.fit(ddff[columns])   
ages = imputer.transform(ddff[columns])
train_df['Age'] = ages[:, 0]
train_df = train_df.sort_values(by='Age')


itable = train_df[['Age', 'Survived']]

def populateBoundaries():
    
    boundaries = []
    
    while len(boundaries) < total_boundaries:
        val = np.random.randint(TEST_MIN_VALUE, TEST_MAX_VALUE)
        if val not in boundaries:
            boundaries.append(val)
                
    boundaries.sort()
    
    boundaries.append(boundaries[-1])
    
    return boundaries

def scoreIndividual(individual):

    start = -9999999
    end = 0
    
    msg = individual.short()
        
    for i in range(total_boundaries + 1):
        
        if i > 0:
            msg += " |"
        
        if i == total_boundaries:
            end = 9999999
        else:
            end = individual[i]
            
        vals = itable.loc[(itable['Age'] >= start) & (itable['Age'] < end), 'Survived']
        res = np.bincount(vals)
        if len(res) > 0:
            msg += " {%d, %d} => %d, %d" % (start, end, res[0], res[1])

        start = end
        
    return msg

def evaluateIndividual(individual):
    score_sum = 0   
    start = -9999999
    end = 0
    
    individual.scores = []
        
    for i in range(total_boundaries + 1):
        
        if i == total_boundaries:
            end = 9999999
        else:
            end = individual[i]
           
        vals = itable.loc[(itable['Age'] >= start) & (itable['Age'] < end), 'Survived']
        if len(vals) > 0:
            score = np.min(np.bincount(vals)) 
            individual.scores.append(score)           
            score_sum += score
        else:
            individual.scores.append(100.0)
            score_sum += 100.0
                
        start = end
  
    return score_sum,

def mateTwoIndividuals(child1, child2):
               
#    print("BEFORE: ", child1, child2)

    for i in range(total_boundaries):
        chance = np.random.randint(0, 4)
        
        adjustment = 0
        if chance == 0:
            adjustment = 1
        elif chance == 2:
            adjustment = -1 
            
        score1 = (child1.scores[i] + child1.scores[i + 1]) / 2
        score2 = (child2.scores[i] + child2.scores[i + 1]) / 2

                    
        if score1 < score2:    
            child = child2
            child2.boundaries[i] = child1.boundaries[i] + adjustment
        else:
            child = child1
            child1.boundaries[i] = child2.boundaries[i] + adjustment
        
        if child.boundaries[i] <= TEST_MIN_VALUE:
            child.boundaries[i] = TEST_MIN_VALUE
        elif child.boundaries[i] >= TEST_MAX_VALUE:
            child.boundaries[i] = TEST_MAX_VALUE
  
    child1.boundaries[total_boundaries] = child1.boundaries[total_boundaries - 1]
    child2.boundaries[total_boundaries] = child2.boundaries[total_boundaries - 1]
            
    child1.normalize()
    child2.normalize()
    
#    print("AFTER: ", child1, child2)
            
    return child1, child2

def mutateIndividual(individual, indpb):
               
    if np.random.random() <= indpb:
        pos = np.random.randint(0, total_boundaries)
                
        if pos == 0:
            individual[pos] = individual[pos] + 1
            return individual
        
        if pos == len(individual) - 1:
            individual[pos] = individual[pos] - 1
            return individual
         
        if np.random.randint(0, 1) == 2:
            if individual[pos] - 1 > individual[pos - 1]:
                individual[pos] = individual[pos] - 1
            else:
                individual[pos] = individual[pos] + 1
        else:
            if individual[pos] + 1 < individual[pos + 1]:
                individual[pos] = individual[pos] + 1
            else:
                individual[pos] = individual[pos] - 1
                
        individual.boundaries[total_boundaries] = individual.boundaries[total_boundaries - 1]
                
        individual.normalize()
                
    return individual


def selectIndividuals(individuals, k, tournsize):
    
    global GUESTS_RATIO
       
    scores = list(map(lambda ind : evaluateIndividual(ind), individuals))
       
    dd = pd.DataFrame({
        "individual": individuals,
        "scores": scores
        })
    
    dd.sort_values(by="scores", inplace=True)
        
    people = dd.head(tournsize)['individual'].tolist()
    
    guests = int(POPLUATION_SIZE * GUESTS_RATIO)
    for i in range(k - tournsize - guests):
        people.append(toolbox.clone(individuals[i % tournsize]))
        
    for i in range(guests):
        people.append(toolbox.individual())
        
    np.random.shuffle(people)
    
    GUESTS_RATIO *= 0.9      
            
    return people


class Passengers:
      
    def __init__(self, func):
        self.boundaries = np.squeeze(list(func))
        self.fitness.values = evaluateIndividual(self)
        
    def __getitem__(self, i):
        return self.boundaries[i]
    
    def __setitem__(self, i, val):
        self.boundaries[i] = val
    
    def __len__(self):
        return len(self.boundaries)
    
    def __str__(self):
        return self.short() + " - " + str(list(map(lambda x : round(x, 2), self.scores)))
    
    def short(self):
        return str(self.boundaries) + " (" + str(round(self.fitness.values[0], 4)) + ")"
    
    def set_boundaries(self, val):
        self.boundaries = val[:]
        self.fitness.values = ()
        
    def normalize(self):
        self.boundaries.sort()
        self.fitness.values = evaluateIndividual(self)
        


toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", Passengers, fitness=creator.FitnessMin)


toolbox.register("gen_boundaries", populateBoundaries)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gen_boundaries, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


   
def varAnd(gen, population, cxpb, mutpb):
    offsprings = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offsprings), 2):
        if random.random() < cxpb:
            offsprings[i - 1], offsprings[i] = toolbox.mate(offsprings[i - 1],
                                                          offsprings[i])
            del offsprings[i - 1].fitness.values, offsprings[i].fitness.values
            
    for i in range(len(offsprings)):
        if random.random() < mutpb:
            offsprings[i] = toolbox.mutate(offsprings[i])
            del offsprings[i].fitness.values
           
    return offsprings

def proceed():
    population = toolbox.population(n=POPLUATION_SIZE)
    avgfit = POPLUATION_SIZE

    for gen in range(NGEN):
#        print("Generation: %d Fitness: %0.2f" % (gen, avgfit / POPLUATION_SIZE))
        offspring = varAnd(gen, population, cxpb=0.8, mutpb=0.05)
        
        avgfit = 0
        fits = toolbox.map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
            avgfit += fit[0]
            
        population = toolbox.select(offspring, k=POPLUATION_SIZE, tournsize=TOURNAMENT_SIZE)

    return tools.selBest(population, k=10)

toolbox.register("evaluate", evaluateIndividual)
toolbox.register("mate", mateTwoIndividuals)
toolbox.register("mutate", mutateIndividual, indpb=0.05)
toolbox.register("select", selectIndividuals, tournsize=POPLUATION_SIZE)

for i in [ 6, 7, 8, 9, 10, 12, 13 ]:
    total_boundaries = i
    print("<", i, ">", scoreIndividual(proceed()[0]))