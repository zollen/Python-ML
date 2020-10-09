'''
Created on Oct. 8, 2020

Non-dominated Sorting Genetic Algorithm 3 (NSGA-3)
Multiple-Objectives Optimization

@author: zollen
'''

import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import benchmarks
import numpy as np

# simultaneously maximize sin(x) and mininize cos(x)
# the target is locate the x that has the best of both objectives
# within inteval of -1 and 1:
# x = [-1.0], [7.777105436868048e-06], [-0.6508562126643136], [-0.9982355282437874], [-0.02815364548010235]
# achieves the best results
def multiObjsFunc(x):
    return np.sin(x[0]), np.cos(x[0]) 
    

BOUND_LOW, BOUND_UP = -1.0, 1.0
NOBJS = 2
NGEN = 400
MU=100
CXPB = 0.9
MUTPB = 0.1
REFERENCE_PTS = 2
NDIMS = 1  # one point - x



ref_points = tools.uniform_reference_points(NOBJS, REFERENCE_PTS)

creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIMS)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", multiObjsFunc)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIMS)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

def main(seed=None):
    random.seed(seed)
    
    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)
    

    return toolbox.select(pop, 5)

if __name__ == "__main__":
    pop = main()
    
    print(pop)