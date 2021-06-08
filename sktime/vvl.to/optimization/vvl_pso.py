'''
Created on Sep. 16, 2020

@author: zollen
'''

import operator
import random

import numpy as np
import math

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, 
    smin=None, smax=None, best=None)



def generate_params(length):
    arr = []
    
    for _ in range(length):
        arr.append(np.random.randint(1, 500))
    
    return arr

def mape(params):
    return np.sum(params)


def evaluate(p):
    score = mape(p)
    return score,

def fesiable(p):
    
    if len(np.unique(p)) != len(p):
        return False
    
    return all(n > 0 for n in p)

def distance(p):
    return 3

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(generate_params(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=3, pmin=0, pmax=1, smin=-0.2, smax=0.2)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=0.2, phi2=0.2)
toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.DeltaPenality(fesiable, (99999,)))

def main():
    pop = toolbox.population(n=1000)
 
    GEN = 500
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
            

    print("FINAL: %0.8f" % best.fitness.values, " params: ", best)
    
    return pop, best

if __name__ == "__main__":
    main()