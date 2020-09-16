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


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
    smin=None, smax=None, best=None)

def numbers():
    arr = []
    for _ in range(3):
        arr.append(1.0 / 3.0)
    
    scen, s1, s2 = np.random.randint(0, 3), np.random.rand() * 0.3, np.random.rand() * 0.3
    
    if scen == 0:
        arr[0], arr[1], arr[2] = arr[0] + s1 + s2, arr[1] - s1, arr[2] - s2
    elif scen == 1:
        arr[0], arr[1], arr[2] = arr[0] - s1, arr[1] + s1 + s2, arr[2] - s2
    else:
        arr[0], arr[1], arr[2] = arr[0] - s1, arr[1] - s2, arr[2] + s1 + s2   
   
    if arr[2] > 0.8:
        remained = (arr[2] - 0.8) / 2
        arr[2] = 0.8
        arr[0], arr[1] = arr[0] + remained, arr[1] + remained
       
    return arr

def formula(a, b, c):
    return 3 * np.cos(a) ** 4 + 4 * np.cos(b) ** 3 + 2 * np.sin(c) ** 2 * np.cos(c) ** 2 + 5

def evaluate(p):
    score = formula(p[0], p[1], p[2])
    return score,

def fesiable(p):
    sums = p[0] + p[1] + p[2]
    if sums > 0.999999 and sums < 1.0000001 and p[2] <= 0.8 and p[0] > 0 and p[1] > 0 and p[2] > 0:
        return True    
    return False

def distance(p):
    return 3

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(numbers()) 
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
toolbox.decorate("evaluate", tools.DeltaPenality(fesiable, (-1,)))

def main():
    pop = toolbox.population(n=1000)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

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

        # Gather all the fitnesses in one list and print the stats
#        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
#        print(logbook.stream)

    print(best, " ==> ", formula(best[0], best[1], best[2]))
    print("BEST ==> ", formula(0.1, 0.1, 0.8))
    
    return pop, logbook, best

if __name__ == "__main__":
    main()