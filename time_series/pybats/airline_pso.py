'''
Created on Sep. 16, 2020

@author: zollen
'''

from contextlib import contextmanager
import sys, os

import operator
import random

import numpy as np
import math

from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from pybats.analysis import *
from pybats.point_forecast import *
from pybats.plot import *

from datetime import datetime, timedelta
import pandas_datareader.data as web
import pandas as pd
import time
import threading

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from numba.cuda.stubs import threadIdx
import warnings

warnings.filterwarnings('ignore')

np.random.seed(int(round(time.time())))

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, 
    smin=None, smax=None, best=None)


end_date = datetime(2021, 6, 4)
start_date = end_date - timedelta(weeks=64)


class Worker(threading.Thread):     
    
    def __init__(self, threadId, rnd, queue):
        threading.Thread.__init__(self)
        self.threadID = threadId
        self.rnd = rnd
        self.queue = queue
        
    def run(self):
        
        try:       
            while True:
                self.evaluate(self.queue.pop(0))   
   
        except:
            pass
            
        
    def evaluate(self, part):
        
        global TEST_SIZE, y_to_train, y_to_test
        
        scores = []
        for _ in range(0, 5):
      
            with suppress_stdout():
                samples = analysis(y_to_train.values, 
                    k=36, 
                    forecast_start=forecast_start, 
                    forecast_end=forecast_end,
                    family='poisson',
                    seasPeriods=[12], 
                    seasHarmComponents=[ part ],
                    ntrend=2,  
                    prior_length=5, 
                    dates=y_to_train.index,
                    rho=0.5,
                    deltrend=0.95,      # Discount factor on the trend component (the intercept)
                    delregn=0.98,       # Discount factor on the regression component
                    delseas=0.98,       # Discount factor on the seasonal component
                    ret = ['forecast'])
            
            forecast = median(samples).flatten()[-TEST_SIZE:] 
            scores.append(mean_absolute_percentage_error(y_to_test.values, forecast))
        
        score = np.mean(scores)
        part.fitness.values = score, 
        
        global best, wlock
        
        wlock.acquire()

        if not part.best or part.best.fitness < part.fitness:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values
        if not best or best.fitness < part.fitness:
            best = creator.Particle(part)
            best.fitness.values = part.fitness.values
            print("InProgress[%3d][%3d][%3s] => Score: %0.8f" % (self.rnd, threading.activeCount(), str(self.threadID), score), " params: ", part)
            
        wlock.release()
        

def generate_params(length):
    
    arr = []
    
    for _ in range(length):
        arr.append(np.random.randint(1, 15))
    
    return arr
        
def evaluate(p):
    return 1, 

def fesiable(p):
    
    if len(np.unique(p)) != len(p):
        return False
    
    return all(n > 0 for n in p)


def generate(size, smin, smax):
    part = creator.Particle(generate_params(size)) 
    part.speed = [ random.randint(0, 2) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part, best, phi1, phi2):
    u1 = ( random.randint(0, phi1) for _ in range(len(part)))
    u2 = ( random.randint(0, phi2) for _ in range(len(part)))

    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))

    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    
    if best.fitness.values != part.fitness.values:
        part[:] = list(map(operator.add, part, part.speed))
        


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=3, smin=-15, smax=15)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=1, phi2=2)
toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.DeltaPenality(fesiable, (99999,)))


TRAINING_SIZE=36
TEST_SIZE = 36

y = load_airline()
y_to_train, y_to_test = temporal_train_test_split(y, test_size=TEST_SIZE)


forecast_start = y_to_train.index[-TRAINING_SIZE] 
forecast_end = y_to_train.index[-1] 

wlock = threading.Lock()
best = None



def main():
    
    start_time = time.time()
    
    pop = toolbox.population(n=100)
 
    GEN = 3

    for g in range(GEN):
         
        qq = pop.copy()
        threads = []
        
        for id in range(0, 2):
            threads.append(Worker(id, g, qq))

        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join() 
                
        print("Round: ", time.time() - start_time, g)
            
        for part in pop:
            toolbox.update(part, best)

    
    end_time = time.time()

    print("========================================================")
    print("Processing Time: ", end_time - start_time, "secs")
    print("FINAL: %0.8f" % best.fitness.values, " params: ", best)
    
    return pop, best

if __name__ == "__main__":
    main()