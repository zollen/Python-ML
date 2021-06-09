'''
Created on Sep. 16, 2020

@author: zollen
'''

import operator
import random

import numpy as np
import math

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from datetime import datetime, timedelta
import pandas_datareader.data as web
import pandas as pd
import time

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, 
    smin=None, smax=None, best=None)


end_date = datetime(2021, 6, 4)
start_date = end_date - timedelta(weeks=64)
test_size=36

def get_stock(TICKER):
   
    vvl = web.DataReader(TICKER, 'yahoo', start=start_date, end=end_date).Close
    vvl.index = [d.date() for d in vvl.index]
    
    prices = pd.DataFrame({
                            'Date' : vvl.index, 
                            'Prices' : vvl.values, 
                           })
    
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = prices.set_index('Date')
    prices = prices.asfreq(pd.infer_freq(prices.index), method="pad")
    prices['Prices'] = prices['Prices'].astype('float64')
    
    return prices

def evaludate(params):
    
    global y_to_train, y_to_test, fh
        
    exog = pd.DataFrame({'Date': y.index})
    exog = exog.set_index(exog['Date'])
    exog.drop(columns=['Date'], inplace=True)
    
    cols = []
    for coeff in params:
        exog['sin' + str(coeff)] = np.sin(coeff * np.pi * exog.index.dayofyear / 365.25)
        exog['cos' + str(coeff)] = np.cos(coeff * np.pi * exog.index.dayofyear / 365.25)
        cols.append('sin' + str(coeff))
        cols.append('cos' + str(coeff))
        
    exog_train = exog.loc[y_to_train.index]
    exog_test = exog.loc[y_to_test.index]
        
    model = ARIMA(order=(2, 1, 2), seasonal_order=(0, 1, 0, 8), suppress_warnings=True)
    model.fit(y_to_train['Prices'], X=exog_train[cols])
    y_forecast = model.predict(fh, X=exog_test[cols])
    return mean_absolute_percentage_error(y_to_test, y_forecast)

def generate_params(length):
    arr = []
    
    for _ in range(length):
        arr.append(np.random.randint(1, 500))
    
    return arr
        
def evaluate(p):
    score = evaludate(p)
    return score,

def fesiable(p):
    
    if len(np.unique(p)) != len(p):
        return False
    
    return all(n > 0 for n in p)

def distance(p):
    return 3

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
    part[:] = list(map(operator.add, part, part.speed))


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=5, smin=-7, smax=7)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=3, phi2=3)
toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.DeltaPenality(fesiable, (99999,)))

y = get_stock('VVL.TO')
y_to_train, y_to_test = temporal_train_test_split(y, test_size=test_size)
fh = ForecastingHorizon(y_to_test.index, is_relative=False)

def main():
    
    '''
    FINAL: 0.00795964  params:  [87.0, 141.0, 170.0, 346.0, 267.0] 1722.1818044185638
    '''
    
    start_time = time.time()
    
    pop = toolbox.population(n=100)
 
    GEN = 20
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

    
    end_time = time.time()

    print("Processing Time: ", end_time - start_time, "secs")
    print("FINAL: %0.8f" % best.fitness.values, " params: ", best)
    
    return pop, best

if __name__ == "__main__":
    main()