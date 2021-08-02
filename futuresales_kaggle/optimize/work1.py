'''
Created on Aug. 2, 2021

@author: zollen
'''
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import operator
import math

import pandas as pd
import numpy as np
import time
import futuresales_kaggle.lib.future_lib as ft
import warnings
from ngboost import scores


warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

np.random.seed(0)

label = 'item_cnt_month'
base_features = ['date_block_num', 'shop_id', 'item_id', 
            'shop_category', 'shop_city', 
            'item_category_id', 'name2', 
            'name3', 'item_type', 'item_subtype', 'item_price', label]




train = pd.read_csv('../data/monthly_train.csv')
items = pd.read_csv('../data/monthly_items.csv')
cats = pd.read_csv('../data/monthly_cats.csv')
shops = pd.read_csv('../data/monthly_shops.csv')


'''
merge cats, shops and items
'''
items_cats = pd.merge(items, cats, how='left', on='item_category_id')
train_item_cats = pd.merge(train, items_cats, how='left', on='item_id')
train_item_cats_shops = pd.merge(train_item_cats, shops, how='left', on='shop_id')

data = train_item_cats_shops[base_features].values.tolist()




'''
clip values between 0 and 20
'''
train_item_cats_shops[label] = train_item_cats_shops[label].clip(0, 20)

'''
Optimization
'''
def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(np.random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [np.random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part, best, phi1, phi2):
    u1 = ( np.random.uniform(0, phi1) for _ in range(len(part)))
    u2 = ( np.random.uniform(0, phi2) for _ in range(len(part)))

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

'''
np.apply_along_axis               - TIME:  109.6327314376831
native array with inline for loop - TIME:  14.25113844871521
'''
def evaluate(p, data):
    
    score = sum( x[11] - (p[0] + 
                    p[1]  * x[0] + 
                    p[2]  * x[1] + 
                    p[3]  * x[2] + 
                    p[4]  * x[3] + 
                    p[5]  * x[4] + 
                    p[6]  * x[5] + 
                    p[7]  * x[6] + 
                    p[8]  * x[7] + 
                    p[9]  * x[8] +
                    p[10] * x[9] + 
                    p[11] * x[10]) for x in data )

    return score, 




creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, 
               smin=None, smax=None, best=None)

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=12, pmin=-10, pmax=10, smin=-5, smax=5)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=3, phi2=3)
toolbox.register("evaluate", evaluate)


def main():
    pop = toolbox.population(n=10)

    GEN = 2
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part, data)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
           
        for part in pop:
            toolbox.update(part, best)

    return pop, best


if __name__ == "__main__":
    start_t = time.time()
    children, best = main()
    end_t = time.time()
    print(best)
    print("ELAPSE TIME: ", end_t - start_t)
    
    
