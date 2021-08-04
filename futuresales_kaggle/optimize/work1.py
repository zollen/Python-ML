'''
Created on Aug. 2, 2021

@author: zollen
'''
from deap import base
from deap import creator
from deap import tools
import operator
import math

import pandas as pd
import numpy as np
import time
import threading
import warnings


warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

np.random.seed(int(time.time()))

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

def evaluate(p, data):
    
        return abs(sum( x[11] - (p[0] +      
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
                        p[11] * x[10]) for x in data ))
    


class Worker(threading.Thread):     
    
    def __init__(self, threadId, data, rnd, input, output):
        threading.Thread.__init__(self)
        self.threadID = threadId
        self.rnd = rnd
        self.data = data
        self.input = input
        self.output = output
        
    def run(self):      
        try:       
            while True:
                self.output.append(self.process(self.input.pop(0)))
        except:
            pass

    def process(self, part):
        score = evaluate(part, self.data)
        part.fitness.values = score, 
        
        global best, wlock
        
        wlock.acquire()
        
        if not part.best or part.best.fitness < part.fitness:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values
        if not best or best.fitness < part.fitness:
            best = creator.Particle(part)
            best.fitness.values = part.fitness.values
            print("InProgress[%3d][%3d][%3s] => Score: %15.4f" % 
                  (self.rnd, threading.activeCount(), str(self.threadID), score), 
                  " params: ", part)
             
        wlock.release()
        
        return part




creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, 
               smin=None, smax=None, best=None)

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=12, pmin=-2, pmax=2, smin=-1, smax=1)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=1, phi2=1)
toolbox.register("evaluate", evaluate)

wlock = threading.Lock()
best = None

'''
Score:  178297551.8095  params:  [2.811291870424368, 3.772966339674271, 0.8490897745104498, 0.05241583572693642, -6.024180921413107, -9.16834376712352, -6.336982897871892, -7.9467655023426085, -2.840321262776073, -2.703765155062582, 1.818534510671384, 1.3115486005210784]
Score:    5023095.4341  params:  [0.09615478202863481, 2.789629780067158, 0.8235775906629333, 0.01780851560761154, -4.555990430221731, -3.6434438067603763, -4.253546687419635, -2.3914360745301417, -2.1549622899750265, -2.3490020589144183, 0.2226727551740212, 1.1419247882456074]

'''
def main():
    pop = toolbox.population(n=100)
    
    fbest = creator.Particle([0.09615478202863481, 2.789629780067158, 0.8235775906629333, 0.01780851560761154, -4.555990430221731, -3.6434438067603763, -4.253546687419635, -2.3914360745301417, -2.1549622899750265, -2.3490020589144183, 0.2226727551740212, 1.1419247882456074])

    pop.append(fbest)

    GEN = 20
    
    for g in range(1, GEN + 1):
        
        start_time = time.time()
        
        threads = []
        processed = []
        
        for iid in range(0, 50):
            threads.append(Worker(iid, data, g, pop, processed))

        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join() 
            
        end_time = time.time()
                
        print("Gen: ", g, " Output: ", len(processed), " Time: ", end_time - start_time)
        
        pop = processed
        
        for part in pop:
            toolbox.update(part, best)

    return pop, best


if __name__ == "__main__":
    start_t = time.time()
    children, best = main()
    end_t = time.time()
    print("BEST: ", best)
    print("SCORE: ",  evaluate(best, data))
    print("ELAPSE TIME: ", end_t - start_t)
    
    
