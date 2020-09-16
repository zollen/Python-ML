'''
Created on Sep. 15, 2020

@author: zollen
'''

import numpy as np
from deap import benchmarks as bm
import pprint

# f(a, b, c) = 3 * cos(a)^4 + 4 * cos(b)^3 + 2 sin(c)^2 * cos(c)^2 + 5
# constraint1: a + b + c = 1
# constraint2: c <= 0.8 
def formula(x1, x2):
    return bm.h1( [x1, x2] )[0]

def numbers(_):
   
    x1 = np.random.uniform(-6, 6)
    x2 = np.random.uniform(-6, 6)
              
    return 0, x1, x2

def fitness(x1, x2):
    return formula(x1, x2)


def best_scores(best, rec):
    if rec[0] >  best[0]:
        best[0] = rec[0]
        best[1] = rec[1]
        best[2] = rec[2]
    

SWARMS_SIZE = 1000
PARAM_SIZE = 2
positions = np.zeros((SWARMS_SIZE, PARAM_SIZE + 1))
velocities = np.zeros((SWARMS_SIZE, PARAM_SIZE))
pbests = np.zeros((SWARMS_SIZE, PARAM_SIZE + 1))
gbest = np.squeeze(np.zeros((1, PARAM_SIZE + 1)))

r1 = np.zeros((SWARMS_SIZE, 1))
r2 = np.zeros((SWARMS_SIZE, 1))
c1 = 1.0
c2 = 1.0
w = 1.0

pp = pprint.PrettyPrinter(indent=3) 

positions = np.apply_along_axis(numbers, 1, positions)
r1 = np.expand_dims([ np.random.rand() for x in r1 ], 1)
r2 = np.expand_dims([ np.random.rand() for x in r2 ], 1)

for _ in range(500):
    
    positions[:, 0] = list(map(fitness, positions[:, 1], positions[:, 2]))
    
    map(best_scores, pbests, positions)
        
    cbest = positions[np.argmax(positions[:,0])]
   
    best_scores(gbest, cbest)
  
    pp.pprint("Score: %0.4f ==> x1: %0.4f, x2: %0.4f" % 
          (cbest[0], cbest[1], cbest[2]))
    
    velocities = w * velocities + c1 * r1 * (pbests[:,1:3] - positions[:,1:3]) + c2 * r2 * (gbest[1:3] - positions[:,1:3])
   
    positions[:,1:3] = positions[:,1:3] + velocities
    


print("======= FINAL =======")
pp.pprint("Score: %0.4f ==> x1: %0.4f, x2: %0.4f" % 
          (gbest[0], gbest[1], gbest[2]))
