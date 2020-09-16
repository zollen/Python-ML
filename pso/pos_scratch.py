'''
Created on Sep. 15, 2020

@author: zollen
'''

import numpy as np
import pprint

# f(a, b, c) = 3 * cos(a)^4 + 4 * cos(b)^3 + 2 sin(c)^2 * cos(c)^2 + 5
# constraint1: a + b + c = 1
# constraint2: c <= 0.8 
def formula(a, b, c):
    return 3 * np.cos(a) ** 4 + 4 * np.cos(b) ** 3 + 2 * np.sin(c) ** 2 * np.cos(c) ** 2 + 5

def numbers(_):
    arr = []
    for _ in range(3):
        arr.append(np.random.rand() * 0.5)
      
    sums = 0     
    for i in range(2):
        sums += arr[i]
    
    arr[2] = 1 - sums
    if arr[2] > 0.8:
        remained = (arr[2] - 0.8) / 2
        arr[2] = 0.8
        for i in range(2):
            arr[i] += remained
              
    return 0, arr[0], arr[1], arr[2]

def fitness(a, b, c):
    score = formula(a, b, c)
    sc1 = np.abs(1 - (a + b + c)) * 10
    sc2 = np.abs(c - 0.8) * 10
    
    return score - sc1 - sc2

def best_scores(best, rec):
    if rec[0] >  best[0]:
        best[0] = rec[0]
        best[1] = rec[1]
        best[2] = rec[2]
        best[3] = rec[3]
    

SWARMS_SIZE = 1000
positions = np.zeros((SWARMS_SIZE, 3 + 1))
velocities = np.zeros((SWARMS_SIZE, 3))
pbests = np.zeros((SWARMS_SIZE, 3 + 1))
gbest = np.squeeze(np.zeros((1, 3 + 1)))

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
    
    positions[:, 0] = list(map(fitness, positions[:, 1], positions[:, 2], positions[:, 3]))
    
    map(best_scores, pbests, positions)
        
    cbest = positions[np.argmax(positions[:,0])]
   
    best_scores(gbest, cbest)
  
    pp.pprint("Score: %0.4f ==> a: %0.4f, b: %0.4f, c: %0.4f" % 
          (cbest[0], cbest[1], cbest[2], cbest[3]))
    
    velocities = w * velocities + c1 * r1 * (pbests[:,1:4] - positions[:,1:4]) + c2 * r2 * (gbest[1:4] - positions[:,1:4])
   
    positions[:,1:4] = positions[:,1:4] + velocities
    


print("======= FINAL =======")
pp.pprint("Score: %0.4f ==> a: %0.4f, b: %0.4f, c: %0.4f" % 
          (gbest[0], gbest[1], gbest[2], gbest[3]))
