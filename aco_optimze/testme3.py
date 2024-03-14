'''
Created on Mar 12, 2024

@author: STEPHEN
@url: https://courses.lumenlearning.com/waymakermath4libarts/chapter/shortest-path/
@url: https://youtu.be/KvRwplnIoEM

'''

import numpy as np
from aco_optimze.lib.aco_optimze import ACO_Optimization
''' 
Shortest path
A -> B -> D -> E -> G
'''
 
#                        A  B  C  D  E  F  G             
cost_matrix = np.array([[0, 1, 4, 0, 0, 0, 0],  #A
                        [1, 0, 0, 3, 6, 0, 0],  #B
                        [4, 0, 0, 2, 0, 5, 0],  #C
                        [0, 3, 2, 0, 2, 4, 0],  #D
                        [0, 6, 0, 2, 0, 2, 7],  #E
                        [0, 0, 5, 4, 2, 0, 6],  #F   
                        [0, 0, 0, 0, 7, 6, 0]]  #G
                        , dtype=float)

A=0
G=6
worker = ACO_Optimization(cost_matrix, [A], [G], 1000)
worker.start(40)

np.set_printoptions(precision=4)
print(worker.pheromone_matrix)