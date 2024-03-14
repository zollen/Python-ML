'''
Created on Mar 12, 2024

@author: STEPHEN
@url: https://stackoverflow.com/questions/10254542/dijkstras-algorithm-does-not-generate-shortest-path

'''

import numpy as np
from aco_optimze.lib.aco_optimze import ACO_Optimization
''' 
Shortest path
  4    2    2    2    2    2    2      = 16
A -> C -> F -> I -> M -> P -> S -> Z    

  1    5    2    2    2    2    2      = 16
A -> D -> F -> I -> M -> P -> S -> Z
'''
 
#                        A  B  C  D  E  F  G  H  I  J  K  L  M  N  O  P  Q  R  S  T  Z           
cost_matrix = np.array([[0, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #A
                        [2, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #B
                        [4, 3, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #C
                        [1, 0, 0, 0, 0, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #D
                        [0, 1, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #E
                        [0, 0, 2, 5, 0, 0, 3, 3, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #F   
                        [0, 0, 0, 4, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #G
                        [0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 1, 0, 0, 8, 0, 0, 0, 0, 0, 0],  #H
                        [0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0],  #I
                        [0, 0, 0, 0, 0, 4, 0, 0, 3, 0, 6, 0, 6, 3, 0, 0, 0, 0, 0, 0, 0],  #J
                        [0, 0, 0, 0, 0, 0, 2, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0],  #K
                        [0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 3, 0, 6, 0, 0, 0, 0, 0, 0],  #L
                        [0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 0, 3, 0, 5, 4, 2, 0, 0, 0, 0, 0],  #M
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 5, 0, 0, 0, 2, 1, 0, 0, 0],  #N
                        [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 6, 4, 0, 0, 2, 0, 0, 6, 0, 0],  #O
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 1, 0, 2, 1, 0],  #P
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 8, 0, 3, 0],  #Q
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 8, 0, 0, 5, 0],  #R
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 2, 0, 0, 0, 0, 2],  #S
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 0, 0, 8],  #T
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8, 0]   #Z
                        ], dtype=float)

A=0
Z=20
worker = ACO_Optimization(cost_matrix, [A], [Z], 1000)
worker.start(20)

np.set_printoptions(precision=4)
print(worker.pheromone_matrix)
print(worker.print_best_path())