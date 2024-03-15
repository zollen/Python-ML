'''
Created on Mar 12, 2024

@author: STEPHEN
    ---> 1 --->
0 --|          |---> 3
    ---> 2 --->
'''

import numpy as np
from aco_optimze.lib.aco_optimze import ACO_Optimization
 
 
cost_matrix = np.array([[0, 3, 1, 0],
                        [0, 0, 0, 2],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0]], dtype=float)

worker = ACO_Optimization(cost_matrix, [0], [3], 10)
worker.start()

np.set_printoptions(precision=8)
print(worker.pheromone_matrix)
print(worker.best_path(['0', '1', '2', '3']))
print("Total Score: ", worker.best_scores())