'''
Created on Apr 5, 2024

@author: STEPHEN
@url: https://pymoo.org/problems/multi/osy.html
@best: minimize problem

    f1(x) = - [ 25 * (x1 - 2)^2 + (x2 - 2)^2 + (x3 - 1)^2 + (x4 - 4)^2 + (x5 - 1)^2 ]
    f2(x) = x1^2 + x2^2 + x3^2 + x4^2 + x5^2 + x6^2

    C1(x): (x1 + x2 - 2) / 2 >= 0
    C2(x): (6 - x1 - x2) / 6 >= 0
    C3(x): (2 - x2 + x1) / 2 >= 0
    C4(x): (2 - x1 + 3x2) / 2 >= 0
    C5(x): (4 - (x3 - 3)^2 - x4) / 4 >= 0
    C6(x): ((x5 - 3)^2 + x6 - 4) / 4 >= 0
    
    0 <= x1,x2,x6 <= 10
    1 <= x3,x5    <= 5
    0 <= x4       <= 6
    
The Pareto-optimal region is a concatenation of five regions. Every region lies on some of 
the constraints. However, for the entire Pareto-optimal region, best(x4)=best(x6) = 0. In 
table below shows the other variable values in each of the five regions and the constraints 
that are active in each region.

          x1            x2                 x3     x5     Constraints
   ------------------------------------------------------------------------
           5            1             (1...5)    5      2,4,6
           5            1             (1...5)    1      2,4,6
 (4.056,...5)  (best(x1) - 2)/3            1     1      4,5,6
           0            2         (1...3.732)    1      1,3,6
    (0,...,1)      2 - best(x1)            1     1      1,5,6     
'''


import numpy as np
from pymoo.problems import get_problem
from optimization.lib.GreyWolfs import WolfPack


problem = get_problem("osy")


def osy6d(X):
    return problem.evaluate(X).squeeze()

def fitness(X):
    return osy6d(X)

def data(n):
    return np.random.rand(n, 6) * np.array([[10], [10], [4], [6], [4], [10]]) + \
            np.array([[0], [0], [1], [0], [1], [0]])



wolves = WolfPack(fitness, data, 'max', 1000, 
                  obj_type = 'multiple', LB = [0, 0, 1, 0, 1, 0], 
                                         UB = [10, 10, 5, 6, 5, 10] )
best = wolves.start(30)
print("WolfPack optimal {} ==> {}".format(best, osy6d(best[:,0], best[:,1], best[:,2])))