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
           5            1             (1...5)      5      2,4,6
           5            1             (1...5)      1      2,4,6
 (4.056,...5)  (best(x1) - 2)/3            1       1      4,5,6
           0            2         (1...3.732)      1      1,3,6
    (0,...,1)      2 - best(x1)            1       1      1,5,6     
'''


import numpy as np
from paretoset import paretoset
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem
from optimization.lib.GreyWolfs import WolfPack


problem = get_problem("osy")

def osy6d(X):
    res = problem.evaluate(X)
    return res[0]

def normalize(X):
    minn = np.min(X, axis=0)
    maxx = np.max(X, axis=0)
    return (X - minn) / (maxx - minn)

def fitness(X):
    res = osy6d(X)
    score =  normalize(res[:,0]) * 0.5 + (res[:,1]) * 0.5
    c1 = (X[:,0] + X[:,1] - 2) / 2
    c2 = (6 - X[:,0] - X[:,1]) / 6
    c3 = (2 - X[:,1] + X[:,0]) / 2
    c4 = (2 - X[:,0]  + 3 * X[:,1]) / 2
    c5 = (4 - (X[:,2] - 3)**2 - X[:,3]) / 4
    c6 = ((X[:,4] - 3)**2 + X[:,5] - 4) / 4
    c1_score = np.where(c1 >= 0, 0, 500000)
    c2_score = np.where(c2 >= 0, 0, 500000)
    c3_score = np.where(c3 >= 0, 0, 500000)
    c4_score = np.where(c4 >= 0, 0, 500000)
    c5_score = np.where(c5 >= 0, 0, 500000)
    c6_score = np.where(c6 >= 0, 0, 500000)
    return score + c1_score + c2_score + c3_score + c4_score + c5_score + c6_score

def data(n):
    return np.random.rand(n, 6) * np.array([[10, 10, 4, 6, 4, 10]]) + \
            np.array([[0, 0, 1, 0, 1, 0]])






wolves = WolfPack(osy6d, fitness, data, 'min', 10000, 
                  obj_type = 'multiple', LB = [[0, 0, 1, 0, 1, 0]], 
                                         UB = [[10, 10, 5, 6, 5, 10]] )
best = wolves.start(200)
vals = osy6d(best)
for i in range(best.shape[0]):
    print("WolfPack optimal f({0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}) ==> [ {6:.4f}, {7:.4f} ]".format(
        best[i, 0], best[i, 1], best[i, 2], best[i, 3], best[i, 4], best[i, 5],
        vals[i, 0], vals[i, 1]))

   
''' 
# create the reference directions to be used for the optimization
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

# create the algorithm object
algorithm = NSGA3(pop_size=2000,
                  ref_dirs=ref_dirs)

# execute the optimization
res = minimize(problem,
               algorithm,
               seed=1,
               termination=('n_gen', 600))

np.printoptions(precision=4)
for i in range(res.X.shape[0]):
    print("NAGA3 optimal f({0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}) ==> [ {6:.4f}, {7:.4f} ]".format(
        res.X[i, 0], res.X[i, 1], res.X[i, 2], 
        res.X[i, 3], res.X[i, 4], res.X[i, 5], 
        res.F[i, 0], res.F[i, 1]))
'''
'''
NAGA3 
NAGA3 optimal f(5.0000, 1.0000, 2.0173, 0.0000, 5.0000, 0.0002) ==> [ -259.0301, 55.0689 ]
NAGA3 optimal f(4.3802, 0.7941, 1.0001, 0.0000, 1.0000, 0.0000) ==> [ -159.0848, 21.8167 ]
NAGA3 optimal f(5.0000, 1.0000, 5.0000, 0.0000, 5.0000, 0.0002) ==> [ -273.9965, 75.9994 ]
NAGA3 optimal f(4.8631, 0.9545, 1.0000, 0.0000, 1.0000, 0.0000) ==> [ -222.0311, 26.5612 ]
NAGA3 optimal f(0.0000, 2.0000, 1.8533, 0.0000, 1.0000, 0.0001) ==> [ -116.7280, 8.4347 ]
NAGA3 optimal f(4.0894, 0.6984, 1.0000, 0.0000, 1.0000, 0.0001) ==> [ -126.8323, 19.2108 ]
NAGA3 optimal f(1.0039, 0.9968, 1.0000, 0.0000, 1.0000, 0.0000) ==> [ -41.8120, 4.0014 ]
NAGA3 optimal f(4.5874, 0.8625, 1.0002, 0.0000, 1.0000, 0.0002) ==> [ -184.6598, 23.7886 ]
NAGA3 optimal f(4.7426, 0.9147, 1.0000, 0.0000, 1.0000, 0.0001) ==> [ -205.2241, 25.3289 ]
NAGA3 optimal f(5.0000, 1.0000, 2.2868, 0.0000, 1.0000, 0.0007) ==> [ -243.6527, 32.2292 ]
NAGA3 optimal f(4.9595, 0.9870, 1.0015, 0.0000, 1.0000, 0.0000) ==> [ -235.9869, 27.5735 ]
NAGA3 optimal f(0.0000, 2.0000, 2.9274, 0.0000, 1.0000, 0.0000) ==> [ -119.7149, 13.5697 ]
NAGA3 optimal f(5.0000, 1.0000, 3.8014, 0.0000, 1.0000, 0.0009) ==> [ -249.8431, 41.4506 ]
'''
