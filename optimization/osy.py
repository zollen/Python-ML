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
from pymoo.problems import get_problem
from optimization.lib import ParetoFront



problem = get_problem("osy")

def osy6d(X):
    res = problem.evaluate(X)
    return res[0]

def normalize(X):
    minn = np.min(X, axis=0)
    maxx = np.max(X, axis=0)
    return (X - minn) / (maxx - minn)
   
def constraints(X):
    result = 0
    result = np.where(X[:,0] + X[:,1] - 2 >= 0, result + 1, result - 1)
    result = np.where(6 - X[:,0] - X[:,1] >= 0, result + 1, result - 1)
    result = np.where(2 - X[:,1] + X[:,0] >= 0, result + 1, result - 1)
    result = np.where(2 - X[:,0]  + 3 * X[:,1] >= 0, result + 1, result - 1)
    result = np.where(4 - (X[:,2] - 3)**2 - X[:,3] >= 0, result + 1, result - 1)
    result = np.where((X[:,4] - 3)**2 + X[:,5] - 4 >= 0, result + 1, result - 1)
    return np.where(result == 6, 1, -1)

def fitness(X):
    return osy6d(X)          

def data(n):
    return np.random.rand(n, 6) * np.array([[10, 10, 4, 6, 4, 10]]) + \
            np.array([[0, 0, 1, 0, 1, 0]])    
    
def validate(x1, x2, x3, x4, x5, x6):
    return constraints(np.array([[x1, x2, x3, x4, x5, x6]]))

# res = np.sum(stddev((idealp - arr)**2) - stddev((worst - arr)**2), axis=1)


idealp = problem.ideal_point() * -1
worst = problem.nadir_point() * -1

print("IDEAL: ", idealp)
print("WORST: ", worst)

'''
arr = np.array([[ 257.6918, -51.9837 ], 
                [ 257.4334, -51.9639 ], 
                [ 256.4842, -51.9007 ],
                [ 255.1407, -51.8148 ], 
                [ 254.8689, -51.7834 ],
                [ 254.3618, -51.7602 ], 
                [ 253.9392, -51.7314 ], 
                [ 258.3297, -56.1750 ],
                [ 259.0073, -57.1929 ],
                [ 259.3465, -58.7281 ], 
                [ 260.9150, -59.3890 ],
                [ 261.9141, -60.4340 ], 
                [ 262.2444, -61.0180 ], 
                [ 262.2707, -62.2349 ], 
                [ 79.3368, -4.6959   ], 
                [ 78.9540, -4.6934   ], 
                [ 262.7659, -62.9264 ],
                [ 78.6885, -4.6926   ], 
                [ 78.4029, -4.6842   ], 
                [ 78.3505, -4.6664   ], 
                [ 77.8476, -4.6544   ], 
                [ 77.7724, -4.6506   ], 
                [ 77.2556, -4.6426   ], 
                [ 76.8762, -4.6360   ], 
                [ 76.8396, -4.6277   ], 
                [ 76.3221, -4.6173   ], 
                [ 75.9309, -4.6069   ], 
                [ 75.6879, -4.6009   ],
                [ 263.9524, -64.3781 ], 
                [ 259.0301, -55.0689 ],
                [ 75.2595, -4.5920   ], 
                [ 75.1213, -4.5763   ], 
                [ 74.8112, -4.5701   ], 
                [ 74.2132, -4.5654   ],
                [ 74.1166, -4.5572   ], 
                [ 73.8877, -4.5417   ], 
                [ 73.0211, -4.5200   ], 
                [ 72.5452, -4.5183   ], 
                [ 72.2420, -4.5081   ], 
                [ 72.1211, -4.5036   ],
                [ 72.0619, -4.4949   ], 
                [ 71.6180, -4.4843   ], 
                [ 264.1401, -67.2079 ], 
                [ 264.8929, -68.9447 ],
                [ 266.6896, -70.4061 ], 
                [ 267.2317, -70.5713 ], 
                [ 268.4478, -71.7778 ],
                [ 269.4316, -74.0596 ], 
                [ 270.6301, -74.5528 ], 
                [ 270.9788, -75.7957 ], 
                [ 270.7229, -75.7951 ]])



from numpy import std

def stddev(X):
    return np.abs(np.mean(X, axis=0) - X) / std(X, axis=0)
    
res = np.sum(stddev(np.abs(idealp - arr)**3) - stddev(np.abs(worst - arr)**2), axis=1)
ind = np.argsort(res, axis=0)
arr = arr[ind]
res = res[ind]
pts = (1 - normalize(res))**2
prob = pts / np.sum(pts)
for idx in range(arr.shape[0]):
    print("[ {0:.4f}, {1:.4f} ] ==> {2:.8f}, {3:.8f}, {4:.8f}".format(
            arr[idx,0], arr[idx,1], res[idx], pts[idx], prob[idx]))
    
print("TOTAL: ", arr.shape[0])

exit()
'''

agents = ParetoFront(osy6d, data, constraints,  
                     'min', 1000,  
                     ideal_scores = problem.ideal_point(), 
                     nadir_scores = problem.nadir_point(),
                     LB=[0, 0, 1, 0, 1, 0], UB=[6, 6, 5, 6, 5, 6])

best = agents.start(70)

# LB=[4.8, 0.8, 2, 0, 4.8, 0], UB=[5.1, 1.2, 2.2, 0.1, 5.1, 0.1])
#aa = np.array([[5.0000, 1.0000, 2.0173, 0.0000, 5.0000, 0.0002]])
#best = np.vstack([aa, best])

res = agents.modifier(agents.fitness(best))
pts = np.sum(agents.stddev(np.abs(idealp - res)**3) - agents.stddev(np.abs(worst - res)**2), axis=1)
idx = np.argsort(res[:,0] * -1, axis=0)
best = best[idx]
res = res[idx]
pts = pts[idx]
for i in range(best.shape[0]):
    print("ParetoFront optimal<{0:}> f({1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}, {6:.4f}) ==> [ {7:.4f}, {8:.4f} ], [{9:.8f}] [{10:}]".format(
        (i+1), best[i, 0], best[i, 1], best[i, 2], best[i, 3], best[i, 4], best[i, 5], 
        res[i, 0], res[i, 1], pts[i], validate(best[i, 0], best[i, 1], 
                                           best[i, 2], best[i, 3], 
                                           best[i, 4], best[i, 5])))

print("TOTAL: ", best.shape[0])

'''
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

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
    print("NAGA3 optimal f({0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}) ==> [ {6:.4f}, {7:.4f} ], {8:}".format(
        res.X[i, 0], res.X[i, 1], res.X[i, 2], 
        res.X[i, 3], res.X[i, 4], res.X[i, 5], 
        res.F[i, 0], res.F[i, 1], validate(res.X[i, 0], res.X[i, 1], 
                                           res.X[i, 2], res.X[i, 3], 
                                           res.X[i, 4], res.X[i, 5])))

'''
'''
ParetoFront
ParetoFront optimal f(4.9983, 1.0000, 2.1787, 0.0000, 5.0000, 0.0000) ==> [ -259.1332, 55.7298 ], [1]

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
