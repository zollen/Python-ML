'''
Created on Jun. 14, 2022

@author: zollen
@url: https://towardsdatascience.com/pymoode-differential-evolution-in-python-78e4221e5cbe
'''

import sys
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)


def printme(x):
    print(x, " {:.4f}, {:.4f}".format(objective1(x[0], x[1], x[2]), 
                                      objective2(x[0], x[1], x[2])))

def objective1(x, y, z):
    return (x - 0.5) ** 2 + 0.7 * (x * y) + 1.2 * (y ) ** 2 + 0.5 * (y * z)

def objective2(x, y, z):
    return 1/3 * (x * y) + 1/3 * (x * z) + 1/3 * (y * z)

def constaint1(x, y, z):
    return x ** 2 + (y - 1) ** 2 + z ** 2 - 5

def constaint2(x, y, z):
    return -1 * (x + 0.5) ** 2 - (y - 1) ** 2 - (z - 0.5) ** 2 + 2 
     
class DTLZXProblem(Problem):
    def __init__(self):
        xl = np.full(3, -5.0)
        xu = np.full(3, 5.0)
        super().__init__(
            n_var=3, n_obj=2, n_constr=2,  xl=xl, xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
        F1 = objective1(x[:,0], x[:,1], x[:,2])
        F2 = objective2(x[:,0], x[:,1], x[:,2])
        out["F"] = np.column_stack([F1, F2])
        G1 = constaint1(x[:,0], x[:,1], x[:,2])
        G2 = constaint2(x[:,0], x[:,1], x[:,2])
        out["G"] = np.column_stack([G1, G2])
        
      
        
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=30)
nsga3 = NSGA3(ref_dirs, pop_size=500)
unaga3 = UNSGA3(ref_dirs=ref_dirs, pop_size=500)
agemoea = AGEMOEA(pop_size=500)
smsmoea = SMSEMOA(pop_size=500)


res = minimize(
    DTLZXProblem(),
    nsga3,
    ('n_gen', 500),
    seed=1,
    save_history=True,
    verbose=False)

print("========== NSGA3 =========")
np.apply_along_axis(printme, 1, res.X)


res = minimize(
    DTLZXProblem(),
    unaga3,
    ('n_gen', 500),
    seed=1,
    save_history=True,
    verbose=False)

print("========== UNAGA3 =========")
np.apply_along_axis(printme, 1, res.X)



res = minimize(
    DTLZXProblem(),
    agemoea,
    ('n_gen', 500),
    seed=1,
    save_history=True,
    verbose=False)

print("========== AGEMOEA =========")
np.apply_along_axis(printme, 1, res.X)



res = minimize(
    DTLZXProblem(),
    smsmoea,
    ('n_gen', 500),
    seed=1,
    save_history=True,
    verbose=False)

print("========== SMSMOEA =========")
np.apply_along_axis(printme, 1, res.X)

