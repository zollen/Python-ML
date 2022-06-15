'''
Created on Jun. 14, 2022

@author: zollen
@url: https://towardsdatascience.com/pymoode-differential-evolution-in-python-78e4221e5cbe
'''

import sys
import numpy as np
from pymoo.core.problem import Problem
from pymoode.gde3 import GDE3
from pymoode.survivors import RankSurvival
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.factory import get_reference_directions

np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)


def printme(x):
    print(x, " {:.4f}, {:.4f}".format(objective1(x), objective2(x)))

def objective1(x):
    if x.ndim > 1:
        return (x[:, 0] - 0.5) ** 2 + 0.7 * (x[:, 0] * x[:, 1]) + 1.2 * (x[:, 1] ) ** 2 + 0.5 * (x[:, 1] * x[:, 2])
    else:
        return (x[0] - 0.5) ** 2 + 0.7 * (x[0] * x[1]) + 1.2 * (x[1] ) ** 2 + 0.5 * (x[1] * x[2])

def objective2(x):
    if x.ndim > 1:
        return 1/3 * (x[:, 0] * x[:, 1]) + 1/3 * (x[:, 0] * x[:, 2]) + 1/3 * (x[:, 1] * x[:, 2])
    else:
        return 1/3 * (x[0] * x[1]) + 1/3 * (x[0] * x[2]) + 1/3 * (x[1] * x[2])

def constaint1(x):
    return x[:, 0] ** 2 + (x[:, 1] - 1) ** 2 + x[:, 2] ** 2 - 5

def constaint2(x):
    return -1 * (x[:, 0] + 0.5) ** 2 - (x[:, 1] - 1) ** 2 - (x[:, 2] - 0.5) ** 2 + 2 
     
class DTLZXProblem(Problem):
    def __init__(self):
        xl = np.full(3, -5.0)
        xu = np.full(3, 5.0)
        super().__init__(
            n_var=3, n_obj=2, n_constr=2,  xl=xl, xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
        F1 = objective1(x)
        F2 = objective2(x)
        out["F"] = np.column_stack([F1, F2])
        G1 = constaint1(x)
        G2 = constaint2(x)
        out["G"] = np.column_stack([G1, G2])
        
      
        
        

gde3 = GDE3(pop_size=500, variant="DE/rand/1/bin", F=(0.0, 1.0), CR=0.5, 
            survival=RankSurvival(crowding_func="cd"))

ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=30)
popsize = ref_dirs.shape[0] + ref_dirs.shape[0] % 4
nsga3 = NSGA3(ref_dirs, pop_size=popsize)


res = minimize(
    DTLZXProblem(),
    gde3,
    ('n_gen', 250),
    seed=1,
    save_history=True,
    verbose=False)

print("========== GDE3 ======= Obj#1 = Obj#2 ===")
np.apply_along_axis(printme, 1, res.X)


res = minimize(
    DTLZXProblem(),
    nsga3,
    ('n_gen', 250),
    seed=1,
    save_history=True,
    verbose=False)

print("========== NAGA3 ====== Obj#1 = Obj#2 ===")
np.apply_along_axis(printme, 1, res.X)