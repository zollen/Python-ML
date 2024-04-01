'''
Created on Jun. 5, 2022

@author: zollen
@url: https://pymoo.org/problems/definition.html#nb-problem-definition-elementwise
@url: https://towardsdatascience.com/pymoode-differential-evolution-in-python-78e4221e5cbe
@desc: Differential evolution (DE) (Storn & Price, 1997) was originally designed for scalar objective 
    optimization. However, because of its simple implementation and efficient problem-solving quality, 
    DE has been modified in different ways to solve multi-objective optimization problems.
    
NSDE: Non-dominated Sorting Differential Evolution, a multi-objective algorithm that combines 
    DE mutation and crossover operators to NSGA-II (Deb et al., 2002) survival.
GDE3: Generalized Differential Evolution 3, a multi-objective algorithm that combines DE mutation and 
    crossover operators to NSGA-II survival with a hybrid type survival strategy. In this algorithm, 
    individuals might be removed in a one-to-one comparison before truncating the population by the 
    multi-objective survival operator. It was proposed by Kukkonen, S. & Lampinen, J. (2005).
NSDE-R: Non-dominated Sorting Differential Evolution based on Reference directions 
    (Reddy & Dulikravich, 2019). It is an algorithm for many-objective problems that works as an 
    extension of NSDE using NSGA-III (Deb & Jain, 2014) survival strategy.
'''

import numpy as np
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
#Termination
from pymoo.termination.default import DefaultMultiObjectiveTermination

'''
In this example, I will introduce two conflicting convex objectives with additional difficulty 
    introduced by constraints. The problem will be defined over two decision variables x1 and x2, 
    both bounded by -5.0 and 5.0.

f1(x) = (x1 - 0.5)^2 + 0.7 * x1 * x2 + 1.2(x2 + 0.7)^2
f2(x) = 1.1 * (x1 + 1.5)^2 + 0.8 * x1 * x2 + 1.3(x2 - 1.7)^2

g1(x) = x1^2 + (x2 - 1)^2 - 9 <= 0
g2(x) = (x1 + 0.5)^2 + (x2 - 1)^2 + 2 >= 0
'''

class ProblemF2(Problem):
    def __init__(self):
        xl = np.full(2, -5.0)
        xu = np.full(2, 5.0)
        super().__init__(n_var=2, n_obj=2, n_constr=2, xl=xl, xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
        # x is an array corresponding to the decision variables in a Population with shape (N, m) 
        # N - population size, m - number of decision variables 
        F1 = (x[:, 0] - 0.5) ** 2 + 0.7 * x[:, 0] * x[:, 1] + 1.2 * (x[:, 1] + 0.7) ** 2
        F2 = 1.1 * (x[:, 0] + 1.5) ** 2 + 0.8 * x[:, 0] * x[:, 1] + 1.3 * (x[:, 1] - 1.7) ** 2
        out["F"] = np.column_stack([F1, F2])
        G1 = x[:, 0] ** 2 + (x[:, 1] - 1) ** 2 - 9
        G2 = - (x[:, 0] + 0.5) ** 2 - (x[:, 1] - 1) ** 2 + 2
        out["G"] = np.column_stack([G1, G2])
        


termination_multi = DefaultMultiObjectiveTermination(
    xtol=1e-8,
    cvtol=1e-8,
    ftol=1e-8,
    period=50,
    n_max_gen=1000,
    n_max_evals=100000)

ref_dirs = get_reference_directions(
    "das-dennis", 2, n_partitions=15)
#Suggestion for NSGA-III
popsize = ref_dirs.shape[0] + ref_dirs.shape[0] % 4
nsga3 = NSGA3(ref_dirs=ref_dirs, pop_size=popsize)
unaga3 = UNSGA3(ref_dirs=ref_dirs, pop_size=popsize)
agemoea = AGEMOEA(pop_size=popsize)
smsmoea = SMSEMOA(pop_size=popsize)

res = minimize(
    ProblemF2(),
    nsga3,
    termination_multi,
    seed=12,
    save_history=True,
    verbose=False)

print("============== NSGA3 ============== ")
print(res.X)
print(res.F)

res = minimize(
    ProblemF2(),
    unaga3,
    termination_multi,
    seed=12,
    save_history=True,
    verbose=False)

print("============== UNSGA3 ============== ")
print(res.X)
print(res.F)


res_nsga2 = minimize(
    ProblemF2(),
    agemoea,
    termination_multi,
    seed=12,
    save_history=True,
    verbose=False)

print("============== AGEMOEA ============== ")
print(res.X)
print(res.F)


res = minimize(
    ProblemF2(),
    smsmoea,
    termination_multi,
    seed=12,
    save_history=True,
    verbose=False)

print("============== SMSMOEA ============== ")
print(res.X)
print(res.F)


