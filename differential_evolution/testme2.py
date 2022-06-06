'''
Created on Jun. 5, 2022

@author: zollen
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
from pymoode.gde3 import GDE3
from pymoode.nsde import NSDE
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_performance_indicator
#Termination
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

'''
In this example, I will introduce two conflicting convex objectives with additional difficulty 
    introduced by constraints. The problem will be defined over two decision variables x1 and x2, 
    both bounded by -5.0 and 5.0.

f1(x) = (x1 - 0.5)^2 + 0.7 * x1 * x2 + 1.2(x2 + 0.7)^2
f2(x) = 1.1 * (x1 + 1.5)^2 + 0.8 * x1 * x2 + 1.3(x2 - 1.7)^2

g1(x) = x1^2 + (x2 - 1)^2 - 9 <= 0
g2(x) = (x1 + 0.5)^2 + (x2 - 1)^2 + 2 <= 0
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
        


termination_multi = MultiObjectiveDefaultTermination(
    x_tol=1e-8,
    cv_tol=1e-8,
    f_tol=1e-8,
    nth_gen=5,
    n_last=50,
    n_max_gen=200)


gde3 = GDE3(pop_size=50, variant="DE/rand/1/bin", F=(0.0, 1.0), CR=0.7)
nsde = NSDE(pop_size=50, variant="DE/rand/1/bin", F=(0.0, 1.0), CR=0.7)
nsga2 = NSGA2(pop_size=30)

res_dge3 = minimize(
    ProblemF2(),
    gde3,
    termination_multi,
    seed=12,
    save_history=True,
    verbose=False)

print("============== GDE3 ============== ")
print(res_dge3.X)

res_nsde = minimize(
    ProblemF2(),
    nsde,
    termination_multi,
    seed=12,
    save_history=True,
    verbose=False)

print("============== NSDE ============== ")
print(res_nsde.X)


res_nsga2 = minimize(
    ProblemF2(),
    nsga2,
    termination_multi,
    seed=12,
    save_history=True,
    verbose=False)

print("============== NSDE ============== ")
print(res_nsga2.X)


objs_p2 = np.row_stack([res_dge3.F, res_nsde.F, res_nsga2.F])
nadir_p2 = objs_p2.max(axis=0)
reference_p2 = nadir_p2 + 1e-6
ideal_p2 =  objs_p2.min(axis=0)
hv_p2 = get_performance_indicator("hv", ref_point=reference_p2, zero_to_one=True,
                                  nadir=nadir_p2, ideal=ideal_p2)

print("============== PERFORMANCE ==============")
print("hv GDE3", hv_p2.do(res_dge3.F))
print("hv NSDE", hv_p2.do(res_nsde.F))
print("hv NSGA-II", hv_p2.do(res_nsga2.F))