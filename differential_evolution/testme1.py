'''
Created on Jun. 4, 2022

@author: zollen
@url: https://pymoo.org/problems/definition.html#nb-problem-definition-elementwise
@url: https://towardsdatascience.com/pymoode-differential-evolution-in-python-78e4221e5cbe
@desc: Differential evolution (DE) (Storn & Price, 1997) was originally designed for scalar objective 
    optimization. However, because of its simple implementation and efficient problem-solving quality, 
    DE has been modified in different ways to solve multi-objective optimization problems.
    
DE: Differential Evolution for single-objective problems proposed by Storn & Price (1997). 
    Other features later implemented are also present, such as dither, jitter, selection variants, 
    and crossover strategies. For details see Price et al. (2005).
'''

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination.default import DefaultSingleObjectiveTermination

#Defining the objective function
def rastrigin(x):
    return np.sum(x * x - 10 * np.cos(2 * np.pi * x)) + 10 * np.size(x)

'''
f(x) = A * n + Σ ( x(i) - A * cos(2 * π * x(i) )

    A is a user-specified parameter (usually 10) and 
    n is the number of decision variables used. In our implementation, 
    we will use two decision variables. Both will be bounded by -5.12 and 5.12.
'''

# Object-oriented definition which implements a function evaluating a single solution at a time.        
class ElementwiseF1(ElementwiseProblem):
    def __init__(self):
        xl = np.full(5, -5.12)
        xu = np.full(5, 5.12)
        super().__init__(
            n_var=5, n_obj=1, n_constr=0,
            xl=xl, xu=xu)
    def _evaluate(self, x, out, *args, **kwargs):
        # x is a single solution
        out["F"] = rastrigin(x)
'''
 Therefore, I have used 30 as the population size N to ensure global convergence. The DE variant was 
 selected as the most usual DE/rand/1/bin, the F parameter as (0.3, 1.0) which uses random uniform 
 distribution dither, and the CR parameter as 0.5, to reinforce the search on the coordinate axis and 
 control convergence.
 
* pop_size (int, optional): Population size.
* sampling (Sampling, Population, or array-like, optional): Sampling strategy of pymoo.
* variant (str, optional): Differential evolution strategy. Must be a string in the format: 
    “DE/selection/n/crossover”. Defaults to “DE/rand/1/bin”.
* CR (float, optional): Crossover parameter. Defined in the range [0, 1]. 
    To reinforce mutation, use higher values. To control convergence speed, use lower values.
* F (iterable of float or float, optional): Scale factor or mutation parameter. 
    Defined in the range (0, 2]. To reinforce exploration, use higher values; 
    whereas for exploitation, use lower.
* gamma (float, optional): Jitter deviation parameter. Should be in the range (0, 2). Defaults to 1e-4.
* pm (Mutation, optional): Pymoo’s mutation operators after crossover. Defaults to None.
* repair (callable or str): Repair strategy to mutant vectors outside problem boundaries. 
    The strategies adopted are based on Price et al. (2005). Defaults to “bounce-back”.
* survival (Survival, optional): Pymoo’s survival strategy. Should be considered in multi-objective algorithms.
'''
de = DE(pop_size=100, sampling=LHS(), variant="DE/rand/1/bin", CR=0.5)

termination_1 = DefaultSingleObjectiveTermination(
    xtol=1e-6,
    cvtol=0.0,
    ftol=1e-6,
    period=20,
    n_max_gen=100,
    n_max_evals=100000)


res = minimize(
    ElementwiseF1(),
    de,
    termination=termination_1,
    seed=12,
    save_history=True,
    verbose=True)


print("BEST: ", res.X)
print("VAL: ", res.F)