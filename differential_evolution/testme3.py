'''
Created on Jun. 6, 2022

@author: zollen
@url: https://pymoo.org/problems/definition.html#nb-problem-definition-elementwise
@url: https://towardsdatascience.com/pymoode-differential-evolution-in-python-78e4221e5cbe
@desc: As GDE3 and NSDE have been originally designed using NSGA-II Rank and Crowding survival, 
    they perform poorly in many-objective problems. However, simple modifications from a user 
    perspective can solve this issue.
    
NSDE-R: It combines the survival operator of NSGA-III with the reproduction operators of DE, which leads 
    to great performance in many-objective problems. GDE3-MNN is a variant of GDE3 proposed by Kukkonen 
    and Deb (2006a) that replaces original crowding distances of Rank and Crowding survival with an 
    M-Nearest Neighbors based crowding metric with recursive elimination and re-calculation. 
    It has improved a lot the performance of GDE3 in many-objective problems.
'''
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.problems  import get_problem

'''
DTLZ2 sample problem
====================

f1(x) = cos( π / (2x1) ) * cos ( π / (2x2) ) ( 1 + g(x) )
f2(x) = cos( π / (2x1) ) * sin ( π / (2x2) ) ( 1 + g(x) )
f3(x) = sin( π / (2x1) ) ( 1 + g(x) )

g(x) = Σ(i=1-3) ( xi - 0.5)^2

0 <= xi <= 1

No explicit constaint
'''

SEED=5

'''
Notice reference directions are a mandatory argument for NSDE-R and NSGA-III. Fortunately, 
    pymoo has an interface to easily obtain usual directions. Moreover, notice, that in GDE3 the 
    RankSurvival operator is being used with crowding_func = "mnn", which uses the M-NN 
    strategy aforementioned.
'''
#Define the reference directions
ref_dirs = get_reference_directions(
    "das-dennis", 3, n_partitions=15)
#Suggestion for NSGA-III
popsize = ref_dirs.shape[0] + ref_dirs.shape[0] % 4
print("POPULATION SIZE: ", popsize)
nsga3 = NSGA3(ref_dirs=ref_dirs, pop_size=popsize)
unaga3 = UNSGA3(ref_dirs=ref_dirs, pop_size=popsize)
agemoea = AGEMOEA(pop_size=popsize)
smsmoea = SMSEMOA(pop_size=popsize)


'''
I will adopt a simplified termination criterion (‘n_gen’, 250), based only on the number 
     of generations, which is usual for multi-objective problems.
     
NSGA-III has outperformed the DE algorithms in this problem, although the performances of DE 
    algorithms were great as well, especially NSDE-R.
'''

res = minimize(
    get_problem("dtlz2"),
    algorithm=nsga3,
    termination=('n_gen', 200),
    seed=SEED,
    save_history=True,
    verbose=False)

print("============== NSGA3-MNN ============== ")
print("NSGA3: Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

res = minimize(
    get_problem("dtlz2"),
    algorithm=unaga3,
    termination=('n_gen', 200),
    seed=SEED,
    save_history=True,
    verbose=False)

print("============== UNSGA3 ============== ")
print("UNSGA3: Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

res = minimize(
    get_problem("dtlz2"),
    algorithm=agemoea,
    termination=('n_gen', 200),
    seed=SEED,
    save_history=True,
    verbose=False)

print("============== AGEMOEA ============== ")
print("AGEMOEA: Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

res = minimize(
    get_problem("dtlz2"),
    algorithm=smsmoea,
    termination=('n_gen', 200),
    seed=SEED,
    save_history=True,
    verbose=False)

print("============== SMSMOEA ============== ")
print("SMSMOEA: Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
