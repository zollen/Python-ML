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

from pymoode.nsder import NSDER
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoode.gde3 import GDE3
from pymoo.optimize import minimize
from pymoode.survivors import RankSurvival
from pymoo.factory import get_reference_directions, get_problem

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
nsder = NSDER(ref_dirs, pop_size=popsize,
    variant="DE/rand/1/bin", F=(0.0, 1.0), CR=0.5)
gde3mnn = GDE3(pop_size=popsize,
    variant="DE/rand/1/bin", F=(0.0, 1.0), CR=0.2,
    survival=RankSurvival(crowding_func="mnn"))
nsga3 = NSGA3(ref_dirs, pop_size=popsize)

'''
I will adopt a simplified termination criterion (‘n_gen’, 250), based only on the number 
     of generations, which is usual for multi-objective problems.
     
NSGA-III has outperformed the DE algorithms in this problem, although the performances of DE 
    algorithms were great as well, especially NSDE-R.
'''
res_nsder = minimize(
    get_problem("dtlz2"),
    nsder,
    ('n_gen', 250),
    seed=SEED,
    save_history=True,
    verbose=True)

print(res_nsder.X)
