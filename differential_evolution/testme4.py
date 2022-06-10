'''
Created on Jun. 7, 2022

@author: zollen
@desc: Problem

InPipe#A: 15 vol/sec, $50
InPipe#B: 20 vol/sec, $85
InPipe#C: 35 vol/sec, $120

Hub#A: 2 Valves, $60
Hub#B: 5 Valves, $120
Hub#C: 7 Valves, $150

OutPipe#A: 8 vol/sec,  $40
OutPipe#B: 25 vol/sec, $60
OutPipe#C: 30 vol/sec, $65 

Total Inflow <= Total OutFlow
Total pipes <= Total slots of all hubs
Total OutFlow >= 500 vol/sec

Min(cost)
'''

'''
Math
====
x1 - number of InPipe#1
x2 - number of InPipe#2
x3 - number of InPipe#3
x4 - number of OutPipe#1
x5 - number of OutPipe#2
x6 - number of OutPipe#3
x7 - number of Hub#A
x8 - number of Hub#B
x9 - number of Hub#C
IP(a, f) - 15 vol/sec
IP(b, f) - 20 vol/sec
IP(c, f) - 35 vol/sec
OP(a, f) -  8 vol/sec
OP(b, f) - 25 vol/sec
OP(c, f) - 30 vol/sec

Constaints
----------
Total Outflow <= Total Inflow
( x4 * 8 vol/sec + x5 * 25 vol/sec + x6 * 30 vol/sec ) <= ( x1 * 15 vol/sec + x2 * 20 vol/sec + x3 * 35 vol/sec )
    ( x4 * 8 vol/sec + x5 * 25 vol/sec + x6 * 30 vol/sec ) - ( x1 * 15 vol/sec + x2 * 20 vol/sec + x3 * 35 vol/sec ) <= 0

Total pipes <= Total slots of all hubs
x1 + x2 + x3 + x4 + x5 + x6 <= 2 * x7 + 5 * x8 + 7 * x9
    (2 * x7 + 5 * x8 + 7 * x9) - (x1 + x2 + x3 + x4 + x5 + x6) >= 0
    (x1 + x2 + x3 + x4 + x5 + x6) - (2 * x7 + 5 * x8 + 7 * x9) <= 0

Total OutFlow >= 500 vol/sec
x4 * 8 vol/sec + x5 * 25 vol/sec + x6 * 30 vol/sec >= 500
    ( x4 * 8 vol/sec + x5 * 25 vol/sec + x6 * 30 vol/sec ) - 500 >= 0
    500 - ( x4 * 8 vol/sec + x5 * 25 vol/sec + x6 * 30 vol/sec ) <= 0

Objective
---------
Min( $50 * x1 + $85 * x2 + $120 * x3 + $40 * x4 + $60 * x5 +  $65 * x6 + $60 * x7 + $120 * x8 + $150 * x9 ) 

'''

import numpy as np
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

class WaterFlowProblem(Problem):
    def __init__(self):
        xl = np.full(9, 0)
        xu = np.full(9, 20)
        super().__init__(
            n_var=9, n_obj=1, n_constr=3,  xl=xl, xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x[:, 0] * 50 + x[:, 1] * 85 + x[:, 2] * 120 + x[:, 3] * 40 + x[:, 4] * 60 + x[:, 5] * 65 + \
            x[:, 6] * 60 + x[:, 7] * 120 + x[:, 8] * 150
        G1 = 500 - (x[:, 3] * 8 + x[:, 4] * 25 + x[:, 5] * 30)
        G2 = (x[:, 3] * 8 + x[:, 4] * 25 + x[:, 5] * 30) - (x[:, 0] * 15 + x[:, 1] * 20 + x[:, 2] * 35)
        G3 = (x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3] + x[:, 4] + x[:, 5]) - (x[:, 6] * 2 + x[:, 7] * 5 + x[:, 8] * 7) 
        out["G"] = np.column_stack([G1, G2, G3])

# standard generic algorithm
ga = get_algorithm("ga",
                       pop_size=100,
                       sampling=get_sampling("int_random"),
                       crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
                       mutation=get_mutation("int_pm", eta=3.0),
                       eliminate_duplicates=True,
                       )

res = minimize(
    WaterFlowProblem(),
    ga,
    ('n_gen', 200),
    seed=12,
    save_history=True,
    verbose=True)

print("BEST: ", res.X)
print("VAL: ", res.F)