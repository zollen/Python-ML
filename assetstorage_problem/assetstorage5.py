'''
Created on Jul 29, 2024

@author: STEPHEN
@url: https://ergo-code.github.io/HiGHS/dev/interfaces/python/example-py/
'''

import highspy
import itertools  
import numpy as np

samples = np.array([[4, 3, 4], 
                    [10, 1, 1],
                    [8, 1, 2],
                    [5, 2, 3],
                    [2, 4, 2],
                    [3, 2, 6],
                    [2, 4, 10]])

data = np.zeros((np.sum(samples[:,0]), 3), dtype='int32')
i = 0
for q, h, w in samples:
    for _ in range(q):
        data[i,0] = i
        data[i,1] = h
        data[i,2] = w
        i += 1



M=999999
TOTAL_MAX_VALUE=18
MAX_VALUE = 15
TOTAL_ITEMS = data.shape[0]
ITEMS_INDEX = list(range(TOTAL_ITEMS))
ITEMS_COMBO = list(itertools.combinations(ITEMS_INDEX, 2))



h = highspy.Highs()

X = h.addIntegral(lb = 0, ub = 4)
Y = h.addIntegral(lb = 1, ub = 7)



h.addConstr(5 <=   X + 2*Y <= 15)
h.addConstr(6 <= 3*X + 2*Y)

h.minimize(X + Y)

lp = h.getLp()
num_nz = h.getNumNz()
print('LP has ', lp.num_col_, ' columns', lp.num_row_, ' rows and ', num_nz, ' nonzeros')
h.run()
solution = h.getSolution()
basis = h.getBasis()
info = h.getInfo()
model_status = h.getModelStatus()
print('Model status = ', h.modelStatusToString(model_status))
print()
print('Optimal objective = ', info.objective_function_value)
print('Iteration count = ', info.simplex_iteration_count)
print('Primal solution status = ', h.solutionStatusToString(info.primal_solution_status))
print('Dual solution status = ', h.solutionStatusToString(info.dual_solution_status))
print('Basis validity = ', h.basisValidityToString(info.basis_validity))