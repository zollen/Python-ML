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

samples = np.array([[1, 3, 4],
                    [1, 2, 1]])

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

# 1. Create the variables we want to optimize
X = h.addIntegral(lb = 0, ub = MAX_VALUE)
Y = h.addIntegral(lb = 0, ub = MAX_VALUE)
x = []
y = []
r = []
for i in ITEMS_INDEX:
    x.append(h.addIntegral(0, MAX_VALUE, i, f'x{i}'))
    y.append(h.addIntegral(0, MAX_VALUE, i, f'y{i}'))
    r.append(h.addIntegral(0, MAX_VALUE, i, f'r{i}'))

b0 = []
b1 = []
b2 = []
b3 = []
k = 0
for i in ITEMS_COMBO:
    b0.append(h.addIntegral(0, 1, k, f'b{i[0], i[1]} 0'))
    b1.append(h.addIntegral(0, 1, k, f'b{i[0], i[1]} 1'))
    b2.append(h.addIntegral(0, 1, k, f'b{i[0], i[1]} 2'))
    b3.append(h.addIntegral(0, 1, k, f'b{i[0], i[1]} 3'))
    k += 1
    
   
    
# 2. Add constraints for each resource
for i in ITEMS_INDEX:
    h.addConstr(X >= x[i] + (r[i] * data[i, 1]) + ((1 - r[i]) * data[i, 2]))
    h.addConstr(Y >= y[i] + (r[i] * data[i, 2]) + ((1 - r[i]) * data[i, 1]))

k = 0
for ind in ITEMS_COMBO:
    i = ind[0]
    j = ind[1]
    h.addConstr(x[i] + (r[i] * data[i, 1]) + ((1 - r[i]) * data[i, 2]) <= x[j] + M * (1 - b0[k]))
    h.addConstr(x[j] + (r[j] * data[j, 1]) + ((1 - r[j]) * data[j, 2]) <= x[i] + M * (1 - b1[k]))
    h.addConstr(y[i] + (r[i] * data[i, 2]) + ((1 - r[i]) * data[i, 1]) <= y[j] + M * (1 - b2[k]))
    h.addConstr(y[j] + (r[j] * data[j, 2]) + ((1 - r[j]) * data[j, 1]) <= y[i] + M * (1 - b3[k]))
    h.addConstr(b0[k] + b1[k] + b2[k] + b3[k] >= 1)
    k += 1



# 3. Minimize the objective function  
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
print('Optimal objective = ', info.objective_function_value)

best_X = solution.col_value[0]
best_Y = solution.col_value[1]
best_x = []
best_y = []
best_r = []
for i in ITEMS_INDEX:
    best_x.append(solution.col_value[i+2])
    best_y.append(solution.col_value[i+TOTAL_ITEMS+2])
    best_r.append(solution.col_value[i+TOTAL_ITEMS+TOTAL_ITEMS+2])

print(f'Best area: {best_X * best_Y}')
print("===============================")
for i in ITEMS_INDEX:
    print(f'x: {best_x[i]} y: {best_y[i]}, r: {best_r[i]}')
