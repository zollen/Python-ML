'''
Created on Jul 29, 2024

@author: STEPHEN
@url: https://coin-or.github.io/pulp/CaseStudies/a_transportation_problem.html
'''

import itertools  
import numpy as np
import  time as tm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle    
from pulp import *


samples = np.array([[4, 3, 4], 
                    [10, 1, 1],
                    [8, 1, 2],
                    [5, 2, 3],
                    [2, 4, 2],
                    [3, 2, 6],
                    [2, 4, 10]])

samples = np.array([[4, 3, 4], 
                    [2, 4, 2]])



data = np.zeros((np.sum(samples[:,0]), 3), dtype='int32')
i = 0
for q, h, w in samples:
    for _ in range(q):
        data[i,0] = i
        data[i,1] = h
        data[i,2] = w
        i += 1



M=999999
TOTAL_MAX_VALUE=30
MAX_VALUE = 18
TOTAL_ITEMS = data.shape[0]
ITEMS_INDEX = list(range(TOTAL_ITEMS))
ITEMS_COMBO = list(itertools.combinations(ITEMS_INDEX, 2))

'''
print("All Solvers:", listSolvers())
All Solvers: ['GLPK_CMD', 'PYGLPK', 'CPLEX_CMD', 'CPLEX_PY', 'GUROBI', 'GUROBI_CMD', 
                'MOSEK', 'XPRESS', 'XPRESS', 'XPRESS_PY', 'PULP_CBC_CMD', 'COIN_CMD', 
                'COINMP_DLL', 'CHOCO_CMD', 'MIPCL_CMD', 'SCIP_CMD', 'FSCIP_CMD', 'SCIP_PY', 
                'HiGHS', 'HiGHS_CMD', 'COPT', 'COPT_DLL', 'COPT_CMD']
'''

model = pulp.LpProblem('Asset Storage Problem', LpMinimize)

# get solver
solver = pulp.PULP_CBC_CMD(msg=True, warmStart=True)  #  options=["threads=4"]
   



# declare decision variables
X = LpVariable(name='X', lowBound = 0, upBound = MAX_VALUE, cat='Integer')
Y = LpVariable(name='Y', lowBound = 0, upBound = MAX_VALUE, cat='Integer')
x = [ LpVariable(name=f'x{i}', lowBound = 0, upBound = MAX_VALUE, cat='Integer') for i in ITEMS_INDEX ]
y = [ LpVariable(name=f'y{i}', lowBound = 0, upBound = MAX_VALUE, cat='Integer') for i in ITEMS_INDEX ]
r = [ LpVariable(name=f'r{i}', lowBound = 0, upBound = 1, cat='Integer') for i in ITEMS_INDEX ]

b0 = [ LpVariable(name = f'b{i[0], i[1]} 0', lowBound = 0, upBound = 1, cat='Integer') for i in ITEMS_COMBO ]
b1 = [ LpVariable(name = f'b{i[0], i[1]} 1', lowBound = 0, upBound = 1, cat='Integer') for i in ITEMS_COMBO ]
b2 = [ LpVariable(name = f'b{i[0], i[1]} 2', lowBound = 0, upBound = 1, cat='Integer') for i in ITEMS_COMBO ]
b3 = [ LpVariable(name = f'b{i[0], i[1]} 3', lowBound = 0, upBound = 1, cat='Integer') for i in ITEMS_COMBO ]




# declare objective
for i in ITEMS_INDEX:
    model += X >= x[i] + (r[i] * data[i, 1]) + ((1 - r[i]) * data[i, 2])
    model += Y >= y[i] + (r[i] * data[i, 2]) + ((1 - r[i]) * data[i, 1])

k = 0
for ind in ITEMS_COMBO:
    i = ind[0]
    j = ind[1]
    model += x[i] + (r[i] * data[i, 1]) + ((1 - r[i]) * data[i, 2]) <= x[j] + M * (1 - b0[k])
    model += x[j] + (r[j] * data[j, 1]) + ((1 - r[j]) * data[j, 2]) <= x[i] + M * (1 - b1[k])
    model += y[i] + (r[i] * data[i, 2]) + ((1 - r[i]) * data[i, 1]) <= y[j] + M * (1 - b2[k])
    model += y[j] + (r[j] * data[j, 2]) + ((1 - r[j]) * data[j, 1]) <= y[i] + M * (1 - b3[k])
    model += b0[k] + b1[k] + b2[k] + b3[k] >= 1
    k += 1



# set objective function
model += X + Y, "Minimize Storage Space"




# solve 
start_time = tm.time()
results = model.solve(solver=solver)
end_time = tm.time()
execution_time = end_time - start_time





# print results
if LpStatus[results] == 'Optimal': print('The solution is optimal.')
print(f'Objective value: z* = {value(model.objective)}')
print(f'Execution Time = {execution_time:0.2f} seconds')
for i in ITEMS_INDEX:
        print(f'x: {value(x[i])} y: {value(y[i])}, r: {value(r[i])}')
print("=================================================")
print(f'X: {value(X)} * Y: {value(Y)} = {value(X) * value(Y)}')

total_X, total_Y = value(X), value(Y)
num = TOTAL_ITEMS

_, ax = plt.subplots()

for i in ITEMS_INDEX:
    coords = (value(x[i]), value(y[i]))
    if value(r[i]) == 1:
        wid = data[i, 1]    
        hig = data[i, 2]
    else:
        wid = data[i, 2]
        hig = data[i, 1]
    
    ax.add_patch(Rectangle(coords, wid, hig,
              edgecolor = 'black',
              facecolor = "Grey",
              fill = True,
              alpha = 0.5,
              lw=2))
    
ax. set_xlim(0, total_X )
ax. set_ylim(0, total_Y )

ax.set_xticks(range(int(total_X)+1))
ax.set_yticks(range(int(total_Y)+1))
ax.grid()
ax.set_title(f" Total area {total_X:0.0f} x {total_Y:0.0f} = {total_X * total_Y:0.0f}")

plt.show()
