'''
Created on Jul 23, 2024

@author: STEPHEN
@url: https://towardsdatascience.com/how-to-solve-an-asset-storage-problem-with-mathematical-programming-3b96b7cc22d1
@desc The assortment problem
A general version of the assortment problem involves selecting a subset of items from a larger set to 
maximize a certain objective, often revenue or profit, under constraints such as shelf space or budget. 
It’s a common problem in retail and operations management, involving optimization techniques and often 
consumer choice modeling. The specific assortment problem we are dealing with in this article is also 
known as the 2D rectangle packing problem, which frequently appears in logistics and production contexts.

@data
Quantity      Heights       Widths
4              3              4 
10             1              1
8              1              2
5              2              3
2              4              2
3              2              6
2              4             10


'''

from ortools.sat.python import cp_model
import numpy as np
import itertools  
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle     

  
samples = np.array([[4, 3, 4], 
                    [10, 1, 1],
                    [8, 1, 2],
                    [5, 2, 3],
                    [2, 4, 2],
                    [3, 2, 6],
                    [2, 4, 10]])

'''
samples = np.array([[4, 3, 4], 
                    [2, 4, 2],
                    [3, 2, 6],
                    [2, 4, 10]])
                    
================= Solution =================
Solved in 493.9364454 seconds

Optimal value = 27.0
x: 2 y: 10, r: 0
x: 0 y: 6, r: 0
x: 6 y: 0, r: 0
x: 10 y: 10, r: 0
x: 0 y: 9, r: 0
x: 4 y: 6, r: 0
x: 0 y: 0, r: 0
x: 0 y: 4, r: 0
x: 0 y: 2, r: 0
x: 6 y: 3, r: 1
x: 10 y: 0, r: 1
=================================================
X: 14 * Y: 13 = 182
         
                    
'''


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

# Create the linear solver using integer only optimizer. It is the fastest
model = cp_model.CpModel()


# 1. Create the variables we want to optimize
X = model.new_int_var(0, TOTAL_MAX_VALUE, 'X') 
Y = model.new_int_var(0, TOTAL_MAX_VALUE, 'Y')
x = [ model.new_int_var(0, MAX_VALUE, f'x{i}') for i in ITEMS_INDEX ]
y = [ model.new_int_var(0, MAX_VALUE, f'y{i}') for i in ITEMS_INDEX ]
r = [ model.new_int_var(0, 1, f'r{i}') for i in ITEMS_INDEX ]

b0 = [ model.new_int_var(0, 1, f'b{i[0], i[1]} 0') for i in ITEMS_COMBO ]
b1 = [ model.new_int_var(0, 1, f'b{i[0], i[1]} 1') for i in ITEMS_COMBO ]
b2 = [ model.new_int_var(0, 1, f'b{i[0], i[1]} 2') for i in ITEMS_COMBO ]
b3 = [ model.new_int_var(0, 1, f'b{i[0], i[1]} 3') for i in ITEMS_COMBO ]


# 2. Add constraints for each resource
for i in ITEMS_INDEX:
    model.add(X >= x[i] + (r[i] * data[i, 1]) + ((1 - r[i]) * data[i, 2]))
    model.add(Y >= y[i] + (r[i] * data[i, 2]) + ((1 - r[i]) * data[i, 1]))

k = 0
for ind in ITEMS_COMBO:
    i = ind[0]
    j = ind[1]
    model.add(x[i] + (r[i] * data[i, 1]) + ((1 - r[i]) * data[i, 2]) <= x[j] + M * (1 - b0[k]))
    model.add(x[j] + (r[j] * data[j, 1]) + ((1 - r[j]) * data[j, 2]) <= x[i] + M * (1 - b1[k]))
    model.add(y[i] + (r[i] * data[i, 2]) + ((1 - r[i]) * data[i, 1]) <= y[j] + M * (1 - b2[k]))
    model.add(y[j] + (r[j] * data[j, 2]) + ((1 - r[j]) * data[j, 1]) <= y[i] + M * (1 - b3[k]))
    model.add(b0[k] + b1[k] + b2[k] + b3[k] >= 1)
    k += 1
    

# 3. Minimize the objective function  
model.Minimize(X + Y)


# Solve problem
solver = cp_model.CpSolver()
solver.parameters.num_search_workers = 8
solver.parameters.log_search_progress = True
status = solver.solve(model)


# If an optimal solution has been found, print results
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print('================= Solution =================')
    print(f'Solved in {solver.wall_time} seconds')
    print()
    print(f'Optimal value = {solver.objective_value}')
    for i in ITEMS_INDEX:
        print(f'x: {solver.value(x[i])} y: {solver.value(y[i])}, r: {solver.value(r[i])}')
    print("=================================================")
    print(f'X: {solver.value(X)} * Y: {solver.value(Y)} = {solver.value(X) * solver.value(Y)}')
else:
    print('The solver could not find an optimal solution.')
    exit()
    


total_X, total_Y = solver.value(X), solver.value(Y)
num = TOTAL_ITEMS

_, ax = plt.subplots()

for i in ITEMS_INDEX:
    coords = (solver.value(x[i]), solver.value(y[i]))

    if solver.value(r[i]) == 1:
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
ax.set_title(f" Total area {total_X} x {total_Y} = {total_X * total_Y}")

plt.show()