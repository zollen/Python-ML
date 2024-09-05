'''
Created on Sep 4, 2024

@author: STEPHEN
@url: https://towardsdatascience.com/linear-programming-the-stock-cutting-problem-dc6ba3bf3de1
'''


from ortools.sat.python import cp_model
import numpy as np
import itertools  
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle     

  
samples = np.array([[1, 6, 1], 
                    [1, 6, 1],
                    [1, 5, 1],
                    [1, 4, 1],
                    [1, 3, 1],
                    [1, 3, 1]])



data = np.zeros((np.sum(samples[:,0]), 3), dtype='int32')
i = 0
for q, h, w in samples:
    for _ in range(q):
        data[i,0] = i
        data[i,1] = h
        data[i,2] = w
        i += 1



M=999999
MAX_WOODEN_LENGTH = 9
TOTAL_MAX_VALUE = 30
MAX_VALUE = 9
TOTAL_ITEMS = data.shape[0]
ITEMS_INDEX = list(range(TOTAL_ITEMS))
ITEMS_COMBO = list(itertools.combinations(ITEMS_INDEX, 2))

# Create the linear solver using integer only optimizer. It is the fastest
model = cp_model.CpModel()


# 1. Create the variables we want to optimize
Y = model.new_int_var(0, TOTAL_MAX_VALUE, 'Y')
x = [ model.new_int_var(0, MAX_VALUE, f'x{i}') for i in ITEMS_INDEX ]
y = [ model.new_int_var(0, MAX_VALUE, f'y{i}') for i in ITEMS_INDEX ]

b0 = [ model.new_int_var(0, 1, f'b{i[0], i[1]} 0') for i in ITEMS_COMBO ]
b1 = [ model.new_int_var(0, 1, f'b{i[0], i[1]} 1') for i in ITEMS_COMBO ]
b2 = [ model.new_int_var(0, 1, f'b{i[0], i[1]} 2') for i in ITEMS_COMBO ]
b3 = [ model.new_int_var(0, 1, f'b{i[0], i[1]} 3') for i in ITEMS_COMBO ]

total_variables = 2 + len(x) + len(y) + len(b0) + len(b1) + len(b2) + len(b3)



# 2. Add constraints for each resource
for i in ITEMS_INDEX:
    model.add(MAX_WOODEN_LENGTH >= x[i] + data[i, 1]) 
    model.add(Y >= y[i] + data[i, 2])

k = 0
for ind in ITEMS_COMBO:
    i = ind[0]
    j = ind[1]
    model.add(x[i] + data[i, 1] <= x[j] + M * (1 - b0[k]))
    model.add(x[j] + data[j, 1] <= x[i] + M * (1 - b1[k]))
    model.add(y[i] + data[i, 2] <= y[j] + M * (1 - b2[k]))
    model.add(y[j] + data[j, 2] <= y[i] + M * (1 - b3[k]))
    model.add(b0[k] + b1[k] + b2[k] + b3[k] >= 1)
    k += 1
    
total_constraints = TOTAL_ITEMS * 2 + len(ITEMS_COMBO) * 5



# 3. Minimize the objective function  
model.Minimize(Y)


# Solve problem
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 240.0   # exit 240 seconds after stuck
solver.parameters.num_search_workers = 4
solver.parameters.log_search_progress = True
status = solver.solve(model)


# If an optimal solution has been found, print results
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print('================= Solution =================')
    print(f'Total Variables: {total_variables},  Total Constraints: {total_constraints}')
    print(f'Solved in {solver.wall_time} seconds')
    print()
    print(f'Optimal value = {solver.objective_value}')
    for i in ITEMS_INDEX:
        print(f'x: {solver.value(x[i])} y: {solver.value(y[i])}')
    print("=================================================")
    print(f'Y: {solver.value(Y)}')
else:
    print('The solver could not find an optimal solution.')
    exit()
    


total_X, total_Y = MAX_WOODEN_LENGTH, solver.value(Y)
num = TOTAL_ITEMS

_, ax = plt.subplots()

for i in ITEMS_INDEX:
    coords = (solver.value(x[i]), solver.value(y[i]))
    wid = data[i, 1]    
    hig = data[i, 2]
    
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

