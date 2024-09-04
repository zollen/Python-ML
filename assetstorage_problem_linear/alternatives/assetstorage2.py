'''
Created on Jul 20, 2024

@author: STEPHEN
@url: https://towardsdatascience.com/how-to-solve-an-asset-storage-problem-with-mathematical-programming-3b96b7cc22d1

@desc This library is good at solving large number of optimized parameters.

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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import chain, combinations, permutations, product
import gurobipy as gp
from gurobipy import GRB
from copy import deepcopy


widths = [4,1,2,3,2,6,10]
heights =[3,1,1,2,4,2,4]
quants = [1,1,1,1,1,1,1]

WIDTHS = []
HEIGHTS = []
for q,w,h in zip(quants,widths,heights):
  for x in range(q):
    HEIGHTS.append(w)
    WIDTHS.append(h)

data_df = pd.DataFrame()
data_df['HEIGHTS'] = HEIGHTS
data_df['WIDTHS'] = WIDTHS

N = len(data_df) # number of items
top = max(data_df['HEIGHTS'].sum(),data_df['WIDTHS'].sum()) # maximum value of X or Y
M = top # to be used as big M for the OR constraints

I = range(N) # for the indexes of each asset "i"
K = range(4) # for the index of the OR variables "b"

model = gp.Model("Assortment")

# (x,y) Coordinate variables
x = model.addVars(I,lb = 0,ub = top,vtype=GRB.CONTINUOUS, name="x")
y = model.addVars(I,lb = 0,ub = top,vtype=GRB.CONTINUOUS, name="y")

# Rotation variables 
R = model.addVars(I,vtype=GRB.BINARY,name = 'R')

X = model.addVar(lb=0,ub = top,vtype = GRB.CONTINUOUS,name = "X")
Y = model.addVar(lb=0,ub = top,vtype = GRB.CONTINUOUS,name = "Y")

# b variables for OR condition
b_vars = [(i,j,k) for i in I for j in I if j!=i for k in K]
B = model.addVars(b_vars,vtype = GRB.BINARY,name = "B")

# Objective function
model.setObjective(X*Y,GRB.MINIMIZE);

# constraints (1) and (2)
for i in I:
    model.addConstr(X >= x[i] + WIDTHS[i]*R[i] + (1-R[i])*HEIGHTS[i])
    model.addConstr(Y >= y[i] + HEIGHTS[i]*R[i] + (1-R[i])*WIDTHS[i])


# Constraints (3-7)
for i in I:
    for j in I:
        if i == j:
            continue
        else:
            #constraint (3)
            model.addConstr(x[i] + WIDTHS[i]*R[i] + (1-R[i])*HEIGHTS[i] <= x[j] + M*(1-B[i,j,0]))
            #constraint (4)
            model.addConstr(x[j] + WIDTHS[j]*R[j] + (1-R[j])*HEIGHTS[j] <= x[i] + M*(1-B[i,j,1]))
            #constraint (5)
            model.addConstr(y[i] + HEIGHTS[i]*R[i] + (1-R[i])*WIDTHS[i] <= y[j] + M*(1-B[i,j,2]))
            #constraint (6)
            model.addConstr(y[j] + HEIGHTS[j]*R[j] + (1-R[j])*WIDTHS[j] <= y[i] + M*(1-B[i,j,3]))
            #constraint (7)
            model.addConstr(B[i,j,0] + B[i,j,1] + B[i,j,2] + B[i,j,3] >= 1)
            
tl = 600
mip_gap = 0.05

model.setParam('TimeLimit', tl)
model.setParam('MIPGap', mip_gap)
model.optimize()


all_vars = model.getVars()
values = model.getAttr("X", all_vars)
names = model.getAttr("VarName", all_vars)

obj = round(model.getObjective().getValue(),0)

total_X = int(round((X.x),0))
total_Y = int(round((Y.x),0))

fig, ax = plt.subplots()

for item in I:

    coords = (x[item].x,y[item].x)

    if R[item].x <= 0.01:
        wid = HEIGHTS[item]
        hig = WIDTHS[item]
    else:
        wid = WIDTHS[item]
        hig = HEIGHTS[item]

    ax.add_patch(Rectangle(coords, wid, hig,
            edgecolor = 'black',
            facecolor = "Grey",
            fill = True,
            alpha = 0.5,
            lw=2))
ax. set_xlim(0, total_X )
ax. set_ylim(0, total_Y )

ax.set_xticks(range(total_X+1))
ax.set_yticks(range(total_Y+1))
ax.grid()
ax.set_title(f" Total area {total_X} x {total_Y} = {int(obj)}")

plt.show()