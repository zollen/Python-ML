'''
Created on Apr. 9, 2022

@author: zollen
@url: https://towardsdatascience.com/introduction-to-linear-programming-in-python-9261e7eb44b
@desc: Imagine you are a strategist recruiting an army. You have:

    Three resources: 🌾food, 🪵wood, and 🪙gold
    Three units: 🗡️swordsmen, 🏹bowmen, and 🐎horsemen.
    
    Horsemen are stronger than bowmen, who are in turn stronger than swordsmen. 
    The following table provides the cost and power of each unit: 
    
    Unit          |    Food    |    Wood  |  Gold  |  Power
   ----------------------------------------------------
    Swordman      |     60     |    20    |   0    |   70
    Bowman        |     80     |    10    |   40   |   95
    Horseman      |    140     |     0    |  100   |   230
    
    Now we have 1200 🌾food, 800 🪵wood, and 600 🪙gold. How should we maximize the 
    power of our army considering these resources?
'''

# Import OR-Tools wrapper for linear programming
from ortools.linear_solver import pywraplp

# Create a solver using the GLOP backend
solver = pywraplp.Solver('Maximize army power', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

# Create the variables we want to optimize
swordsmen = solver.IntVar(0, solver.infinity(), 'swordsmen')
bowmen = solver.IntVar(0, solver.infinity(), 'bowmen')
horsemen = solver.IntVar(0, solver.infinity(), 'horsemen')

'''
Constaints
In our case, we have a limited number of resources we can use to produce units. In other words, 
we can’t spend more resources than we have: for instance, the 🌾food spent to recruit units 
cannot be higher than 1200. The same is true with 🪵wood (800) and 🪙gold (600).

1 swordsman = 🌾60 + 🪵20;
1 bowman = 🌾80 + 🪵10 + 🪙40;
1 horseman = 🌾140 + 🪙100.

We can write one constraint per resource as follows:
 60 * swordsman + 80 * bowman + 140 * horseman <= 1200
 20 * swordsman + 10 * bowman                  <= 800
                  40 * bowman + 100 * horseman <= 600
'''
# Add constraints for each resource
solver.Add(swordsmen*60 + bowmen*80 + horsemen*140 <= 1200) # Food
solver.Add(swordsmen*20 + bowmen*10 <= 800) # Wood
solver.Add(bowmen*40 + horsemen*100 <= 600) # Gold

'''
Objective
In linear programming, this function has to be linear (like the constraints), 
so of the form ax + by + cz + d. In our example, the objective is quite clear: 
we want to recruit the army with the highest power. The table gives us the following power values:

1 swordsman = 💪70;
1 bowman = 💪95;
1 horseman = 💪230.

Maximizing the power of the army amounts to maximizing the sum of the power of each unit. 
Our objective function can be written as:

max (70 x swordsman + 95 x bowman + 230 * horseman)
'''

solver.Maximize(swordsmen*70 + bowmen*95 + horsemen*230)

'''
 There are five steps to model any linear optimization problem:
    1. Choosing a solver
    2. Declaring the variables to optimize with lower and upper bounds;
    3. Adding constraints to these variables;
    4. Defining the objective function to maximize or to minimize.
    5. Optimizing
    
let's optimize!
'''

status = solver.Solve()

# If an optimal solution has been found, print results
if status == pywraplp.Solver.OPTIMAL:
    print('================= Solution =================')
    print(f'Solved in {solver.wall_time():.2f} milliseconds in {solver.iterations()} iterations')
    print()
    print(f'Optimal power = {solver.Objective().Value()} 💪power')
    print('Army:')
    print(f' - 🗡️Swordsmen = {swordsmen.solution_value()}')
    print(f' - 🏹Bowmen = {bowmen.solution_value()}')
    print(f' - 🐎Horsemen = {horsemen.solution_value()}')
else:
    print('The solver could not find an optimal solution.')

'''
1. The solver decided to take the maximum number of 🐎horsemen (6, since we only have 🪙600 and they 
    each cost 🪙100);
2. The remaining resources are spent in 🗡️swordsmen: we have 1200 – 6*140 = 360🌾food left, which is 
    why the solver chose 6 🗡️swordsmen
3. We can deduce that the horsemen are the best unit and the bowmen are the worst one because they 
    haven’t been chosen at all.
'''

'''
But the expected answer should be integer, why not?

GLOP is a pure linear programming solver. This means that it cannot understand concepts like integers. 
It is limited to continuous parameters with a linear relationship.

This is the difference between linear programming (LP) and integer linear programming (ILP). 
In summary, LP solvers can only use real numbers and not integers as variables. So why did we 
declare our variables as integers if it doesn’t take it into account?

GLOP cannot solve ILP problems, but other solvers can. Actually, a lot of them are mixed 
integer linear programming (MILP, commonly called MIP) solvers. This means that they can 
consider both continuous (real numbers) and discrete (integers) variables. A particular case 
of discrete values is Boolean variables to represent decisions with 0–1 values.
'''
    
# Create the linear solver using the CBC backend
solver = pywraplp.Solver('Maximize army power', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

# 1. Create the variables we want to optimize
swordsmen = solver.IntVar(0, solver.infinity(), 'swordsmen')
bowmen = solver.IntVar(0, solver.infinity(), 'bowmen')
horsemen = solver.IntVar(0, solver.infinity(), 'horsemen')

# 2. Add constraints for each resource
solver.Add(swordsmen*60 + bowmen*80 + horsemen*140 <= 1200)
solver.Add(swordsmen*20 + bowmen*10 <= 800)
solver.Add(bowmen*40 + horsemen*100 <= 600)

# 3. Maximize the objective function
solver.Maximize(swordsmen*70 + bowmen*95 + horsemen*230)

# Solve problem
status = solver.Solve()

# If an optimal solution has been found, print results
if status == pywraplp.Solver.OPTIMAL:
    print('================= Solution =================')
    print(f'Solved in {solver.wall_time():.2f} milliseconds in {solver.iterations()} iterations')
    print()
    print(f'Optimal value = {solver.Objective().Value()} 💪power')
    print('Army:')
    print(f' - 🗡️Swordsmen = {swordsmen.solution_value()}')
    print(f' - 🏹Bowmen = {bowmen.solution_value()}')
    print(f' - 🐎Horsemen = {horsemen.solution_value()}')
else:
    print('The solver could not find an optimal solution.')
  
  
'''
In general, we just round up these values since the error is insignificant, but it is important 
to remember to choose the appropriate solver according to the studied problem: LP (continuous variables);
MIP/MILP (combination of continuous and discrete variables).
'''
