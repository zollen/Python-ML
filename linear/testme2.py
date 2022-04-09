'''
Created on Apr. 9, 2022

@author: zollen
@url: https://towardsdatascience.com/integer-programming-vs-linear-programming-in-python-f1be5bb4e60e
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

'''
One of the best perks of OR-Tools is that it uses a general-purpose programming language like Python. 
Instead of static numbers, we can store our parameters in objects like dictionaries or lists.

The code won’t be as readable, but it becomes much more flexible: actually, it can be so flexible 
that we can solve an entire class of optimization problems without changing the model (just the parameters).
'''

# Import OR-Tools wrapper for linear programming
from ortools.linear_solver import pywraplp

UNITS = ['🗡️Swordsmen', '🏹Bowmen', '🐎Horsemen']

DATA = [[60, 20, 0, 70],
        [80, 10, 40, 95],
        [140, 0, 100, 230]]

RESOURCES = [1200, 800, 600]

def maximize_army(solver, UNITS, DATA, RESOURCES):
 
    # 1. Create the variables we want to optimize
    units = [solver.IntVar(0, solver.infinity(), unit) for unit in UNITS]

    # 2. Add constraints for each resource
    for r, _ in enumerate(RESOURCES):
        solver.Add(sum(DATA[u][r] * units[u] for u, _ in enumerate(units)) <= RESOURCES[r])

    # 3. Maximize the objective function
    solver.Maximize(sum(DATA[u][-1] * units[u] for u, _ in enumerate(units)))

    # Solve problem
    status = solver.Solve()

    # If an optimal solution has been found, print results
    if status == pywraplp.Solver.OPTIMAL:
        print('================= Solution =================')
        print(f'Solved in {solver.wall_time():.2f} milliseconds in {solver.iterations()} iterations')
        print()
        print(f'Optimal value = {solver.Objective().Value()} 💪power')
        print('Army:')
        for u, _ in enumerate(units):
            print(f' - {units[u].name()} = {units[u].solution_value()}')
    else:
        print('The solver could not find an optimal solution.')

maximize_army(pywraplp.Solver('Maximize army power', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING), 
              UNITS, DATA, RESOURCES)

