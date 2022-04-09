'''
Created on Apr. 9, 2022

@author: zollen
@url: https://towardsdatascience.com/integer-programming-vs-linear-programming-in-python-f1be5bb4e60e
@desc: Imagine you are a strategist recruiting an army. You have:

   UNIT             |  Food  | Wood | Gold | Attack | Health
   ----------------------------------------------------------
   Swordsman        |  60   |   20  |   0  |   6    |  70
   Man-at-arms      | 100   |    0  |  20  |  12    | 155
   Bowman           |  30   |   50  |   0  |   5    |  70
   Crossbowman      |  80   |    0  |  40  |  12    |  80
   HandCannoneer    | 120   |    0  | 120  |  35    | 150
   Horseman         | 100   |   20  |   0  |   9    | 125
   Knight           | 140   |    0  | 100  |  24    | 230
   Battering Ram    |   0   |  300  |   0  | 200    | 700
   SpringLd         |   0   |  250  | 250  |  30    | 200 

Let's add combining constraints 
Let’s take 10 as an example, so power = 10*attack + health. Our objective function becomes:
min (  Σ ( food + wood + gold ) * number_of_each_unit_type )
'''

# Import OR-Tools wrapper for linear programming
from ortools.linear_solver import pywraplp

UNITS = [
    '🗡️Swordsmen',
    '🛡️Men-at-arms',
    '🏹Bowmen',
    '❌Crossbowmen',
    '🔫Handcannoneers',
    '🐎Horsemen',
    '♞Knights',
    '🐏Battering rams',
    '🎯Springalds',
    '🪨Mangonels',
]

DATA = [
    [60, 20, 0, 6, 70],
    [100, 0, 20, 12, 155],
    [30, 50, 0, 5, 70],
    [80, 0, 40, 12, 80],
    [120, 0, 120, 35, 150],
    [100, 20, 0, 9, 125],
    [140, 0, 100, 24, 230],
    [0, 300, 0, 200, 700],
    [0, 250, 250, 30, 200],
    [0, 400, 200, 12*3, 240]
]

RESOURCES = [183000, 90512, 80150]

def solve_army(solver, UNITS, DATA, RESOURCES):
 
    # 1. Create the variables we want to optimize
    units = [solver.IntVar(0, solver.infinity(), unit) for unit in UNITS]

    # 2. Add constraints for each resource
    for _, _ in enumerate(RESOURCES):
        solver.Add(sum((10 * DATA[u][-2] + DATA[u][-1]) * units[u] for u, _ in enumerate(units)) >= 1000001)

    # 3. Minimize the objective function
    solver.Minimize(sum((DATA[u][0] + DATA[u][1] + DATA[u][2]) * units[u] for u, _ in enumerate(units)))

    # Solve problem
    status = solver.Solve()

    # If an optimal solution has been found, print results
    if status == pywraplp.Solver.OPTIMAL:
        print('================= Solution =================')
        print(f'Solved in {solver.wall_time():.2f} milliseconds in {solver.iterations()} iterations')
        print()

        power = sum((10 * DATA[u][-2] + DATA[u][-1]) * units[u].solution_value() for u, _ in enumerate(units))
        print(f'Optimal value = {solver.Objective().Value()} 🌾🪵🪙resources')
        print(f'Power = 💪{power}')
        print('Army:')
        for u, _ in enumerate(units):
            print(f' - {units[u].name()} = {units[u].solution_value()}')
        print()

        food = sum((DATA[u][0]) * units[u].solution_value() for u, _ in enumerate(units))
        wood = sum((DATA[u][1]) * units[u].solution_value() for u, _ in enumerate(units))
        gold = sum((DATA[u][2]) * units[u].solution_value() for u, _ in enumerate(units))
        print('Resources:')
        print(f' - 🌾Food = {food}')
        print(f' - 🪵Wood = {wood}')
        print(f' - 🪙Gold = {gold}')
    else:
        print('The solver could not find an optimal solution.')
        
        

solve_army(pywraplp.Solver('Minimize resource consumption', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING),
           UNITS, DATA, RESOURCES)

'''
The solver found an optimal solution: we need to build 371 🐏battering rams for a total cost of 111,300 🪵wood. 
Wait, what if we don’t have that much wood? In the previous section, we only had 🪵90512: 
we cannot produce 371 🐏battering rams. 😱
So is it possible to take these limited resources into account and still try to build the best army? 
Actually, it’s super easy: we just have to copy/paste the constraints from the previous section.

In this version, we have two types of constraints:
    1. The total power must be greater than 1,000,000;
    2. We cannot spend more than our limited resources.
'''

def maximize_army(solver, UNITS, DATA, RESOURCES):
 
    # Create the variables we want to optimize
    units = [solver.IntVar(0, solver.infinity(), unit) for unit in UNITS]

    # 1. Add constraints for each resource
    for r, _ in enumerate(RESOURCES):
        solver.Add(sum((10 * DATA[u][-2] + DATA[u][-1]) * units[u] for u, _ in enumerate(units)) >= 1000001)

    # 2. Add old constraints for limited resources
    for r, _ in enumerate(RESOURCES):
        solver.Add(sum(DATA[u][r] * units[u] for u, _ in enumerate(units)) <= RESOURCES[r])

    # Minimize the objective function
    solver.Minimize(sum((DATA[u][0] + DATA[u][1] + DATA[u][2]) * units[u] for u, _ in enumerate(units)))

    # Solve problem
    status = solver.Solve()

    # If an optimal solution has been found, print results
    if status == pywraplp.Solver.OPTIMAL:
        print('================= Solution =================')
        print(f'Solved in {solver.wall_time():.2f} milliseconds in {solver.iterations()} iterations')
        print()

        power = sum((10 * DATA[u][-2] + DATA[u][-1]) * units[u].solution_value() for u, _ in enumerate(units))
        print(f'Optimal value = {solver.Objective().Value()} 🌾🪵🪙resources')
        print(f'Power = 💪{power}')
        print('Army:')
        for u, _ in enumerate(units):
            print(f' - {units[u].name()} = {units[u].solution_value()}')
        print()
    
        food = sum((DATA[u][0]) * units[u].solution_value() for u, _ in enumerate(units))
        wood = sum((DATA[u][1]) * units[u].solution_value() for u, _ in enumerate(units))
        gold = sum((DATA[u][2]) * units[u].solution_value() for u, _ in enumerate(units))
        print('Resources:')
        print(f' - 🌾Food = {food}')
        print(f' - 🪵Wood = {wood}')
        print(f' - 🪙Gold = {gold}')
    else:
        print('The solver could not find an optimal solution.')

maximize_army(pywraplp.Solver('Minimize resource consumption', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING),
              UNITS, DATA, RESOURCES)