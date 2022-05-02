'''
Created on Apr. 26, 2022

@author: zollen
@desc: My example

attributes: costs, range, maintenance, 
1. Predator Drone
2. F15 Eagle
3. Precision Self-Propelled Howitzer (M109)

1. Hellfire Missiles
2. Harpoon Missiles
3. Precision 


Defintition
-----------
v - vechile, w - weapon, s - site, t - target

Defined
-------
I(w)   - Inventory                     I(w)  ∈ (0, +∞)
VT(v)  - Max trips of vechile         TP(v)  ∈ (0, +∞)
C1(vs) - Gas/Pilots/Maintenance       C1(vs) ∈ (0, +∞)
C2(w)  - Armament Cost                C2(w)  ∈ (0, +∞)

Optimization
------------
T(vs)     - boolean variable of trip                       T(vs)  ∈ {0, 1}
F(vstw)   - boolean variable of firing solution            F(vst) ∈ {0, 1}



Objective - mininize cost
-------------------------

Min ( Σ(vs) ( T(vs) * C1(vs)  +  Σ(tw) ( F(vwst) * C2(w) )  )  )


Constaints
----------
TripFireSolutionCheck
    Σ(tw) ( 1000 * T(vs) - Σ(tw) F(vwst) ) >= 0 for all instances of vs
    
Inventory Limits
    Σ(vst) ( F(vwst) ) <= I(w) for all instance of w
    
Vechile Trip Limits
    Σ(s) ( T(vs) ) <= VT(v)  for all instance of v
    
Must Kill All Targets
    Σ(st) ( Σ(tw) ( F(vwst) ) == 1 ) for all instance of st
'''

import pandas as pd
import re
import random
import itertools
from ortools.linear_solver import pywraplp

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)


W_HELLFIRE = 1
W_HARPOON = 2
W_ARTILLARY = 3
WEAPONS = [ W_HELLFIRE, W_HARPOON, W_ARTILLARY ]
WEAPONS_DICT = { 1: 'W_HELLFIRE', 2: 'W_HARPOON', 3: 'W_ARTILLARY' }

V_PREDATOR_DRONE = 4
V_F15_JET = 5
V_M109_HOWITZER = 6
VECHILES = [ V_PREDATOR_DRONE, V_F15_JET, V_M109_HOWITZER ]
VECHILES_DICT = { 4: 'V_PREDATOR_DRONE', 5: 'V_F15_JET', 6: 'V_M109_HOWITZER' }

TARGETS = { 'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'F': [], 'G': [], 'H': [], 'I': [], 'J': [], 'K': [],
              'L': [], 'M': [], 'N': [], 'O': [], 'P': [], 'Q': [], 'R': [], 'S': [], 'T': [], 'U': [] }
SITES = list(TARGETS.keys())


def vechiles_sites_targets():
    
    targets = []
    
    for v in VECHILES:
        for s in SITES:
            for t in TARGETS[s]:
                targets.append([v, s, t])
                
    return targets




for i in range(1, 10):
    TARGETS[random.choice(SITES)].append(i)


for idx in SITES:
    print(idx, ' => ', TARGETS[idx])
    



C1 = {}
C2 = {}
I = {}
H = {}
VT = {}

for c in SITES:
    C1[V_PREDATOR_DRONE, c] = random.randint(1, 5)
    C1[V_F15_JET, c] = random.randint(1, 5)
    C1[V_M109_HOWITZER, c] = random.randint(1, 5)

C2[W_HELLFIRE] = 3
C2[W_HARPOON] = 2
C2[W_ARTILLARY] = 1    

I[W_HELLFIRE] = 10
I[W_HARPOON] = 15
I[W_ARTILLARY] = 20

H[V_PREDATOR_DRONE] = 2
H[V_F15_JET] = 4
H[V_M109_HOWITZER] = 5

VT[V_PREDATOR_DRONE] = 2
VT[V_F15_JET] = 20
VT[V_M109_HOWITZER] = 100

T = {}
F = {}


    
    

solver = pywraplp.Solver.CreateSolver('SCIP')

# initialization
for v in VECHILES:
    for s in TARGETS.keys():
        T[v, s] = solver.BoolVar(name=f"T({v},{s})")
        
for v in VECHILES:
    for w in WEAPONS:
        for s in SITES:
            for t in TARGETS[s]:
                F[v, w, s, t] = solver.BoolVar(name=f"F({v},{w},{s},{t})")


# constraints

'''
TripFireSolutionCheck
    Σ(tw) ( 1000 * T(vs) - Σ(tw) F(vwst) ) >= 0 for all instances of vs
'''
for v in VECHILES:
    for s in SITES:
        solver.Add(1000 * T[v, s] + 
                   solver.Sum(
                       [-1 * F[v, r[0], s, r[1]] for r in itertools.product(WEAPONS, TARGETS[s]) ]) >= 0, 
                       name =f'TripFireSolution({v},{w},{s},{t})')     
        
'''
InventoryLimits
    Σ(vst) ( F(vwst) ) <= I(w) for all instance of w
'''
for w in WEAPONS: 
    solver.Add(solver.Sum( 
            [ F[r[0], w, r[1], r[2]] for r in vechiles_sites_targets() ]) <= I[w],
            name = f'InventoryLimit({v},{w},{s},{t})' )

'''
Vechile Trip Limits
    Σ(s) ( T(vs) ) <= VT(v)  for all instance of v
'''         
for v in VECHILES:
    solver.Add(solver.Sum(
            [ T[v, s] for s in TARGETS.keys() ]) <= VT[v],
            name = f'VechileTripLimit({v},{s})')
    
'''
Must Kill All Targets
    Σ(st) ( Σ(tw) ( F(vwst) ) == 1 ) for all instance of st
''' 
for s in SITES:                
    for t in TARGETS[s]:
        solver.Add( solver.Sum( [ F[r[0], r[1], s, t] for r in itertools.product(VECHILES, WEAPONS) ] ) == 1,
                    name = f'KillAllTargets({v},{w},{s},{t})')
        
    
if False:    
    print(solver.ExportModelAsLpFormat(obfuscated=False))
           
           
'''
Objective
Min ( Σ(vs) ( T(vs) * C1(vs)  +  Σ(tw) ( F(vwst) * C2(w) )  )  )
'''           
objective_function = []

for v in VECHILES:
    for s in SITES:
        objective_function.append(T[v, s] * C1[v, s])
        for t in TARGETS[s]:
            for w in WEAPONS:
                objective_function.append(F[v, w, s, t] * C2[w])
                
solver.Minimize(solver.Sum(objective_function))

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print(f'Solution: Total cost = ${solver.Objective().Value()}')
else:
    print('A solution could not be found, check the problem specification')
    
print()  

result_list = []
for var in [ T, F ]:
    variable_optimal = [ i.solution_value() for i in var.values() ] 
    var_result=list(zip(var.values(),variable_optimal))
    df=pd.DataFrame(var_result,columns=['Name','Value'])
    result_list.append(df)
    
result_t = result_list[0]
result_f = result_list[1]


def processT(rec):
    
    m = re.split('[\,()]+', rec['Name'].name()[1:])
    idx = 0
    
    for token in m:

        if len(token) <= 0:
            continue
        
        if idx == 0:
            rec['VECHILE'] = VECHILES_DICT[int(token)]
        else:
            rec['SITE'] = token
            
        idx = idx + 1
                    
    return rec

def processF(rec):
    
    m = re.split('[\,()]+', rec['Name'].name()[1:])
    idx = 0
    
    for token in m:

        if len(token) <= 0:
            continue
        
        if idx == 0:
            rec['VECHILE'] = VECHILES_DICT[int(token)]
        elif idx == 1:
            rec['WEAPON'] = WEAPONS_DICT[int(token)]
        elif idx == 2:
            rec['SITE'] = token
        else:
            rec['TARGET'] = token
            
        idx = idx + 1
                    
    return rec
  

result_t = result_t[result_t['Value'] == 1.0].drop(columns=['Value']).apply(processT, axis = 1)
result_f = result_f[result_f['Value'] == 1.0].drop(columns=['Value']).apply(processF, axis = 1)


k = result_t.merge(result_f, how='left', on=['VECHILE', 'SITE'])[['Name_x', 'Name_y', 'VECHILE', 'SITE', 'TARGET', 'WEAPON']]
print(k)


