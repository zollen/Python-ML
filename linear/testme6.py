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

Min ( Σ(vs) ( T(vs) * C1(vs)  +  Σ(tw) ( F(vstw) * C2(w) )  )  )


Constaints
----------
TripFireSolutionCheck
    Σ(tw) ( 1000 * T(vs) - Σ(tw) F(vstw) ) >= 0 for all instances of vs
    
Inventory Limits
    Σ(vst) ( F(vstw) ) <= I(w) for all instance of w
    
Vechile Trip Limits
    Σ(s) ( T(vs) ) <= VT(v)  for all instance of v
    
Must Kill All Targets
    Σ(st) ( 1000 * TARGET[st] - Σ(tw) ( F(vstw) ) <= 999 ) for all instance of st
'''

import pandas as pd
import re
from collections import defaultdict
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

S_ODESSA = 7
S_KYIV = 8
S_MARIUPOL = 9
SITES =  [ S_ODESSA, S_KYIV, S_MARIUPOL ]
SITES_DICT = { 7: 'S_ODESSA', 8: 'S_KYIV', 9: 'S_MARIUPOL' }

T_INFANTRY = 10
T_TANK = 11
T_APC = 12
ENEMIES = [ T_INFANTRY, T_TANK, T_APC ]
ENEMIES_DICT = { 10: 'T_INFANTRY', 11: 'T_TANK', 12: 'T_APC' }






def def_value():
    return 0
   
TARGETS = defaultdict(def_value)
TARGETS[S_KYIV, T_INFANTRY] = 1   
TARGETS[S_ODESSA, T_TANK  ] = 1
TARGETS[S_MARIUPOL, T_APC] = 1


C1 = {}
C2 = {}
I = {}
H = {}
VT = {}

I[W_HELLFIRE] = 10
I[W_HARPOON] = 15
I[W_ARTILLARY] = 20

C1[V_PREDATOR_DRONE, S_ODESSA] = 1
C1[V_PREDATOR_DRONE, S_KYIV] = 1
C1[V_PREDATOR_DRONE, S_MARIUPOL] = 2
C1[V_F15_JET, S_ODESSA] = 4
C1[V_F15_JET, S_KYIV] = 3
C1[V_F15_JET, S_MARIUPOL] = 4
C1[V_M109_HOWITZER, S_ODESSA] = 2
C1[V_M109_HOWITZER, S_KYIV] = 2
C1[V_M109_HOWITZER, S_MARIUPOL] = 2

C2[W_HELLFIRE] = 3
C2[W_HARPOON] = 2
C2[W_ARTILLARY] = 1    

H[V_PREDATOR_DRONE] = 2
H[V_F15_JET] = 4
H[V_M109_HOWITZER] = 5

VT[V_PREDATOR_DRONE] = 2
VT[V_F15_JET] = 20
VT[V_M109_HOWITZER] = 100

T = {}
F = {}

GAME_SITES = []
GAME_TARGETS = []

for t in TARGETS.keys():
    GAME_SITES.append(t[0])
    GAME_TARGETS.append(t[1])
    
    

solver = pywraplp.Solver.CreateSolver('SCIP')

# initialization
for v in VECHILES:
    for s in GAME_SITES:
        T[v, s] = solver.BoolVar(name=f"T({v},{s})")
        
for v in VECHILES:
    for s in GAME_SITES:
        for t in GAME_TARGETS:    
            for w in WEAPONS:
                F[v, s, t, w] = solver.BoolVar(name=f"F({v},{s},{t},{w})")


# constraints

'''
TripFireSolutionCheck
    Σ(tw) ( 1000 * T(vs) - Σ(tw) F(vstw) ) >= 0 for all instances of vs
'''
for v in VECHILES:
    for s in GAME_SITES:
        solver.Add(1000 * T[v, s] + 
                   solver.Sum(
                       [-1 * F[v, s, r[0], r[1]] for r in itertools.product(GAME_TARGETS, WEAPONS) ]) >= 0, 
                       name =f'TripFireSolution({v},{s},{t},{w})')     
        
'''
InventoryLimits
    Σ(vst) ( F(vstw) ) <= I(w) for all instance of w
'''
for w in WEAPONS: 
    solver.Add(solver.Sum( 
            [ F[r[0], r[1], r[2], w] for r in itertools.product(VECHILES, GAME_SITES, GAME_TARGETS) ]) <= I[w],
            name = f'InventoryLimit({v},{s},{t},{w})' )

'''
Vechile Trip Limits
    Σ(s) ( T(vs) ) <= VT(v)  for all instance of v
'''         
for v in VECHILES:
    solver.Add(solver.Sum(
            [ T[v, s] for s in GAME_SITES ]) <= VT[v],
            name = f'VechileTripLimit({v},{s})')
    
'''
Must Kill All Targets
    Σ(st) ( 1000 * TARGET[st] - Σ(tw) ( F(vstw) ) <= 999 ) for all instance of st
'''
for s in GAME_SITES:
    for t in GAME_TARGETS:
            solver.Add(1000 * TARGETS[s, t] - 
                       solver.Sum([ F[r[0], s, t, r[1]] for r in itertools.product(VECHILES, WEAPONS) ]) <= 999,
                       name = f'KillAllTargets({v},{s},{t},{w})')
             
    
    
if False:    
    print(solver.ExportModelAsLpFormat(obfuscated=False))
           
           
'''
Objective
Min ( Σ(vs) ( T(vs) * C1(vs)  +  Σ(tw) ( F(vstw) * C2(w) )  )  )
'''           
objective_function = []

for v in VECHILES:
    for s in GAME_SITES:
        objective_function.append(T[v, s] * C1[v, s])
        for t in GAME_TARGETS:
            for w in WEAPONS:
                objective_function.append(F[v, s, t, w] * C2[w])
                
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

def process(rec):
    
    m = re.split('[A-Z\,()]+', rec['Name'].name())
    idx = 0
    
    for token in m:

        if len(token) <= 0:
            continue
        
        if idx == 0:
            rec['VECHILE'] = VECHILES_DICT[int(token)]
        elif idx == 1:
            rec['SITE'] = SITES_DICT[int(token)]
        elif idx == 2:
            rec['TARGET'] = ENEMIES_DICT[int(token)]
        else:
            rec['WEAPON'] = WEAPONS_DICT[int(token)]
            
        idx = idx + 1
                    
    return rec
  

result_t = result_t[result_t['Value'] == 1.0].drop(columns=['Value']).apply(process, axis = 1)
result_f = result_f[result_f['Value'] == 1.0].drop(columns=['Value']).apply(process, axis = 1)

k = result_t.merge(result_f, how='left', on=['VECHILE', 'SITE'])[['Name_x', 'Name_y', 'VECHILE', 'SITE', 'TARGET', 'WEAPON']]
print(k)


