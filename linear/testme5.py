'''
Created on Apr. 20, 2022

@author: zollen
@url: https://towardsdatascience.com/fleet-and-workforce-planning-with-linear-programming-16db08c7f91d
@desc: let's solve a more complex problem

To simplify the problem, instead of considering units of products, we’ll talk about boxes of products i.e. 
the demand and stock of a product is measured in boxes of the product (in a real life scenario after 
predicting the demand of products that are delivered in boxes/packages we’d need to convert the raw 
number into the number of boxes or grouping units needed as that’s the way of transporting them).

Next, we’ll assume you also know how much it costs to move a box of each product from any of your 
warehouses to any of your stores using any of your vans. Besides this variable cost, there’s also a 
fixed cost for using each vehicle (you can think of it as a depreciation/maintenance cost).

Additionally, we’ll suppose that there’s a limit to the number of trips each van can make. In this case 
the limit is 5 trips. What’s more, each van has a default number of boxes that it can fit in. On top of 
that, each of these vehicles (if used) needs to have employees assigned in pairs (a driver and an 
assistant) in order to be operational. Also, we can only assign each employee to a single van, and 
if we do, he’ll receive a fixed salary of $1500.

Finally, during a chess tournament that occurred the previous week, some of the employees got into a 
brawl so it is desirable to avoid assigning a pair of conflicting workers to the same van; as a matter 
of fact, we have 23 pairs of conflicting employees (J=23). If we do end up assigning them to the same 
van, we’ll have to deal with the consequences i.e. a penalty of $500.

In summary our context variables are the following:
  “P” products = 5
  “W” warehouses = 3
  “S” stores = 3
  “V” vans = 4
  “E” employees = 8
  “J” conflicting pairs of employees = 23


Variable and Constraints
c(pwsv) Unit cost of sending a box of product p from warehouse w to store s using vechile v, where c(pwsv) ∈ (0, +∞)
x(pwsv) Number of boxes of product p sent from warehouse w to store s using van v, where x(pwsv) ∈ [0, +∞)
T(v) Boolean variable created to verify that the vehicle v is used, where T(v) ∈ {0, 1}
F(v) Integer fixed cost for using vechile v, where F(v) ∈ (0, +∞)
A(ve) Boolean variable created to verify if an employee e is assigned to a van v, where A(ve) ∈ {0, 1}
Z(wsv) Boolean variable used to check if a trip from warehouse w to store s was made using van v, where Z(wsv) ∈ {0, 1}
H(vj) Boolean variable used to check if just one employee of a conflicting pair j is assigned to a van v, where H(vj) ∈ {0, 1}
G(vj) Boolean variable used to check if the conflicting pair j is assigned to the same van v, where G(vj) ∈ {0, 1}
 
p ∈ P, w ∈ W, s ∈ S, v ∈ V, e ∈ E, j ∈ J


Objective
Min ( Σ(pwsv) c(pwsv) * x(pwsv) + Σ(v) T(v) * F(v)  + Σ(ve) 1500 A(ve) + Σ(vj) 500 * G(vj) )


Constraints
1. The first restriction specifies that the demand of each product p at each store s must be satisfied, 
    i.e. the sum of all the combinations of product boxes sent from each warehouse w to each store s 
    using any of the vehicles v must equal the number of boxes needed by the store to cover their expected 
    demand;
    
    Σ(wv) x(pwsv) = demand(ps)  for all instances of p, s
    
2. Stipulates that we can’t send boxes that we don’t have. In other words, the sum of product boxes sent 
    from a warehouse must be lower or equal than its available boxes;
    
    Σ(sv) x(pwsv) <= stock(pw)  for all instances of p, w
    
3. Specifies that we can’t surpass the limit of boxes that can fit the van, so the sum of boxes transported 
    in each vehicle must be lower or equal to the van’s capacity(v) at each trip. With this 
    constraint we count the number of trips a van will make with the auxiliary variable Z(wsv), as 
    capacity(v) is the maximum number of boxes that can be transported at each trip by vehicle v, 
    we can count as one trip the transportation of several products. Also, this constraint implicitly 
    blocks the possibility of repeating the same trip;
    
    Σ(p) x(pwsv) <= capacity(v) * Z(wsv)  for all instances of w, s, v
    
4. Signals that none of the vans can make more than 5 trips while also checking if each of them was 
    used or not. Note that this constraint is chained with the previous one. How? Well, once we know 
    if a trip was made or not by using constraint (3), we simply sum the number of Z(wsv) and require 
    that the total is lower or equal than 5 (the trip limit). Here, due to the specification of the 
    equation, T(v) will be equal to 1 unless the van is left unused;
    
    Σ(ws) Z(wsv) <= 5 * T(v) for all instances of v
    
5. Specifies that each van will have 2 or zero employees assigned;

    Σ(e) A(ve) = 2 * T(v) for all instances of v
    
6. Requires that each employee can only be assigned to a single van;

    Σ(v) A(ve) <= 1 for all instances of e

7.  This constraint accounts for the fact that the desirable constraint on the pair of conflicting 
    employees j is optional. When a pair of conflicting employees j=(e1, e2) is assigned to the same 
    van (A(ve1) + A(ve2) = 2) then the penalty is activated (G(vj)=1) so that the equation equals zero. 
    If just one member of the pair is assigned to a van then H(vj)=1. If none of the conflicting employees 
    of pair j is assigned to the van v then all the elements are zero.
    
    Σ(v) A(ve1) + A(ve2) - H(vj) - 2 G(vj) = 0 for all instances of e, j with J = {e1, e2) (1,2) with conflicts
                                                                            and e1 != e2
                                                                            
8.  The final constraint specifies the upper and lower bound for each variable. Here we declare which 
    variables are binary and which ones are integers.  
    
    x(pwsv) ∈ [0, +∞)    T(v), Z(wsv), A(ve), H(vj), G(vj) ∞ {0, 1}                                                      
'''
import pandas as pd
import numpy as np
import itertools
from ortools.linear_solver import pywraplp

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

#1) Specify a seed to be able to replicate the experiment
np.random.seed(2)

#2) Declare the context variables
W_warehouses = 3
S_stores = 3
P_products = 5
V_vans = 4
E_employees = 8
trip_limit = 5

#3) Set some thresholds for the simulated data
min_cost, max_cost = 10, 60
min_stock, max_stock = 8, 12
min_demand, max_demand = 1, 10
min_van_fixed_cost, max_van_fixed_cost = 3000, 5000
fixed_salary = 1500
conflict_penalty = 500

#4) GENERATE DATA
# 1 Cost matrix per product (of size W_warehouses x S_stores x V_vans)
# 1 Stock vector per product (of size W_warehouses)
# 1 Demand vector per product (of size S_stores)
# 1 List of capacities per van (i.e. how many boxes each can transport)
# 1 List of pairs of conflicts between employees

costs = []
stocks = []
demands = []
capacities = [5,6,6,4]
J_employees_conflicts = [(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),
                         (2,3),(2,4),(2,5),(2,6),(2,7),(2,8),
                         (3,4),(3,5),(3,6),(3,7),(3,8),
                         (4,5),(4,6),(4,7),(4,8),
                         (5,7)]

# For each product we'll generate random costs, stocks and demands, given by the intervals [low,high]
for i in range(P_products):
    costs_p=np.random.randint(low=min_cost,high=max_cost,size=W_warehouses*S_stores*V_vans).reshape(W_warehouses,S_stores,V_vans)
    stock_p=np.random.randint(low=min_stock,high=max_stock,size=W_warehouses)
    demand_p=np.random.randint(low=min_demand,high=max_demand,size=S_stores)
   
    costs.append(costs_p)
    stocks.append(stock_p)
    demands.append(demand_p)

# We create a vector of fixed costs for using each of the vans
for v in range(V_vans):
    costs_v=np.random.randint(low=min_van_fixed_cost,high=max_van_fixed_cost,size=V_vans)
    






# In this case, we are using pywraplp from Google OR-Tools. Note that there are several solvers available 
# in pywraplp such as GLOP, SCIP, GUROBI and IBM CPLEX. Since GUROBI and CPLEX require a licence and GLOP 
# is designed for simple Linear Programming but the problem we have at hand requires a solver to approach 
# Integer or Mixed Integer Linear problems, we’ll use SCIP (one of the fastest non-commercial solvers that 
# can handle Integer and Mixed Integer Linear problems).
solver = pywraplp.Solver.CreateSolver('SCIP')

# (lower bound = 0 and upper bound = solver.infinity()).
x = {}
for p in range(P_products):
    for w in range(W_warehouses):
        for s in range(S_stores):
            for v in range(V_vans):
                x[p,w,s,v] = solver.IntVar(lb=0,
                                           ub=solver.infinity(),
                                           name=f"x_{p+1}_{w+1}_{s+1}_{v+1}")
                
T = {}
for v in range(V_vans):
    T[v] = solver.BoolVar(name=f"T_{v+1}")
    
A = {}
for v in range(V_vans):
    for e in range(E_employees):
        A[v,e] = solver.BoolVar(name=f"A_{v+1}_{e+1}")
        
Z = {}
for v in range(V_vans):
    for w in range(W_warehouses):
        for s in range(S_stores):
            Z[w,s,v] = solver.BoolVar(name=f"Z_{w+1}_{s+1}_{v+1}")
            
H = {}
G = {}
for v in range(V_vans):
    for j in range(len(J_employees_conflicts)):
        H[v,j] = solver.BoolVar(name=f"H_{v+1}_{j+1}")
        G[v,j] = solver.BoolVar(name=f"G_{v+1}_{j+1}")
        
        

# Demand constraint
for p in range(P_products):
    for s in range(S_stores):
        solver.Add(
            solver.Sum(
                [x[p, j[0], s, j[1]] for j in itertools.product(
                                  range(W_warehouses),
                                  range(V_vans))]) == demands[p][s],
                                  name='(1) Demand')
# Stock constraint
for p in range(P_products):
    for w in range(W_warehouses):
        solver.Add(
            solver.Sum(
                [x[p, w, j[0],j[1]] for j in itertools.product(
                                   range(S_stores),
                                   range(V_vans))]) <= stocks[p][w],
                                   name='(2) Stock')

# Trip verification constraint
for v in range(V_vans):
    for s in range(S_stores):
        for w in range(W_warehouses):
            solver.Add(
                solver.Sum(
                    [x[p, w, s, v] for p in range(P_products)])
                    <= capacities[v]*Z[w, s, v],
                    name='(3) TripVerification')
            
# Van use and trip limit constraint
for v in range(V_vans):
    solver.Add(
        solver.Sum(
            [Z[j[0], j[1],v] for j in itertools.product(
                            range(W_warehouses),
                            range(S_stores))]) <= trip_limit * T[v],
                            name='(4) TripLimit')
    
# Number of employees per van
for v in range(V_vans):
    solver.Add(
        solver.Sum(
            [A[v,e] for e in range(E_employees)]) == 2 * T[v],
            name='(5) EmployeeRequirement')
    
# Number of vans an employee can be assigned to
for e in range(E_employees):
    solver.Add(
        solver.Sum([A[v,e] for v in range(V_vans)]) <= 1,
        name='(6) JobLimit')
    
# Verification of the constraint compliance
for v in range(V_vans):
    for idx, j in enumerate(J_employees_conflicts):
        solver.Add(
            solver.Sum([A[v,j[0] - 1]]) == -A[v,j[1]-1] + H[v, idx] + 2 * G[v,idx],
            name='(7) ConflictVerification')

if False:        
    print(solver.ExportModelAsLpFormat(obfuscated=False))


# Objective Function
# Min ( Σ(pwsv) c(pwsv) * x(pwsv) + Σ(v) T(v) * F(v)  + Σ(ve) 1500 A(ve) + Σ(vj) 500 * G(vj) )
objective_function = []
# First term -> Transportation variable costs
for p in range(P_products):
    for w in range(W_warehouses):
        for s in range(S_stores):
            for v in range(V_vans):
                objective_function.append(costs[p][w][s][v] * x[p, w, s, v])
                
# Second term -> Transportation fixed costs
for v in range(V_vans):
    objective_function.append(costs_v[v] * T[v])
    
# Third term -> Salary payments
for v in range(V_vans):
    for e in range(E_employees):
        objective_function.append(fixed_salary * A[v,e])
        
# Fourth term -> Penalties for not avoiding conflicts
for v in range(V_vans):
    for j in range(len(J_employees_conflicts)):
        objective_function.append(conflict_penalty * G[v,j])
        
# Type of problem
solver.Minimize(solver.Sum(objective_function))

# Call the solver method to find the optimal solution
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print(f'Solution: \n Total cost = ${solver.Objective().Value()}')
else:
    print('A solution could not be found, check the problem specification')
    
print()
print()    
    
# extract the optimal values for each variable
# preprocess and rearrange the data into a table   
result_list = []
# Extract the solution details and save them in a list of tables
for var in [x,Z,T,A,H,G]:
    variable_optimal = []
    for i in var.values():
        variable_optimal.append(i.solution_value())
    
    var_result=list(zip(var.values(),variable_optimal))
    df=pd.DataFrame(var_result,columns=['Name','Value'])
    result_list.append(df)
        

        
# Concatenate the tables and extract the variable names
results=pd.concat(result_list)
results['Name']=results['Name'].astype(str)
results.reset_index(drop=True,inplace=True)
results['Variable']=results['Name'].str.extract("(^(.)\d?)")[0]
results['Variable']=results['Variable'].str.upper()
results['Value']=results['Value'].map(int)


# Create a mapping of variables and indices to simplify the analysis
variable_indices={'X':'X_product_warehouse_store_van',
                  'A':'A_van_employee',
                  'T':'T_van',
                  'H':'H_van_pair',
                  'G':'G_van_pair',
                  'Z':'Z_warehouse_store_van'}
results['Indices'] = results['Variable'].map(variable_indices)

# Order the columns
results=results[['Variable','Indices','Name','Value']].copy()

# We look for the vans that we are going to use by filtering the column Variable with “T” 
# (binary variable to signal the usage of the vehicle) and the column Value with 1 (which means, 
# the optimal solution implies that we need to use these vans)
# The below result shows vans 1, 2 and 3 will be used
print("Usable Vans: ", np.unique(list(results[(results['Variable']=='T')&(results['Value']==1)].Name)))
print()
print()

# For the next part, we’ll just search for the variables related to van 1, starting with answering which 
# trips are going to be made using this vehicle. Here it is important to remember that van v=1 corresponds 
# to the index v=0:
trips_van_1=[]
for w in range(W_warehouses):
    for s in range(S_stores):
        for v in range(V_vans):
            if v==0:
                trips_van_1.append(str(Z[w,s,v]))
trips_df=results[(results['Variable']=='Z')&(results['Value']>0)]

# Warehouse 1 to store 1 and 3
# Warehouse 2 to store 2
# Warehouse 3 to store 2 and 3
print("Trips are made by using vans 1")
print(trips_df[trips_df['Name'].isin(trips_van_1)].drop_duplicates())
print()
print()


# Next, we need to find the employees that are going to be in charge of the delivery operations of van 1
employees_van_1=[]
for v in range(V_vans):
    for e in range(E_employees):
        if v==0:
            employees_van_1.append(str(A[v,e]))
            
employees_df=results[(results['Variable']=='A')&(results['Value']>0)]


# Employee 5 and 8 are both assgiend to van 1
print("Employee 5 and 8 are both assgiend to van 1")
print(employees_df[employees_df['Name'].isin(employees_van_1)].drop_duplicates())
print()
print()


# Now the final question regarding van 1 is how many boxes and of which products it will have to transport 
# at each of the trips. Let’s do this for the trip from warehouse 2 to store 2

transport_df = results[(results['Variable']=='X')&(results['Value']>0)]
transport_trip_2_2 = []
for p in range(P_products):
    for w in range(W_warehouses):
        for s in range(S_stores):
            for v in range(V_vans):
                if w==1 and s==1 and v==0:
                    transport_trip_2_2.append(str(x[p,w,s,v]))

# Total number of 4 of product 2 would be transported by van 1, from warehouse 2 to store 2
# Total number of 1 of prdouct 3 would be transported by van 1, from warehouse 2 to store 2
print("(5 max) 4 boxes of product 2, 1 box of product 3 would be transported by van 1, from warehouse 2 to store 2")
print(transport_df[transport_df['Name'].isin(transport_trip_2_2)].drop_duplicates())
print()
print()

# conflicting pair of employees
print("Conflicting paris of employees")
print(results[(results['Variable']=='G')&(results['Value']!=0)].drop_duplicates(), " => ", (1, 4))
print(results[(results['Variable']=='A')&(results['Value']!=0)].drop_duplicates())