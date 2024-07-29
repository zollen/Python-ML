'''
Created on Jul 29, 2024

@author: STEPHEN
'''


import highspy

h = highspy.Highs()

swordsmen = h.addIntegral(lb = 0, ub = 10)
bowmen = h.addIntegral(lb = 0, ub = 10)
horsemen = h.addIntegral(lb = 0, ub = 10)


h.addConstr(swordsmen*60 + bowmen*80 + horsemen*140 <= 1200)
h.addConstr(swordsmen*20 + bowmen*10 <= 800)
h.addConstr(bowmen*40 + horsemen*100 <= 600)

h.maximize(swordsmen*70 + bowmen*95 + horsemen*230)

lp = h.getLp()
num_nz = h.getNumNz()
print('LP has ', lp.num_col_, ' columns', lp.num_row_, ' rows and ', num_nz, ' nonzeros')
h.run()
solution = h.getSolution()
basis = h.getBasis()
info = h.getInfo()
model_status = h.getModelStatus()

print('Model status = ', h.modelStatusToString(model_status))
print('Optimal objective = ', info.objective_function_value)
print("Variable Values:", solution.col_value)
