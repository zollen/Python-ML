'''
Created on Feb 23, 2024

@author: STEPHEN
@url: https://gplearn.readthedocs.io/en/stable/examples.html#symbolic-regressor
'''
import numpy as np
from sklearn.utils.random import check_random_state
from gplearn.genetic import SymbolicRegressor


rng = check_random_state(0)

# Training samples
X_train = rng.uniform(-1, 1, 15000).reshape(5000, 3)
y_train = np.sin(X_train[:, 0])**2 + np.cos(X_train[:, 1]) + np.tan(X_train[:, 2])

# Testing samples
X_test = rng.uniform(-1, 1, 15000).reshape(5000, 3)
y_test = np.sin(X_train[:, 0])**2 + np.cos(X_train[:, 1]) + np.tan(X_train[:, 2])


est_gp = SymbolicRegressor(population_size=5000,
                           generations=30, stopping_criteria=0.01,
                           function_set=['add', 'mul', 'sin', 'cos', 'tan'],
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
est_gp.fit(X_train, y_train)

print(est_gp._program)

'''
Result: add(add(mul(sin(X0), sin(X0)), tan(X2)), cos(X1))

Result: sin(X0)^2 + cos(X1) + tan(X2)
    
'''
