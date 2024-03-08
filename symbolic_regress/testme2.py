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
X_train = rng.uniform(-10, 10, 15000).reshape(5000, 3)
y_train = 2 * X_train[:, 0]**3 - 3 * X_train[:, 1]**2 + 4 * X_train[:, 2]

# Testing samples
X_test = rng.uniform(-10, 10, 15000).reshape(5000, 3)
y_test = 2 * X_train[:, 0]**3 - 3 * X_train[:, 1]**2 + 4 * X_train[:, 2]


est_gp = SymbolicRegressor(population_size=10000,
                           generations=40, stopping_criteria=0.0001,
                           function_set=('add', 'mul', 'sub', 'sin', 'cos', 'tan'),
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
est_gp.fit(X_train, y_train)

print(est_gp._program)

'''
Result: add(
            add(
                add(
                    add(X2, X2), 
                    mul(X1, X1)
                    ), 
                    add(
                        add(
                            add(X2, X2), 
                            add(
                                add(X2, X2), 
                                add(
                                    add(X2, X2), 
                                    mul(X1, X1)
                                )
                            )
                        ), 
                        mul(X1, X1)
                    )
                ), 
                mul(
                    add(X0, X0), 
                    mul(X0, X0)
                )
            )

Result: y =  X1^2 + (8 * X2 + 2 * X1^2) + (2* X0 * X0^2)
        y = 2 * X0^3 + 3 * X1^2 + 8 * X2
    
'''
