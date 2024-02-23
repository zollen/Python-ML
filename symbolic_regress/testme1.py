'''
Created on Feb 23, 2024

@author: STEPHEN
@url: https://gplearn.readthedocs.io/en/stable/examples.html#symbolic-regressor
'''
import numpy as np
from sklearn.utils.random import check_random_state
from gplearn.genetic import SymbolicRegressor


x0 = np.arange(-1, 1, 1/10.)
x1 = np.arange(-1, 1, 1/10.)
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - x1**2 + x1 - 1

rng = check_random_state(0)

# Training samples
X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
y_train = X_train[:, 0]**2 - X_train[:, 1]**2 + X_train[:, 1] - 1

# Testing samples
X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
y_test = X_test[:, 0]**2 - X_test[:, 1]**2 + X_test[:, 1] - 1


est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
est_gp.fit(X_train, y_train)

print(est_gp._program)

'''
Result: sub(add(-0.999, X1), mul(sub(X1, X0), add(X0, X1)))
Result: y = (-0.999 + X1) - ((X1 - X0) *  (X0 + X1))
        y = (-0.999 + X1) - (X1X0 + X1^2 - X0^2 - X0X1)
        y = (-0.999 + X1) - (X1^2 - X0^2)
        y = -0.999 + X1 - X1^2 + X0^2
        y = X0^2 - X1^2 + X1 - 1
'''
