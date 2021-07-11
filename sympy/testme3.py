'''
Created on Jul. 11, 2021

@author: zollen
'''

import sympy as sym

x, y, z = sym.symbols('x, y, z')

# List of equations form
print(sym.solvers.linsolve([
        x + y +     z - 1,
        x + y + 2 * z - 3
    ], (x, y, z)))


# augmented matrix form
M = sym.Matrix([
                [1, 1, 1, 1], 
                [1, 1, 2, 3]])

print(sym.linsolve(M, (x, y, z)))

# A * x = b form
system = A, b = M[:, :-1], M[:, -1]
print(sym.linsolve(system, x, y, z))

