'''
Created on Jul. 11, 2021

@author: zollen
'''

import sympy as sym

x, y, z = sym.symbols('x y z')
M = sym.Matrix([
            [1, 0, 1, 3],
            [2, 3, 4, 7],
            [-1, -3, -3, -4]
            ])
sym.pprint(M)
sym.pprint(M.rref())
sym.pprint(sym.solve_linear_system(M, x, y, z))
print(sym.shape(M))
print(sym.shape(M.T))

M = sym.Matrix([
            [3, -2, 4 , -2],
            [5,  3, -3, -2],
            [5, -2, 2,  -2],
            [5, -2, -3,  3]
            ])
print("=====================")
sym.pprint(M.det())
print("=====================")
sym.pprint(M.eigenvals())
print("====================")
sym.pprint(M.eigenvects())
print("==========PDP^-1==========")
sym.pprint(M.diagonalize())

print("===========================")
'''
3x + 2y - x = 1
2x - 2y + 4z = -2
2x - y + 2z = 0
'''
x, y, z = sym.symbols('x y z')
A = sym.Matrix([
                [3, 2, -1], 
                [2, -2, 4], 
                [2, -1, 2]])
b = sym.Matrix([1, -2, 0])
sym.pprint(sym.linsolve((A, b), (x, y, z)))