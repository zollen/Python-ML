'''
Created on Jul. 11, 2021

@author: zollen
'''

import sympy as sym

'''
Eqn1: x + y = 5
Eqn2: x^2 + y^2 = 17

There are 2 solutions
[(1, 4), (4, 1)]
'''
x, y = sym.symbols('x y')
eq1 = sym.Eq(x + y, 5)
eq2 = sym.Eq(x**2 + y**2, 17)
sym.pprint(sym.solve([eq1, eq2], (x, y)))


'''
Eqn1: 2x^2 + y + z = 1
Eqn2: x + 2y + z = c1
Eqn3: -2x + y = -z
'''
x, y, z = sym.symbols('x y z')
c1 = sym.Symbol('c1')
eq1 = sym.Eq(2*x**2 + y + z, 1)
eq2 = sym.Eq(x + 2 * y + z, c1)
eq3 = sym.Eq(-2 * x + y, -z)
sym.pprint(sym.solve([eq1, eq2, eq3], (x, y, z)))
