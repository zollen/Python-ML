'''
Created on Jul. 11, 2021

@author: zollen
'''

import sympy as sym

sym.init_printing(use_unicode=True)

sym.pprint(sym.sqrt(8))
sym.pprint(sym.sqrt(8).evalf())

x, y = sym.symbols('x y')
expr = x + 2 * y
sym.pprint(expr)

sym.pprint(expr + 1)
sym.pprint(expr - x)

ee = sym.expand(x * expr)
sym.pprint(ee)

sym.pprint(sym.factor(ee))


x, y, z, nu = sym.symbols('x t z nu')

# Derivative f(x) = sin(x)
sym.pprint(sym.diff(sym.sin(x), x))

# integral f'(x) = -sin(x)
sym.pprint(sym.integrate(-sym.sin(x), x))

# limit x-> 0 sin(x)/x
sym.pprint(sym.limit(sym.sin(x)/x, x, 0))

# solve x^2 - 2 = 0
sym.pprint(sym.solve(x**2 - 2, x))

expr = x**3 + 4 * x*y - z
sym.pprint(expr.subs([(x, 2), (y, 4), (z, 0)]))

# solve the differential equstion y'' - y = e^t
t = sym.symbols('t')
y = sym.Function('y')
sym.pprint(sym.dsolve(sym.Eq(y(t).diff(t, t) - y(t), sym.exp(t)), y(t)))

# find the eign values of [[1, 2], [2, 2]]
sym.pprint(sym.Matrix([[1,2],[2,2]]).eigenvals())

# print out the latex block of integrate(cos(x)^2 dx
print(sym.latex(sym.Integral(sym.cos(x)**2, (x, 0, sym.pi))))

