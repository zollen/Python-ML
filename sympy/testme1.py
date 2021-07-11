'''
Created on Jul. 11, 2021

@author: zollen
'''

import sympy as sy

sy.init_printing(use_unicode=True)

print(sy.sqrt(8))
print(sy.sqrt(8).evalf())

x, y = sy.symbols('x y')
expr = x + 2 * y
print(expr)

print(expr + 1)
print(expr - x)

ee = sy.expand(x * expr)
print(ee)

print(sy.factor(ee))


x, y, z, nu = sy.symbols('x t z nu')

# Derivative f(x) = sin(x)
print(sy.diff(sy.sin(x), x))

# integral f'(x) = -sin(x)
print(sy.integrate(-sy.sin(x), x))

# limit x-> 0 sin(x)/x
print(sy.limit(sy.sin(x)/x, x, 0))

# solve x^2 - 2 = 0
print(sy.solve(x**2 - 2, x))

expr = x**3 + 4 * x*y - z
print(expr.subs([(x, 2), (y, 4), (z, 0)]))

# solve the differential equstion y'' - y = e^t
t = sy.symbols('t')
y = sy.Function('y')
print(sy.dsolve(sy.Eq(y(t).diff(t, t) - y(t), sy.exp(t)), y(t)))

# find the eign values of [[1, 2], [2, 2]]
print(sy.Matrix([[1,2],[2,2]]).eigenvals())

# print out the latex block of integrate(cos(x)^2 dx
print(sy.latex(sy.Integral(sy.cos(x)**2, (x, 0, sy.pi))))

x, y, z = sy.symbols('x y z')
M = sy.Matrix([
            [1, 0, 1, 3],
            [2, 3, 4, 7],
            [-1, -3, -3, -4]
            ])
print(M)
print(M.rref())
print(sy.solve_linear_system(M, x, y, z))
print(sy.shape(M))
print(sy.shape(M.T))

M = sy.Matrix([
            [3, -2, 4 , -2],
            [5,  3, -3, -2],
            [5, -2, 2,  -2],
            [5, -2, -3,  3]
            ])
print("=====================")
print(M.det())
print("=====================")
print(M.eigenvals())
print("====================")
print(M.eigenvects())
print("==========PDP^-1==========")
print(M.diagonalize())
