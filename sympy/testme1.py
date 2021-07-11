'''
Created on Jul. 11, 2021

@author: zollen
'''

from sympy import * 

print(sqrt(8))

x, y = symbols('x y')
expr = x + 2 * y
print(expr)

print(expr + 1)
print(expr - x)

ee = expand(x * expr)
print(ee)

print(factor(ee))


x, y, z, nu = symbols('x t z nu')
init_printing(use_unicode=True)

# Derivative f(x) = sin(x)
print(diff(sin(x), x))

# integral f'(x) = -sin(x)
print(integrate(-sin(x), x))

# limit x-> 0 sin(x)/x
print(limit(sin(x)/x, x, 0))

# solve x^2 - 2 = 0
print(solve(x**2 - 2, x))

# solve the differential equstion y'' - y = e^t
t = symbols('t')
y = Function('y')
print(dsolve(Eq(y(t).diff(t, t) - y(t), exp(t)), y(t)))

# find the eign values of [[1, 2], [2, 2]]
print(Matrix([[1,2],[2,2]]).eigenvals())

# print out the latex block of integrate(cos(x)^2 dx
print(latex(Integral(cos(x)**2, (x, 0, pi))))