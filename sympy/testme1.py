'''
Created on Jul. 11, 2021

@author: zollen
'''

from sympy import * 

init_printing(use_unicode=True)

print(sqrt(8))
print(sqrt(8).evalf())

x, y = symbols('x y')
expr = x + 2 * y
print(expr)

print(expr + 1)
print(expr - x)

ee = expand(x * expr)
print(ee)

print(factor(ee))


x, y, z, nu = symbols('x t z nu')

# Derivative f(x) = sin(x)
print(diff(sin(x), x))

# integral f'(x) = -sin(x)
print(integrate(-sin(x), x))

# limit x-> 0 sin(x)/x
print(limit(sin(x)/x, x, 0))

# solve x^2 - 2 = 0
print(solve(x**2 - 2, x))

expr = x**3 + 4 * x*y - z
print(expr.subs([(x, 2), (y, 4), (z, 0)]))

# solve the differential equstion y'' - y = e^t
t = symbols('t')
y = Function('y')
print(dsolve(Eq(y(t).diff(t, t) - y(t), exp(t)), y(t)))

# find the eign values of [[1, 2], [2, 2]]
print(Matrix([[1,2],[2,2]]).eigenvals())

# print out the latex block of integrate(cos(x)^2 dx
print(latex(Integral(cos(x)**2, (x, 0, pi))))

x, y, z = symbols('x y z')
M = Matrix([
            [1, 0, 1, 3],
            [2, 3, 4, 7],
            [-1, -3, -3, -4]
            ])
print(M)
print(M.rref())
print(solve_linear_system(M, x, y, z))
print(shape(M))
print(shape(M.T))

M = Matrix([
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
