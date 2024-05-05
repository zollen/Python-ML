'''
Created on Apr 19, 2024

@author: STEPHEN

Global optimal [0.03937731 0.1612441  0.7991728 ] ==> 12.336679368657787
'''

import numpy as np
from pymoo.problems import get_problem

problem = get_problem("osy")

print(problem.bounds())
print("=========================")
print(problem.ideal_point())
print("=========================")
print(problem.nadir_point())
print("=========================")
print(problem.pareto_front())
