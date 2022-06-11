'''
Created on Jun. 11, 2022

@author: zollen
@url: https://pymoo.org/algorithms/soo/pso.html
@desc: PSO: Particle Swarm Optimization
    Particle Swarm Optimization was proposed in 1995 by Kennedy and Eberhart [18] based on the simulating of 
    social behavior. The algorithm uses a swarm of particles to guide its search. Each particle has a 
    velocity and is influenced by locally and globally best-found solutions.
'''

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.factory import Ackley
from pymoo.optimize import minimize

problem = Ackley()

algorithm = PSO(max_velocity_rate=0.025)

res = minimize(problem,
               algorithm,
               seed=1,
               save_history=True,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))