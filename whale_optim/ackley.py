'''
Created on Apr 5, 2024

@author: STEPHEN
@best: minimize problem
    Best solution found:
    X = [ 7.24572396e-08 -4.41993909e-08  1.28265415e-08  2.02157667e-08
     -4.37594377e-08  7.44164226e-08 -5.91928395e-08  4.65969605e-08
     -4.46756245e-08  5.22876418e-08]
    F = [2.02522252e-07]
    
'''


from pymoo.operators.sampling.lhs import LHS
from pymoo.problems import get_problem
from whale_optim.lib.whales import Whales


problem = get_problem("ackley", n_var=10)
sampling = LHS()

def ackley10d(X):
    return problem.evaluate(X).squeeze()

def fitness(X):
    return ackley10d(X)

def data(n):
    return sampling(problem, n).get("X")



moths = Whales(fitness, data, 'min', 1000)    
best = moths.start(60)
print("Global optimal at f({}) ==> {}".format(best, ackley10d(best)))