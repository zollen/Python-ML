'''
Created on Mar 30, 2024

@author: STEPHEN
@url: https://pymoo.org/algorithms/soo/de.html
@best: minimize problem
    Best solution found:
    X = [ 7.24572396e-08 -4.41993909e-08  1.28265415e-08  2.02157667e-08
     -4.37594377e-08  7.44164226e-08 -5.91928395e-08  4.65969605e-08
     -4.46756245e-08  5.22876418e-08]
    F = [2.02522252e-07]
'''


from pymoo.operators.sampling.lhs import LHS
from pymoo.problems import get_problem
from grey_wolf.lib.grey_wolf import WolfPack, MutatedWolfPack, SuperWolfPack


problem = get_problem("ackley", n_var=10)
sampling = LHS()

def ackley10d(X):
    X = problem.evaluate(X)
    return X.squeeze()

def fitness(X):
    return problem.evaluate(X)

def data(n):
    return sampling(problem, n).get("X")



pack = WolfPack(ackley10d, data, 'min', 1000)  
alpha = pack.hunt(50)
print("Global optimal {} ==> {}".format(alpha, ackley10d(alpha)))

pack = MutatedWolfPack(ackley10d, fitness, data, 'min', 1000)  
alpha = pack.hunt(50)
print("Global optimal {} ==> {}".format(alpha, ackley10d(alpha)))
  
pack = SuperWolfPack(ackley10d, fitness, data, 'min', 1000)  
alpha = pack.hunt(50)
print("Global optimal {} ==> {}".format(alpha, ackley10d(alpha)))
