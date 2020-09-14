'''
Created on Sep. 14, 2020

@author: zollen
'''

"""

Initalize the hyper params (N, c1, c2, Wmin, Wmax, Vmax, and MaxIter)

Initialize the population of N particles

PBest - personal best
GBest - Team best

r1, r2 are randomizer between 0 and 1
X{d+1} = X{d} + V{d+1}
V{d+1} = w * V{d} + c1 * r1 (PBest{d} - X{d}) + c2 * r2 (GBest{d} - X{d})

do
    for each particle
        calculate the objective of the particle
        Update PBest if required
        Update GBest if required
    end for
    
    update the inertia weight
    for each particle
        Update the velocity (V)
        Update the position (X)
    end for
    
while the end condition is not satisifed

return GBest as the best estimation of the global optimum



"""