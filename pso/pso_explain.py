'''
Created on Sep. 14, 2020

@author: zollen
'''

"""

Initalize the hyper params (N, c1, c2, Wmin, Wmax, Vmax, and MaxIter)

Initialize the population of N particles
initial values of position and velocity of each particle is randomized with (r1, r2)

PBest - personal best
GBest - Team best

r1, r2 are randomizer arrays between 0 and 1, i.e. each particle has a different random value 
Constants w, c1, c2 are parameters to the PSO algorithm. 
w is the inertia weight constant It is between 0 and 1
V{d} is the original velocity
c1 and c2 are called the cognitive (personal) and the social (team) coefficients. They control how much 
weight should be given between refining the search result of the particle itself and recognizing the 
search result of the swarm.

X{d+1} = X{d} + V{d+1}

          Interia       Cognitive component              Social component 
V{d+1} = w * V{d} +   c1 * r1 (PBest{d} - X{d})   +   c2 * r2 (GBest{d} - X{d})

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