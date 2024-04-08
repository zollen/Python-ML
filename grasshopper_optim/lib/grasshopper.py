'''
Created on Apr 6, 2024

@author: STEPHEN
@url: https://www.baeldung.com/cs/grasshopper-optimization-algorithm
@url: https://www.youtube.com/watch?v=nyfrL_W4ri0
@desc:

Position of swarm
-----------------
S: the social interaction between the solution and the other grasshoppers
G: the gravity force on the solution
A: the wind advection
r1,r2,r3: random factors

    X(i) = r1 * S(i) + r2 * G(i) + r3 * A(i)
    
Force of social interaction
----------------------------

    Unit vector = vector / Vector_Magniude = a / |a|

    d(i,j) = | x(j) - x(i) |  # scalar value of absolute length between x(j) and x(i)

    unitvector_d(i,j) = | x(j) - x(i) | / d(i,j)
    
    
    Social force s(r): l is the attractive length scale and f is the intensity of attraction
    When the distance between two grasshoppers is between {[0, 2.079]}, repulsion occurs, 
    and when the distance between two grasshoppers is {2.079}, neither attraction nor 
    repulsion occurs, which form a comfort zone. When the distance exceeds {2.079}, 
    the attraction force increases, then progressively decreases until it reaches {4}.

    The function {s} fails to apply forces between grasshoppers when the distance between 
    them is larger than {10}. In order to solve this problem, we map the distance of 
    grasshoppers in the interval {[1,4]}
    
    f = 0.5, l = 1.5
    
    s(r) = f * e^(-r/l) - e^(-r)   

           N
    S(i) = Σ  s( d(i,j) ) * unitvector_d(i,j)  where i <> j
          j=1
          
Force of gravity
----------------

    -g: gravitational constant
    b(g): is a unit vector toward the center of Earth
    
    G(i) = -g * b(g)
    
Force of Wind direction
-----------------------

    The nymph and adulthood grasshoppers are correlated with the wind direction A(i).
    
    u: the drift constant
    c(w): is a unit vector toward the wind direction
    
    A(i) = u * c(w)
    
    


Grasshopper Position
====================

           N
    X(i) = Σ s( d(i,j) ) * k(i,j) - g * b(g) + u * c(w)  where i <> j
          j=1
          
    X(i) = Σ s(|x(j) - x(i)|) * |x(j) - x(i)| / d(i,j) - g * b(g) + u * c(w) where i <> j
    
Above equations require modification to prevent agents from quickly reaching their comfort
zone and the swarn from failing to converge to the target location (global optimum)
    
                N      UB(d) - LB(d) 
    X(i,d) = c( Σ c * --------------- * s(|x(j,d) - x(i,d)|) * |x(j) - x(i)| / d(i,j) ) + best_solution
               j=1            2
    
    where i <> j
When gravity(G) = 0, wind(A) is the best_solution in d-th dimension, and UB(d) and LB(d) are the upper and
lower bounds in the d-th dimension.

The parameter c is the decreasing coefficient. It is in charge of decreasing the comfort
zone, repulsion zone, and the attraction zone. In order to balance the exploitation and
exploration phases, the coefficient c decrease according to the number of iterations.
    
    c = c_max + curr_iter ( c_max - c_min) / max_iter
    
    

GOA Algorithm
============

Parameter Initialization: iter, c_max, c_min, l and f
Swarm Initialization X(i) {i=1...n}
Compute fitness value for each grasshopper (search agent)
Select the best solution among all (best search agent)

while curr_iter < max_iter:

    c = c_max - curr_iter (c_max - c_min) / max_citer
    
    for each grasshopper:
        
        Normalize distance between grasshopper in the range between [1,4]
        c( Σ c * ( (UB(d) - LB(d)) / 2 ) * s(|x(j,d) - x(i,d)|) * |x(j) - x(i)| / d(i,j) ) + best_solution
        Bring current grasshopper back if it goes outside the boundaries
        
    Update the best solution if there is a better one.
    
Return the best solution

'''

import numpy as np


class Grasshoppers:
    
    def __init__(self, fitness, data_func, direction, numOfGrasshoppers, LB = -5, UB = 5, c_max = 1, c_min = 0.00004, f = 0.5, l = 1.5, LS = 1, US = 4):
        self.fitness = fitness
        self.data_func = data_func
        self.direction = direction
        self.numOfGrasshoppers = numOfGrasshoppers
        self.LB = LB
        self.UB = UB
        self.cMax = c_max
        self.cMin = c_min
        self.F = f
        self.L = l
        self.LS = LS
        self.US = US
        self.grasshoppers = self.data_func(numOfGrasshoppers)
    
    def unit_vector(self, X):
        return X / np.expand_dims(self.unit_length(X), axis=2)
    
    def unit_length(self, X):
        return np.linalg.norm(X, axis=2)
    
    def normalize(self, X):
        x_max = np.expand_dims(np.max(X, axis=1), axis=1)
        x_min = np.expand_dims(np.min(X, axis=1), axis=1)
        return (X - x_min) / (x_max - x_min) * (self.US - self.LS) + self.LS
    
    def social_factor(self, X):
        return self.F * np.power(np.e, -X/self.L) - np.power(np.e, -X)
    
    def gravity_factor(self):
        return 0
    
    def wind_factor(self):
        return 0
    
    def coefficient(self, rnd, rounds):
        return self.cMax - rnd * (self.cMax - self.cMin) / rounds
    
    def best_solution(self):
        if self.direction == 'max':
            return self.grasshoppers[np.argmax(self.fitness(self.grasshoppers))]
        else:
            return self.grasshoppers[np.argmin(self.fitness(self.grasshoppers))]
    
    def differences(self):
        gh = np.reshape(
                np.hsplit(
                    np.tile(self.grasshoppers, self.numOfGrasshoppers), 
                            self.numOfGrasshoppers),
                                  (self.numOfGrasshoppers, self.grasshoppers.shape[0], 
                                   self.grasshoppers.shape[1]))
        gb = np.expand_dims(self.grasshoppers, axis=1)
        return np.abs(gh - gb)
    
    def move(self, c, best, X):
        result = c * ((self.UB - self.LB) / 2) * np.expand_dims(self.social_factor(self.unit_length(X)), axis=2) * self.unit_vector(X)       
        return c * np.sum(result, axis=1) + self.gravity_factor() + self.wind_factor() + best
    
    def start(self, rounds):
        for rnd in range(rounds):
            best = self.best_solution()
            c = self.coefficient(rnd, rounds)
            N = self.normalize(self.differences())
            self.grasshoppers = self.move(c, best, N)
            
        return self.best_solution()
    

