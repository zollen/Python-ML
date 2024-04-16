'''
Created on Apr 15, 2024

@author: STEPHEN
@url: https://www.youtube.com/watch?v=6SWIfExCN_Q
@url: https://www.youtube.com/watch?v=c-BYVAtgwdk&t=270s
@url: https://ieeexplore.ieee.org/abstract/document/10124217

rn:  random number between [0,1]

1. Chain Food Searching Approach

            α = 2 * r * sqrt(abs(log(r)) )

             p{k,itr} + rn * (Gbest{itr} - p{k,itr} + α * (Gbest{itr} - p{itk,r}) if k = 1
p{k,itr+1} = 
             p{k,itr} + rn * (p{k-1,itr} - p{k,itr} + α * (Gbest{itr} - p{k,itr}) if k > 1
                     
                     

2. Cyclone Food Searching Approach

            β = 2 * e^(rn * (maxitr - itr + 1) / maxitr) * sin(2 * pi * rn)

             Gbest + rn * (Gbest{itr} - p{k,itr} + β * (Gbest{itr} - p{k,itr})  if k = 1
p{k,itr+1} = 
             Gbest + rn * (p{k-1,itr} - p{k,itr}) + β * (Gbest{itr} - p{k,itr}) if k > 1
                     
3. Somersault Food Searching Approach


somersaultfactor: 2
rn{1} and rn{2}: random number between [0,1]

p{k,itr+1} = p{k,itr} + somersaultfactor * (rn{1} * Gbest - rn{2} * p{k,itr} )  i = 1...N




'''

import numpy as np

class MantaRays:
    
    def __init__(self, obj_func, data_func, direction, numOfMantaRays, SF = 2):
        self.obj_func = obj_func
        self.data_func = data_func
        self.direction = direction
        self.numOfMantaRays = numOfMantaRays
        self.SF = SF
        self.mantaRays = self.data_func(self.numOfMantaRays)
        self.best_ray = None
        self.best_score = None
        ind = np.array(range(self.numOfMantaRays))
        self.ind_curr = np.delete(ind, 0)
        self.ind_prev = np.delete(ind, -1)
        
    def fitness(self, X):
        if self.direction == 'max':
            return self.obj_func(X)
        else:
            return self.obj_func(X) * -1
        
    def best(self):
        scores = self.fitness(self.mantaRays)
        ind = np.argmax(scores)
        if self.best_score == None or self.best_score < scores[ind]:
            self.best_ray = self.mantaRays[ind]
            self.best_score = scores[ind]            
    
    def chainSearch(self):
        rn = np.random.rand(self.numOfMantaRays, 3)
        alpha = 2 * rn[:,1] * np.sqrt(np.abs(np.log(rn[:,2])))
        
        mantaRays = np.array(self.mantaRays)

        mantaRays[0] = mantaRays[0] + rn[0, 0] * (self.best_ray - mantaRays[0]) + \
                 alpha[0] * (self.best_ray - mantaRays[0])
                 
        r0 = np.expand_dims(rn[self.ind_curr, 0], axis=1)
        alpha = np.expand_dims(alpha, axis=1)
                 
        mantaRays[self.ind_curr] = mantaRays[self.ind_curr] + r0 * \
                (mantaRays[self.ind_prev] - mantaRays[self.ind_curr]) + \
                alpha[self.ind_curr] * (self.best_ray - mantaRays[self.ind_curr])
                
        return mantaRays
    
    def cycloneSearch(self, rounds, rnd):
        rn = np.random.rand(self.numOfMantaRays, 3)
        beta = 2 * np.e**(rn[:, 1] * (rounds - rnd + 1) / rounds) * np.sin(2 * np.pi, rn[:, 2])
        
        mantaRays = np.array(self.mantaRays)
        
        r0 = np.expand_dims(rn[self.ind_curr, 0], axis=1)
        beta = np.expand_dims(beta, axis=1)
        
        mantaRays[0] = self.best_ray + rn[0, 0] * (self.best_ray - mantaRays[0]) + \
                beta[0] * (self.best_ray - mantaRays[0])
                
        mantaRays[self.ind_curr] = self.best_ray + r0 * \
                (mantaRays[self.ind_prev] - mantaRays[self.ind_curr]) + \
                beta[self.ind_curr] * (self.best_ray - mantaRays[self.ind_curr])
                
        return mantaRays
    
    def randomSearch(self, rounds, rnd):
        rn = np.random.rand(self.numOfMantaRays, 3)
        randRays = self.data_func(self.numOfMantaRays)
        randray = self.data_func(1)
        beta = 2 * np.e**(rn[:, 1] * (rounds - rnd + 1) / rounds) * np.sin(2 * np.pi, rn[:, 2])
        
        mantaRays = np.array(self.mantaRays)
        
        r0 = np.expand_dims(rn[self.ind_curr, 0], axis=1)
        beta = np.expand_dims(beta, axis=1)
        
        mantaRays[0] = randray + rn[0, 0] * (randRays[0] - randray) + \
                beta[0] * (randRays[0] - mantaRays[0])
                
        mantaRays[self.ind_curr] = randray + r0 * \
                (mantaRays[self.ind_prev] - mantaRays[self.ind_curr]) + \
                beta[self.ind_curr] * (randRays[self.ind_curr] - mantaRays[self.ind_curr])
                
        return mantaRays
    
    def somersault(self):
        r1 = np.random.rand(self.numOfMantaRays, 3)
        r2 = np.random.rand(self.numOfMantaRays, 3)
        self.mantaRays = self.mantaRays + self.SF * (r1 * 
                            self.best_ray - r2 * self.mantaRays)        
    
    def start(self, rounds):
        self.best()
        tatic = np.expand_dims(np.random.rand(self.numOfMantaRays), axis=1)
        for rnd in range(rounds):
            chain_m = self.chainSearch()
            cycln_m = self.cycloneSearch(rounds, rnd)
            randn_m = self.randomSearch(rounds, rnd)
            self.mantaRays = np.where(tatic < 0.5, 
                                      chain_m,
                                      np.where(rnd / rounds < tatic, cycln_m, randn_m))
            self.best()
            self.somersault()
            self.best()
        
        return self.best_ray
    
