'''
Created on Apr 17, 2024

@author: STEPHEN
@url: https://link.springer.com/article/10.1007/s11831-023-09887-z

Initialize parameters
Initialize population
Evaluate the fitness of the population

while curr_iter < max_iter:
    select the sparrows(75% of the population) with the best position x_gbest
    select the sparrows(25% of the population) with the worst position x_worst
    Divide the population into producers and scroungers
    for i in all(producers):
    
        α and r are random variable between [0,1]
        ST belongs to [0.5,1] are represent the alarm and the safety threshold
        Q is a random number belongs to the normal distribution
        L is a one-dimensional vector with the length of search space where each element of 
            L is assigned with 1.
        
                      x{i,j,iter} * e^(-i / α * max_iter)    if r <= ST
        x{i,j,iter+1} = 
                      x{i,j,iter} + Q * L                    if r > ST
                      
    for i in all(scroungers):
    
        A is one-dimensional vector with the length of D, each element of A is assigned a 
            random value between [-1, 1]. 
        A_+ = A_transpose * (A * A_transpose)^(-1)
        x_pbest{j} is the best location obtains so far for the j producer
        i: energy/fitness level for checking if it is above or below average of all fitness
        
                      #low energy/starving
                      Q * e^(x_worst{j} - x{i,j,iter} / i^2)     if i > N / 2   
        x{i,j,iter+1} =
                      #high energy
                      x_pbest{j} - abs(x{i,j,iter} - x_pbest{j}) * A_+ * L  if i <= N / 2  
                      
    for i in all(scouters):
    
        β is the step size control parameter, and it assigned a random number that follows a 
            normal distribution with a mean value between 0 and a variance of 1
        K is a assigned a random number between [− 1, 1].
        p is a small random value to avoid division by zero
         
                      x_gbest{j} + β * abs(x{i,j,iter} - x_gbest{j})   f(x{i,iter}) > f{x_gbest)
        x{i,j,iter+1} = 
                      x{i,j,iter) + K * (x{i,j,iter} - x_worst{j}) 
                      ------------------------------------------     f(x{i,iter}) = f{x_gbest)
                             (f(x{i,iter} - f(x_worst) + p)    
                      
    for i in all(population):
        if f(x{i,iter}+1)) < f(x{i,iter)):
            x{i,iter) = x{i,iter+1)    # archives the current locations of sparrows
        if f(x{i,iter+1)) < f(x_gbest):
            x_gbest = x{i}(g + 1)    # update the position of the best sparrow
            

return the best solution x_gbest
'''
import numpy as np


class Sparrows:
    
    def __init__(self, obj_func, data_func, direction, numOfSparrows, numOfProducers = 0.75, numOfScouters = 0.1, L=1, ST = 0.5, LB=-5, UB=5):
        self.obj_func = obj_func
        self.data_func = data_func
        self.direction = direction
        self.numOfSparrows = numOfSparrows
        self.numOfProducers = int(self.numOfSparrows * numOfProducers)
        self.numOfScouters = int(self.numOfSparrows * numOfScouters)
        self.ST = ST
        self.L = L
        self.LB = LB
        self.UB = UB
        self.sparrows = self.data_func(self.numOfSparrows)
        self.producers = []
        self.scroungers = []
        self.scouters = []
        self.best_sparrow = None
        self.best_score = None
        self.worst_sparrow = None
        self.worst_score = None
    
    def fitness(self, X):
        if self.direction == 'max':
            return self.obj_func(X)
        else:
            return self.obj_func(X) * -1
        
    def bound(self, X):
        X = np.where(X > self.LB, X, self.LB)
        X = np.where(X < self.UB, X, self.UB)
        return X
        
    def selection(self):
        scores = self.fitness(self.sparrows)
        self.producers = np.argpartition(scores, -self.numOfProducers)[-self.numOfProducers:]
        self.scroungers = np.delete(np.array(range(self.numOfSparrows)), self.producers)
        ind = np.array(range(self.numOfScouters))
        np.random.shuffle(ind)
        self.scouters = ind[:self.numOfScouters]
    
    def best(self):
        scores = self.fitness(self.sparrows)
        ind = np.argmax(scores)
        if self.best_score == None or self.best_score < scores[ind]:
            self.best_sparrow = self.sparrows[ind]
            self.best_score = scores[ind]
            
    def worst(self):
        scores = self.fitness(self.sparrows)
        ind = np.argmin(scores)
        if self.worst_score == None or self.worst_score > scores[ind]:
            self.worst_sparrow = self.sparrows[ind]
            self.worst_score = scores[ind]
    
    def update_producers(self, rounds):
        producers = self.sparrows[self.producers] 
        r = np.expand_dims(np.random.rand(self.numOfProducers), axis=1)
        alpha = np.random.rand(producers.shape[0], producers.shape[1])
        Q = np.random.uniform(-1, 1, (producers.shape[0], producers.shape[1]))
        moves1 = producers * np.e**(-r / (alpha * rounds))
        moves2 = producers + Q * self.L
        return self.bound(np.where(r < self.ST, moves1, moves2))

    def update_scroungers(self):
        scroungers = self.sparrows[self.scroungers]
        scores = self.fitness(scroungers)
        avg = (np.max(scores) - np.min(scores)) / 2
        scores = np.expand_dims(scores, axis=1)
        r = np.expand_dims(np.random.rand(scroungers.shape[0]), axis=1) + 2
        Q = np.random.uniform(-1, 1, (scroungers.shape[0], scroungers.shape[1]))
        A = np.random.uniform(-1, 1, scroungers.shape[0])
        At = np.transpose(A)
        Ap = np.expand_dims(At * np.sqrt(A * At), axis=1)
        moves1 = Q * np.e**((self.worst_sparrow - scroungers) / (r**2))
        moves2 = self.best_sparrow - np.abs(scroungers - self.best_sparrow) * Ap * self.L
        return self.bound(np.where(scores > avg, moves1, moves2))
    
    def update_scouters(self, rounds, rnd):
        scouters = self.sparrows[self.scouters]
        scores = np.expand_dims(self.fitness(scouters), axis=1)
        best = self.fitness(np.expand_dims(self.best_sparrow, axis=0))
        worst = self.fitness(np.expand_dims(self.worst_sparrow, axis=0))
        beta =  rounds - ((rnd + 1) / (rounds + 2))
        K = np.random.uniform(-1, 1, (scouters.shape[0], scouters.shape[1]))
        moves1 = self.best_sparrow + beta * np.abs(scouters - self.best_sparrow)
        moves2 = scouters + K * ((scouters - self.worst_sparrow)/(scores - worst + 1))
        return self.bound(np.where(scores > best, 
                                moves1,
                                np.where(scores == best,
                                        moves2,
                                        scouters)))
    
    def update(self, producers, scroungers, scouters):
        scores = np.expand_dims(self.fitness(self.sparrows), axis=1)
        nscores = np.expand_dims(self.fitness(producers), axis=1)
        self.sparrows[self.producers] = np.where(scores[self.producers] < nscores, 
                                                 producers, self.sparrows[self.producers])
        nscores =  np.expand_dims(self.fitness(scroungers), axis=1)
        self.sparrows[self.scroungers] = np.where(scores[self.scroungers] < nscores, 
                                                  scroungers, self.sparrows[self.scroungers])
        nscores =  np.expand_dims(self.fitness(scouters), axis=1)
        self.sparrows[self.scouters] = np.where(scores[self.scouters] < nscores,
                                                scouters, self.sparrows[self.scouters])
        
    def start(self, rounds):
        for rnd in range(rounds):
            self.best()
            self.worst()
            self.selection()
            producers = self.update_producers(rounds)
            scroungers = self.update_scroungers()
            scouters = self.update_scouters(rounds, rnd)     
            self.update(producers, scroungers, scouters)

        self.best()
        return self.best_sparrow
