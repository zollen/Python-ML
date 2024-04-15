'''
Created on Apr 11, 2024

@author: STEPHEN
@url: https://www.youtube.com/watch?v=U9ah51wjvgo
@url: https://www.youtube.com/watch?v=vmQ49mRhGGw
@url: http://www.scholarpedia.org/article/Artificial_bee_colony_algorithm


Initialization Phase 
while Curr_Iter less than Max_Iter
  Employed Bees Phase of all food sources
  Determine the probability of each food source
  Onlooker Bees Phase of all NB food sources
  Memorize the best solution 
  if trail of any good source > limit:
        Scout Bees Phase of exhausted food source
  
  

Initialization Phase
--------------------
    l{i] = lower bound of parameter x{mi}
    u{i} = upper bound of parameter x{mi}
    x{mi}= l{i} + rand(0,1) ∗ (u{i} − l{i})

Employed Bees Phase
--------------------
    Employed bees search for new food sources (υm→) having more nectar within the 
    neighbourhood of the food source (xm→) in their memory.
    
    Food = swarm / 2
    EB   = swarm / 2
    ϕ: random number between [-a, a]
    for m in {1..EB}:                   
        Randomly select a partner(k) such that m <> k
        Randomly select a variable i (among multiple spaces) and modify i-th variable *only*
            v{mi} = x{mi} + ϕ{mi} * (x{mi} - x{ki})
        Bound v{mi}
        Evaluate the object function: f(m) and fitness: fit(m)
        Accept v{mi} if fit(v) > fit(m) and set trail{m} = 0, else increase trail{m} by 1
    
    
    f{m}(x{m}): objective function
    fit{m}(x{m}): the fitness value of the solution
    
                   1/(1 + f{m}(x{m}))        if f{m}(x{m}) >= 0
    fit{m}(x{m}) = 
                   1 + abs(f{m}(x{m})        if f{m}(x{m}) < 0
                   
Onlooker Bees Phase
-------------------
    Generate (swarm / 2) new solutions
    Unemployed bees consist of two groups of bees: onlooker bees and scouts. Employed bees share 
    their food source information with onlooker bees waiting in the hive and then onlooker bees 
    probabilistically choose their food sources depending on this information. 
    
            fit{m}(x{m})
    p{m} = -------------
            SN
            ∑ fit{m}(x{m}) 
           m-1
    * OR *
    p{m} = 0.9 * (fit(i) / max(fit in the Food)) + 0.1
    
    Food = swarm / 2
    OB   = swarm / 2
    ϕ: random number between [-a, a]
    m = 0 : FOOD
    n = 1 : OB
    p(n) : probability above
    r = array of random number between [0, 1]
    limit = Np x D: Number of Food Source x Dimension of the problem
    while all OBs has solutions or the last OB has exhausts all potential food sources: 
        while examining each potential food sources
            if r < p(n):        
                Select a random partner(p) such that n != p
                Randomly select a variable i and modify i-th variable
                    v{mi} = x{mi} + ϕ{mi} * (x{mi} - x{ki})
                Bound v{mi}
                Evaluate the object function: f(m) and fitness: fit(m)
                Accept v{mi} if fit(v) > fit(m) and set trail{m} = 0, else increase trail{m} by 1
                If accepted as food source, then it cannot be accepted as 
                    food source by other OB. so, m = m + 1
        n = n + 1
        Reset n = 1 if the value of n is greater than max(n), then OB look for previous 
            available food source that failed by other OBs.
           
Scout Bees Phase (This phase occur if there is at least one failed solution)
-----------------
    Solution with trail exceeds the limit, then candidates to be discarded
    Only a *single* solution among multiple failed solutions will be discarded in this phase
    A *single* solution with its trail greater than limit is replaced with a new random solution
    Trail counter of newly included solution is reset to 0
    Best solution may have high failure trail because other OBs no longer able to generate a 
        better solution. The best solution in the population could get eliminated
    It is *important* to memorize the best solution before performing scout phase
    
        1. Identify a *single* food source (k) whose trail > limit
        2. Replace the solution with X{k} = lb + r * (ub - lb)
        3. Evaluate the object function: f(k) and fitness: fit(k)
        
''' 

import numpy as np


class Bees:
    
    def __init__(self, fitness, data_func, direction, numOfBees, LB = -5, UB = 5):
        self.fitness = fitness
        self.data_func = data_func
        self.direction = direction
        self.numOfBees = numOfBees
        self.LB = LB
        self.UB = UB
        self.bees = self.data_func(self.numOfBees)
        self.best_score = None
        self.best_bee = None
        
        ind = np.array(range(self.numOfBees))
        np.random.shuffle(ind)
        bb = np.split(ind, 2)
        self.eb_idx = bb[0]
        self.ob_idx = bb[1]
        self.trails = np.expand_dims(np.zeros((self.numOfBees)), axis=1)
        self.scores = np.expand_dims(np.zeros((self.numOfBees)), axis=1)
        
    def bfitness(self, x):
        scores = self.fitness(x)   
        if self.direction == 'max':
            return np.expand_dims(scores, axis=1)
        else:
            return np.expand_dims(scores * -1, axis=1)
        
    def bound(self, X):
        X = np.where(X > self.LB, X, self.LB)
        X = np.where(X < self.UB, X, self.UB)
        return X
    
    def new_solution(self, P, X):
        theta = np.random.uniform(-1, 1, size=(X.shape[0], 2))
        j = np.random.choice(range(X.shape[1]))
        X_new = np.array(X) 
        X_new[:,j] = self.bound(X[:,j] + theta[:,0] * (X[:,j] - P[:,j]) + theta[:,1] * 0.1)
        return X_new
    
    def shuffle(self, X):
        P = np.array(X)
        np.random.shuffle(P)
        return P
    
    def probabilities(self, X):
        smallest = np.min(X)
        extra = 0
        if smallest < 0:
            extra = smallest + 1
        Xp = X + extra
        return Xp / np.sum(Xp)
         
    def update(self, fit_new, fit_now, idx, eb_now, eb_new):
        self.bees[idx] = np.where(fit_now >= fit_new, eb_now, eb_new)
        self.trails[idx] = np.where(fit_now >= fit_new, self.trails[idx] + 1, 0)
        self.scores[idx] = np.where(fit_now >= fit_new, fit_now, fit_new)
    
    def employedBee(self):
        eb_now = self.bees[self.eb_idx]
        eb_new = self.new_solution(self.shuffle(eb_now), eb_now)
        fit_now = self.bfitness(eb_now)
        fit_new = self.bfitness(eb_new)
        self.update(fit_new, fit_now, self.eb_idx, eb_now, eb_new)
       
    def onlookerBee(self):
        p = np.random.choice(self.eb_idx, self.ob_idx.shape[0], 
                             p=self.probabilities(self.scores[self.eb_idx].squeeze()))
        eb_now = self.bees[p]
        ob_now = self.bees[self.ob_idx]
        ob_new = self.new_solution(ob_now, eb_now)
        fit_now = self.scores
        fit_new = self.bfitness(ob_new)
        
        eb_now = np.hstack((np.expand_dims(p, axis=1), eb_now))
        while p.size > 0:
            res = np.unique(p, return_index=True) 
            self.update(fit_new[res[1]], fit_now[res[1]], 
                        eb_now[res[1]][:,0].astype('int64'), 
                        eb_now[res[1]][:,1:], ob_new[res[1]])
            p = np.delete(p, res[1])
            eb_now = np.delete(eb_now, res[1], axis=0)
            ob_new = np.delete(ob_new, res[1], axis=0)
       
    def scoutBee(self):
        self.bees[self.ob_idx] = self.data_func(self.ob_idx.size)
        self.trails[:] = 0
        
    def best(self):
        idx = np.argmax(self.scores)
        score = self.scores[idx].squeeze()
        if self.best_score == None:
            self.best_score = score
            self.best_bee = self.bees[idx]
        else:
            if self.best_score < score:
                self.best_score = score
                self.best_bee = self.bees[idx]
    
    def start(self, rounds):
        for _ in range(rounds):
            self.employedBee()
            self.onlookerBee()
            self.best()
            self.scoutBee()

        return self.best_bee