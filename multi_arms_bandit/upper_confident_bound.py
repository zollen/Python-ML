'''
Created on Aug 29, 2024

@author: STEPHEN
@url: https://www.youtube.com/watch?v=Ih1xpFANy-4
@url: https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning/
@desc:
The goal of the agent is to maximize the expected reward by selecting the action that has the 
highest action-value by balancing between exploration and exploitation

Qt(a) = sum_of_rewards_when_(a)_take_prior_to_(t) / number_of_times_(a)_taken_prior_to_(t)
Nt(a) = number oftimes action (a) is performed
t = timesteps

A_t = argmax_a ( Qt(a) + c sqrt( ln(t) / Nt(a) )

@algo:
Step 1: At each round n, we consider two numbers of each strategy i:
    Ni(n) - the number of times the strategy i was selected to round n
    Ri(n) - the sum of rewards of the strategy i up to round n
    
Step 2: From these two numbers we compute:
    * the average reward of strategy i up to round n
        ri(n) = Ri(n) / Ni(n)
        
    * the confidence interval [ ri(n) - Δi(n), ri(n) + Δi(n) ] at round n with
    
        Δi(n) = sqrt( (3 * ln(n)) / (2 * Ni(n)) )
        
Step 3: We select the strategy i that has the maximun UCB ri(n) + Δi(n) for the next round 

'''
