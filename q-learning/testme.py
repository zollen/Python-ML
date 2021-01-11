'''
Created on Jan. 11, 2021

@author: zollen
@url: https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56


What is q-learning?
-------------------
Q-learning is an off policy reinforcement learning algorithm that seeks to find the best action 
to take given the current state. It’s considered *off-policy* because the q-learning function 
learns from actions that are outside the current policy, like taking random actions, and 
therefore a policy isn’t needed. More specifically, q-learning seeks to learn a policy that 
maximizes the total reward.

Q-Table
-------
When q-learning is performed we create what’s called a q-table or matrix that follows the 
shape of [state, action] and we initialize our values to zero. We then update and store our 
q-values after an episode. This q-table becomes a reference table for our agent to select the 
best action based on the q-value.

Here are the 3 basic steps:
1. Agent starts in a state (s1) takes an action (a1) and receives a reward (r1)
2. Agent selects action by referencing Q-table with highest value (max) OR by random (epsilon, ε)
3. Update q-values

Learning Rate: 
lr or learning rate, often referred to as alpha or α, can simply be defined as how much you 
accept the new value vs the old value. Above we are taking the difference between new and old 
and then multiplying that value by the learning rate. This value then gets added to our previous 
q-value which essentially moves it in the direction of our latest update.

Gamma: 
gamma or γ is a discount factor. It’s used to balance immediate and future reward. From our 
update rule above you can see that we apply the discount to the future reward. Typically this 
value can range anywhere from 0.8 to 0.99.

Reward: 
reward is the value received after completing a certain action at a given state. A reward 
can happen at any given time step or only at the terminal time step.


Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])
'''

import numpy as np

## qtable = np.zeros((state_size, action_size))
## pretending we have 100 states, simple state transition i => i + 1
qtable = np.zeros((100 + 1, 3))
reward = np.array([ 0.3, 0.3, 0.3 ])

learningRate = 0.1
gamma = 0.9
epsilon = 0.1


for rnd in range(0, 100):
    
    if np.random.uniform(0, 1) <= epsilon:
        # explore!
        best_action = np.random.randint(0, 3)
    else:
        # exploit! 
        # The probabilty/scores of the next possible actions are already calculated or pre-defined.
        # No need to be worried by the q learning method
        # Hence, *off policy*
        qtable[rnd + 1, 0] = np.random.uniform(0, 1)  
        qtable[rnd + 1, 1] = np.random.uniform(0, 1) 
        qtable[rnd + 1, 2] = np.random.uniform(0, 1) 
        # Find the best action in the next state
        best_action = np.max(qtable[rnd + 1, :])
    
    # reward would change in each round based on the new state
    reward[0] = np.random.uniform(0, 1)  
    reward[1] = np.random.uniform(0, 1)  
    reward[2] = np.random.uniform(0, 1)  
    
    qtable[rnd, :] = qtable[rnd, :] + learningRate * (reward * gamma * best_action - qtable[rnd, :])
    print("Round: [", rnd + 1, "] ==> ", qtable[rnd, :])
