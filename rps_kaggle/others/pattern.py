'''
Created on Dec. 19, 2020

@author: zollen
@url: https://www.kaggle.com/ilialar/multi-armed-bandit-vs-deterministic-agents
'''
import numpy as np

class agent():
    
    def initial_step(self):
        return np.random.randint(3)
    
    def history_step(self, history):
        return np.random.randint(3)
    
    def step(self, history):
        if len(history) == 0:
            return int(self.initial_step())
        else:
            return int(self.history_step(history))
        
class pattern_matching(agent):
    
    def __init__(self, steps = 3, deterministic = False, counter_strategy = False, init_value = 0.1, decay = 1):
        self.deterministic = deterministic
        self.counter_strategy = counter_strategy
        if counter_strategy:
            self.step_type = 'step' 
        else:
            self.step_type = 'competitorStep'
        self.init_value = init_value
        self.decay = decay
        self.steps = steps
        
    def history_step(self, history):
        if len(history) < self.steps + 1:
            return self.initial_step()
        
        next_step_count = np.zeros(3) + self.init_value
        pattern = [history[i][self.step_type] for i in range(- self.steps, 0)]
        
        for i in range(len(history) - self.steps):
            next_step_count = (next_step_count - self.init_value)/self.decay + self.init_value
            current_pattern = [history[j][self.step_type] for j in range(i, i + self.steps)]
            if np.sum([pattern[j] == current_pattern[j] for j in range(self.steps)]) == self.steps:
                next_step_count[history[i + self.steps][self.step_type]] += 1
        
        if next_step_count.max() == self.init_value:
            return self.initial_step()
        
        if  self.deterministic:
            step = np.argmax(next_step_count)
        else:
            step = np.random.choice([0,1,2], p = next_step_count/next_step_count.sum())
        
        if self.counter_strategy:
            # we predict our step using transition matrix (as competitor can do) and beat probable competitor step
            return (step + 2) % 3 
        else:
            # we just predict competitors step and beat it
            return (step + 1) % 3
        

        
history = []
agent = pattern_matching(3, True, False, decay = 1.001)
    
def multi_armed_bandit_agent (observation, configuration):
    
    global history
        
    # load history
    if observation.step == 0:
        pass
    else:
        history[-1]['competitorStep'] = int(observation.lastOpponentAction)
        
    step = agent.step(history)
    
    if step is None:
        step = np.random.randint(3)
    if history is None:
        history = []
    history.append({'step': step, 'competitorStep': None})
    
    return step


class Observation:
    def __init__(self):
        self.step = 0
        self.lastOpponentAction = 0
        
class Configuration:
    def __init__(self):
        self.signs = 3
        
observation = Observation()
configuration = Configuration()

for rnd in range(0, 100):
    observation.step = rnd
    observation.lastOpponentAction = np.random.randint(0, 3)  
    print("Round [%d] MY PREDICTED MOVE: [%d]" %(rnd + 1, multi_armed_bandit_agent(observation, configuration)))