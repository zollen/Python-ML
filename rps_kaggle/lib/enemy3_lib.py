'''
Created on Jan. 13, 2021

@author: zollen
'''


import random
import numpy as np 
from typing import List, Dict, Tuple
from operator import itemgetter
from collections import defaultdict


class MemoryPatterns:
    
    def __init__(self, states=3, min_memory=2, max_memory=20, warmup=5, verbose=False):
        self.min_memory = min_memory
        self.max_memory = max_memory
        self.warmup     = warmup
        self.verbose    = verbose
        self.states     = states
        self.step       = 0
        self.record     = True
        self.last       = None
        self.history = {
            "step":      [],
            "reward":    [],
            "opponent":  [],
            "pattern":   [],
            "action":    []
        }

    
    def __str__(self):
        return "MemoryPatterns(" + str(self.min_memory) + ", " + str(self.max_memory) + ")"
    
    def decide(self):
        return self.agent()
    
    def deposit(self, token):
        self.history['action'].append(token)
        
    def reset(self):
        self.last = None
    
    def myQueue(self, index = None):
        if index == None:
            return np.array(self.history['action'])
            
        return self.history['action'][index]
    
    def opQueue(self, index = None):
        if index == None:
            return np.array(self.history['opponent'])
            
        return self.history['opponent'][index]
    
    def estimate(self):
        self.record = False
        return self.agent()
    
    def add(self, lastOpponentAction):
        self.update_state(lastOpponentAction)

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def agent(self):
         
        if self.step < self.warmup:
            expected = self.random_action()
        else:
            for keys in [ ("opponent", "action"), ("opponent",) ]:
                # history  = self.generate_history(["opponent", "action"])  # "action" must be last
                history  = self.generate_history(["opponent"])  
                memories = self.build_memory(history) 
                patterns = self.find_patterns(history, memories)
                if len(patterns): break
            score, expected, pattern = self.find_best_pattern(patterns)
            self.history['pattern'].append(pattern)    
            if self.verbose:
                print('keys    ', keys)
                print('history ', history)
                print('memories', memories)
                print('patterns', patterns)
                print('score   ', score)
                print('expected', expected)
                print('pattern ', pattern)

        action = (expected + 1) % self.states
        if self.record == True:
            self.history['action'].append(action)
        
        if self.verbose:
            print('action', action)
        self.step += 1
        self.last = int(action)
        return self.last
       
    def random_action(self) -> int:
        return random.randint(0, self.states-1)
  
    def reward(self):
        if len(self.history['action']) <= 0 or len(self.history['opponent']) <= 0:
            return 0
        
        res = (self.history['action'][-1] - self.history['opponent'][-1]) % self.states
        if res == 1:
            return 1
        elif res == 2:
            return -1
        return 0

    def update_state(self, lastOpponentAction):
        if self.step != 0:
            self.history['opponent'].append( lastOpponentAction )
        self.history['step'].append( self.step )
        self.history['reward'].append( self.reward() )

    def generate_history(self, keys: List[str]) -> List[Tuple[int]]:
        history = list(zip(*[ reversed(self.history[key]) for key in keys ]))
        history = list(reversed(history))
        return history 
    
    def build_memory(self, history: List[Tuple[int]]) -> List[ Dict[Tuple[int], List[int]] ]:
        output    = [ dict() ] * self.min_memory
        expecteds = self.generate_history(["opponent"])
        for batch_size in range(self.min_memory, self.max_memory+1):
            if batch_size >= len(history): break  # ignore batch sizes larger than history
            output_batch    = defaultdict(lambda: [0,0,0])
            history_batches  = list(self.batch(history, batch_size+1))
            expected_batches = list(self.batch(expecteds, batch_size+1))
            for _, (pattern, expected_batch) in enumerate(zip(history_batches, expected_batches)):
                previous_pattern = tuple(pattern[:-1])
                expected         = (expected_batch[-1][-1] or 0) % self.states  # assume "action" is always last 
                output_batch[ previous_pattern ][ expected ] += 1
            output.append( dict(output_batch) )
        return output

    
    def find_patterns(self, history: List[Tuple[int]], memories: List[ Dict[Tuple[int], List[int]] ]) -> List[Tuple[float, int, Tuple[int]]]:
        patterns = []
        for n in range(1, self.max_memory+1):
            if n >= len(history): break
                
            pattern = tuple(history[-n:])
            if pattern in memories[n]:
                score    = np.std(memories[n][pattern])
                expected = np.argmax(memories[n][pattern])
                patterns.append( (score, expected, pattern) )
        patterns = sorted(patterns, key=itemgetter(0), reverse=True)
        return patterns
    
    
    def find_best_pattern(self, patterns: List[Tuple[float, int, Tuple[int]]] ) -> Tuple[float, int, Tuple[int]]:
        patterns       = sorted(patterns, key=itemgetter(0), reverse=True)
        for (score, expected, pattern) in patterns:
            break
        else:
            score    = 0.0
            expected = self.random_action()
            pattern  = tuple()
        return score, expected, pattern
    
    
    def get_pattern_scores(self):
        pattern_rewards = defaultdict(list)
        for reward, pattern in self.generate_history(["reward", "pattern"]):
            pattern_rewards[pattern].append( reward )
        pattern_scores = { pattern: np.mean(rewards) for pattern, rewards in pattern_rewards.items() }
        return pattern_scores
                    
            
            
