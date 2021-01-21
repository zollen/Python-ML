'''
Created on Jan. 20, 2021

@author: zollen
'''

import numpy as np
import itertools
import pprint


class StatsAgency(BaseAgent):
        
    def __init__(self, scorers, agents, states = 3, random_threshold = -10):
        super().__init__(states, 0, 0, None)
        self.scorers = scorers
        self.agents = agents
        self.random_threshold = random_threshold
        self.rnd = 0
        self.combos = self.generate()
    
    def generate(self):
        seq = [ x for x in range(len(self.scorers)) ]
        llist = []
        for ll in range(0, len(seq)):
            for subset in itertools.combinations(seq, ll):
                if len(subset) > 0:
                    llist.append(subset)           
        llist.append(tuple(seq))
        return llist
        
    def __str__(self): 
        return "StatsAgency()"
    
    def add(self, token):
        super().add(token)
        for agent, _, _ in self.agents:
            agent.add(token)
            
    def submit(self, token):
        super().submit(token)
        for agent, _, _ in self.agents:
            agent.deposit(token)
        return token
            
    def reward(self, mymove, opmove):
        return -1 if (mymove - opmove) % self.states == 2 else (mymove - opmove)
    
    def calculate(self):
        
        default_scores = [ 0 ] * len(self.scorers)
        current_scores = []
        for scorer in self.scorers:
            score = scorer.normalize(scorer.calculate())
            print("SCORE: ", score)
            current_scores.append(score)
            default_scores = [ x + y for x, y in zip(default_scores, score) ]
            
        
                       
        results1 = {}
        results2 = {}
        for combo in self.combos:
            scores = [ 0 ] * len(self.agents)
            winOnly = True
            winAndEvenOnly = True
            for entry in combo:
                res = (self.agents[np.argmax(current_scores[entry])][1][-1] - self.opponent[-1]) % self.states
                flag = 0.5 if res == 0 else 1
                if res == 0:
                    winOnly = False
                if res == 2:
                    winOnly = False
                    winAndEvenOnly = False
                    break
                scores = [ x + (y * flag) for x, y in zip(scores, current_scores[entry]) ]
            if winOnly == True:
                results1[combo] = scores
            if winAndEvenOnly == True:
                results2[combo] = scores
                
        import pprint
        pp = pprint.PrettyPrinter(indent=3)         
        pp.pprint(results1)
        pp.pprint(results2)
        
        if len(results1) <= 0 and len(results2) > 0:
            results1 = results2
            
        if len(results1) <= 0:
            results1['default'] = default_scores
        
        best_score = -1
        best_combo = None        
        for combo, scores in results1.items():
            score = np.std(scores)
            if score > best_score:
                best_score = score
                best_combo = combo
        
        return results1[best_combo]
    
    
    def decide(self):
        
        final_scores = [0] * len(self.scorers)
        
        if self.rnd > 0:
            for _, predicted, outcome in self.agents:
                outcome.append(self.reward(predicted[-1], self.opponent[-1]))
                        
            final_scores = self.calculate()
                
        for agent, predicted, _ in self.agents:
            try :
                predicted.append(agent.estimate())
            except:
                predicted.append(np.random.randint(self.states))
            
        
        self.rnd += 1
                              
        return self.submit(self.agents[np.argmax(final_scores)][1][-1])
