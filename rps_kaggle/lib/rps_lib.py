'''
Created on Dec. 15, 2020

@author: zollen
'''
import numpy as np
import time

class BaseAgent:
    
    def __init__(self, states = 3, window = 3, counter = None):
        np.random.seed(int(round(time.time())))
        self.mines = np.array([])
        self.opponent = np.array([])
        self.results = np.array([])
        self.states = states
        self.window = window
        self.counter = counter
        
    def __str__(self):
        if self.counter == None:
            name = self.__class__.__name__
        else:
            name = self.__class__.__name__ + "(" + self.counter.__class__.__name__ + ")"
        return name
        
    def add(self, token):
        self.opponent = np.append(self.opponent, token)
        
    def submit(self, token):
        if self.counter != None and len(self.opponent) > self.window + 2:
            token = self.counter.predict(token)
            
        self.mines = np.append(self.mines, token)
        
        return token
    
    def random(self):
        return np.random.randint(0, self.states)
        
    def predict(self):
        pass


class AgressiveCounterMover:
    
    def __init__(self, agent, states = 3, interval = 5):
        self.states = states
        self.interval = interval
        self.agent = agent
        self.wincounter = 0
        self.evencounter = 0
        
    def won(self, index):
        
        res = self.agent.mines[index] - self.agent.opponent[index]  
        if res == 0:
            return 0
        elif res == 1:
            return 1
        elif res == -1:
            return -1
        elif res == 2:
            return -1
        else:             
            return 1
        
    def predict(self, token):
      
        if self.wincounter <= 0 and self.won(-1) < 0 and self.won(-2) < 0:
            self.wincounter = np.random.randint(1, self.interval + 1)
            
        if self.evencounter <= 0 and self.won(-1) == 0 and self.won(-2) == 0:
            self.evencounter = np.random.randint(1, self.interval + 1)
                
        if self.wincounter > 0:
            self.wincounter = self.wincounter - 1             
            return (token + 2) % self.states
      
        if self.evencounter > 0:
            self.evencounter = self.evencounter - 1
            return (token + 1) % self.states

        return token
    
class RandomCounterMover:
    
    def __init__(self, agent, states = 3, interval = 5):
        self.states = states
        self.interval = interval
        self.agent = agent
        self.wincounter = 0
        self.evencounter = 0
        
    def won(self, index):
        
        res = self.agent.mines[index] - self.agent.opponent[index]  
        if res == 0:
            return 0
        elif res == 1:
            return 1
        elif res == -1:
            return -1
        elif res == 2:
            return -1
        else:             
            return 1
        
    def predict(self, token):
      
        if self.wincounter <= 0 and self.won(-1) < 0 and self.won(-2) < 0:
            self.wincounter = np.random.randint(1, self.interval + 1)
                
        if self.wincounter > 0:
            self.wincounter = self.wincounter - 1
            if np.random.randint(0, 2) == 0:
                return (token + 2) % self.states
            else:
                choices = [0, 1, 2]
                choices.remove(token)
                return np.random.choice(choices)

        return token    

class StandardCounterMover:
    
    def __init__(self, agent, states = 3, interval = 5):
        self.states = states
        self.interval = interval
        self.agent = agent
        self.enable = 0
        
    def won(self, index):
        
        res = self.agent.mines[index] - self.agent.opponent[index]  
        if res == 0:
            return 0
        elif res == 1:
            return 1
        elif res == -1:
            return -1
        elif res == 2:
            return -1
        else:             
            return 1
        
    def predict(self, token):
      
        if self.enable <= 0 and self.won(-1) < 0 and self.won(-2) < 0:
            self.enable = np.random.randint(1, self.interval + 1)
                
        if self.enable > 0:
            self.enable = self.enable - 1
            return (token + 2) % self.states

        return token
    
class ClassifierHolder:
    
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.current = None
        
    def fit(self, X, y):
        self.current = self.classifiers[np.random.randint(0, len(self.classifiers))]
        self.current.fit(X, y)
        
    def predict(self, X):
        return self.current.predict(X)


         
class Classifier(BaseAgent):
    
    def __init__(self, classifier, states = 3, window = 3, delay_process = 5, counter = None):
        super().__init__(states, window, counter)
        self.classifier = classifier
        self.delayProcess = delay_process
        self.row = 0
        self.data = np.zeros(shape = (1100, self.window * 2))
        
    def __str__(self):
        if self.counter == None:
            name = self.classifier.__class__.__name__
        else:
            name = self.classifier.__class__.__name__ + "(" + self.counter.__class__.__name__ + ")"
        return name
   
    def add(self, token):
        super().add(token)
        
        if len(self.opponent) >= self.window + 1: 
            self.buildtrain() 
    
    def buildtrain(self):
        self.data[self.row] = self.mines[self.row:self.row+self.window].tolist() + self.opponent[self.row:self.row+self.window].tolist()
        self.results = np.append(self.results, self.opponent[-1])    
        self.row = self.row + 1
        
    def buildtest(self):
        return np.array(self.mines[-self.window:].tolist() + self.opponent[-self.window:].tolist()).reshape(1, -1)
  
    def predict(self):
        
        if len(self.opponent) > self.window + self.delayProcess + 1:
            self.classifier.fit(self.data[:self.row], self.results)  
            return self.submit((int(self.classifier.predict(self.buildtest()).item()) + 1) % self.states)
            
        return self.submit(self.random())
    
    
class MClassifier(Classifier):
    
    def __init__(self, classifier, states = 3, window = 3, delay_process = 5, counter = None):
        super().__init(classifier, states, window, delay_process, counter)
        
    def buildtrain(self):
        pass
    
    def buildtest(self):
        pass
    
        
        
class Randomer(BaseAgent):
    
    def __init__(self):
        super().__int__()
        pass
        
    def predict(self):
        return self.random()
        

    
'''
High order Markov. It holds a sequence of transitions (as oppose to just a single transition 
in the transition matrix
'''
class NMarkov(BaseAgent):
    
    def __init__(self, states = 3, window = 1, counter = None):
        super().__init__(states, window, counter)
        self.power = window
        self.dimen = np.power(self.states, self.power)

        
    def positions(self, vals):
        
        total = 0

        for index in range(0, self.power):
            total = total + vals[index] * np.power(self.states, self.power - index - 1)

        return int(total.item())
        
    def predict(self):
           
        totalMoves = len(self.opponent)
        
        if totalMoves <= self.power:
            return self.submit(self.random())
        
        initials = np.zeros(self.dimen).astype('float64')
        transitions = np.zeros((self.dimen, self.dimen)).astype('float64')
        
        initials[self.positions(self.opponent[0:self.power])] = 1
        for index in range(self.power, totalMoves):  
            dest = self.positions(self.opponent[index - self.power + 1:index + 1])
            src = self.positions(self.opponent[index - self.power:index])
            transitions[dest, src] = transitions[dest, src] + 1
        
        for col in range(0, self.dimen):
            transitions[:, col] = 0 if transitions[:, col].sum() == 0 else transitions[:, col] / transitions[:, col].sum()
            
        probs = np.matmul(np.linalg.matrix_power(transitions, totalMoves - self.power + 1), initials)    
         
        res = np.argwhere(probs == np.amax(probs)).ravel()
        
        return self.submit((np.random.choice(res).item() + 1) % self.states)


'''
https://towardsdatascience.com/mixture-transition-distribution-model-e48b106e9560
Borrowing the concept of generalized mixture transaction to capture sequence information
then flatten it into a pandas dataframe

https://math.stackexchange.com/questions/362412/generating-a-monotonically-decreasing-sequence-that-adds-to-1-for-any-length
for generating decreasing sequence that adds up to 1
'''
class GMarkov(BaseAgent):
    
    def __init__(self, states, window = 3, buff_win = 0, counter = None):
        super().__init__(states, window, counter)     
        self.dimen = np.power(self.states, 1)
        self.buffWin = buff_win
        self.lambdas = self.priors()
        self.transitions = []
            
    def priors(self):
        
        seq = []
        for index in range(self.window, 0, -1):
            seq.append((2 * index - 1) / (self.window * self.window))
            
        return seq
    
    def predict(self):
        
        totalMoves = len(self.opponent)
        
        if totalMoves <= self.window:
            return self.submit(self.random())

        for _ in range(0, self.window):
            self.transitions.append(np.zeros((self.dimen, self.dimen)).astype('float64'))
        
        for index in range(0, totalMoves - self.window):
            submoves = self.opponent[index:index + self.window + 1]
            submines = self.mines[index:index + self.window + 1]
            
            length = len(submoves)
            for subindex in range(0, length - 1):
                dest = int(submoves[-1])
                src = int(submoves[subindex])
                self.transitions[subindex][src, dest] = self.transitions[subindex][src, dest] + 1
            
                res = submines[subindex] - submoves[subindex]
                if res == 1 or res == -2:
                    self.transitions[subindex][src, dest] = self.transitions[subindex][src, dest] + self.buffWin
        
         
        for subindex in range(0, self.window):
            for row in range(0, self.dimen):
                self.transitions[subindex][row, :] = 0 if self.transitions[subindex][row, :].sum() == 0 else self.transitions[subindex][row, :] / self.transitions[subindex][row, :].sum()
               
        
        submoves = self.opponent[totalMoves - self.window:]

        best_score = -99
        best_move = 0
        for target in range(0, self.states):
            prob = 0.0
            for subindex in range(0, len(submoves)):
                dest = target
                src = int(submoves[subindex])
                prob += self.transitions[subindex][src, dest] * self.lambdas[subindex] 
 
            if prob > best_score:
                best_score = prob
                best_move = target
            
        return self.submit((best_move + 1) % self.states)
    

