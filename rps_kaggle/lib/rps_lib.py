'''
Created on Dec. 15, 2020

@author: zollen
'''
import numpy as np
import time


class BaseCounterMover:
    
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
        
    def random(self):
        return np.random.randint(0, self.states)
    
    def reset(self):
        self.wincounter = 0
        self.evencounter = 0
        
    def decide(self, token):
        return self.random()
        
    
class AgressiveCounterMover(BaseCounterMover):
    
    def __init__(self, agent, states = 3, interval = 5):
        super().__init__(agent, states, interval)
        
    def decide(self, token):
      
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
    
class RandomCounterMover(BaseCounterMover):
    
    def __init__(self, agent, states = 3, interval = 5):
        super().__init__(agent, states, interval)
        
    def decide(self, token):
      
        if self.wincounter <= 0 and self.won(-1) < 0 and self.won(-2) < 0:
            self.wincounter = np.random.randint(1, self.interval + 1)
            
        if self.evencounter <= 0 and self.won(-1) == 0 and self.won(-2) == 0:
            self.evencounter = np.random.randint(1, self.interval + 1)
                
        if self.wincounter > 0:
            self.wincounter = self.wincounter - 1
            if np.random.randint(0, 2) == 0:
                return (token + 2) % self.states
            else:
                choices = [0, 1, 2]
                choices.remove(token)
                return choices[np.random.randint(0, 2)]
            
        if self.evencounter > 0:
            self.evencounter = self.evencounter - 1
            if np.random.randint(0, 2) == 0:
                return (token + 1) % self.states
            else:
                choices = [0, 1, 2]
                choices.remove((token + 2) % self.states)
                return choices[np.random.randint(0, 2)]
            

        return token    

class StandardCounterMover(BaseCounterMover):
    
    def __init__(self, agent, states = 3, interval = 5):
        super().__init__(agent, states, interval)
        
    def decide(self, token):
      
        if self.wincounter <= 0 and self.won(-1) < 0 and self.won(-2) < 0:
            self.wincounter = np.random.randint(1, self.interval + 1)
                
        if self.wincounter > 0:
            self.wincounter = self.wincounter - 1
            return (token + 2) % self.states

        return token
    


class BaseAgent:
    
    def __init__(self, states = 3, window = 3, ahead = 1, counter = None):
        np.random.seed(int(round(time.time())))
        self.mines = np.array([]).astype('int64')
        self.opponent = np.array([]).astype('int64')
        self.results = np.array([]).astype('int64')
        self.states = states
        self.window = window
        self.ahead = ahead
        self.counter = counter
        self.record = True
               
    def myQueue(self, index = None):
        if index == None:
            return self.mines
        
        return self.mines[index]
    
    def opQueue(self, index = None):
        if index == None:
            return self.opponent
        
        return self.opponent[index]
        
    def __str__(self):
        if self.counter == None:
            name = self.__class__.__name__ + "{" + str(self.ahead) + "}"
        else:
            name = self.__class__.__name__ + "(" + self.counter.__class__.__name__ + "){" + str(self.ahead) + "}"
        return name
        
    def add(self, token):
        self.opponent = np.append(self.opponent, token)
        
    def deposit(self, token):
        self.mines = np.append(self.mines, token)
            
    def submit(self, token):
        bak = token
        
        if self.counter != None and len(self.opponent) > self.window + 2:
            token = self.counter.decide(token)
        
        if token is None:
            token = bak
        
        if self.record:    
            self.deposit(token)
        
        return token
    
    def reset(self):
        if self.counter != None:
            self.counter.reset()
    
    def random(self):
        return np.random.randint(0, self.states)
    
    def estimate(self):
        self.record = False
        return self.decide()
        
    def decide(self):
        pass



    
class RandomHolder:
    
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.current = None
        
    def __str__(self):
        return "[" + self.current.__class__.__name__ + "]"
        
    def fit(self, X, y):
        self.current = self.classifiers[np.random.randint(0, len(self.classifiers))]
        self.current.fit(X, y)
        
    def predict(self, X):
        return self.current.predict(X)
        
        

         
class Classifier(BaseAgent):
    
    def __init__(self, classifier, states = 3, window = 3, ahead = 1, delay_process = 5, history = -1, counter = None):
        super().__init__(states, window, ahead, counter)
        self.classifier = classifier
        self.history = history
        self.delayProcess = delay_process
        self.row = 0
        self.pos = 0
        self.full = False
        self.data = np.zeros(shape = (1100, self.window * 2)).astype('int64')
        self.results = np.zeros(shape = (1100, 1)).astype('int64')
        self.last = None
             
    def __str__(self):
        clsName = self.classifier.__class__.__name__
        if self.counter == None:
            name = self.__class__.__name__ + "(" + clsName + "){" + str(self.ahead) + "}"
        else:
            name = self.__class__.__name__ + "(" + clsName + "(" + self.counter.__class__.__name__ + ")){" + str(self.ahead) + "}"
        return name
    
    def reset(self):
        self.last = None
        super().reset()
   
    def add(self, token):
        super().add(token)
        
        if len(self.opponent) >= self.window + 1: 
            self.prepare() 
    
    def prepare(self):
        self.data[self.pos] = self.convert(self.mines[self.row:self.row+self.window].tolist() + self.opponent[self.row:self.row+self.window].tolist())
        self.results[self.pos] = self.opponent[-1]   
        self.row = self.row + 1
        if self.history <= 0:
            self.pos = self.row
        else:
            if self.row == self.history:
                self.full = True
            self.pos = (self.row % self.history)
        
    def test(self):
        return np.array(self.convert(self.mines[-self.window:].tolist() + self.opponent[-self.window:].tolist())).reshape(1, -1)
  
    def decide(self):
        
        lpos = self.pos if self.full == False else self.history

        if len(self.opponent) > self.window + self.delayProcess + 1:
            self.classifier.fit(self.data[:lpos], self.results[:lpos])  
            
            self.last = (int(self.classifier.predict(self.test()).item()) + self.ahead) % self.states
            return self.submit(self.last)
        
        self.last = self.random()    
        return self.submit(self.last)

    
    def convert(self, buf):
        return buf
    

    
class MClassifier(Classifier):
    
    def __init__(self, classifier, states = 3, window = 3, ahead = 1, delay_process = 5, history = -1, counter = None):
        super().__init__(classifier, states, window, ahead, delay_process, history, counter)
        self.data = np.zeros(shape = (1100, self.window * 2 * 2)).astype('int64')
        
    def convert(self, buf):  
        arr = []
        manifest = [[0, 0], [0, 1], [1, 0]]
        for ch in buf:
            arr.extend(manifest[ch])
    
        return np.array(arr).astype('int64')
        



class SClassifier(Classifier):
    
    MANIFEST = [ [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0]
                ]
    
    def __init__(self, classifier, states = 3, window = 3, ahead = 1, delay_process = 5, history = -1, counter = None):
        super().__init__(classifier, states, window, ahead, delay_process, history, counter)
        self.data = np.zeros(shape = (1100, (self.window - 1) * 2 * 8)).astype('int64')
         
    def convert(self, buf):
        def encode(last, curr):
            return last * self.states + curr
      
        arr = []
        for index in range(1, self.states):
            arr.extend(self.MANIFEST[encode(buf[index - 1], buf[index])])
        
        for index in range(self.states + 1, len(buf)):
            arr.extend(self.MANIFEST[encode(buf[index - 1], buf[index])])

        return np.array(arr).astype('int64')


    

class OClassifier(Classifier):
    
    def __init__(self, classifier, states = 3, window = 3, ahead = 1, delay_process = 5, history = -1, counter = None):
        super().__init__(classifier, states, window, ahead, delay_process, history, counter)
        self.data = np.zeros(shape = (1100, self.window * 3)).astype('int64')
            
    def won(self, me, opp):
        
        res = me - opp
        if res == 1 or res == -2:
            return 2
        elif res == -1 or res == 2:
            return 0
        
        return 1
        
    def convert(self, buf):
                
        arr = []
        for index in range(self.window):
            arr.append(self.won(buf[index], buf[index + self.window]))
        
        return np.array(buf + arr).astype('int64')
    



class KClassifier(Classifier):
    
    TABLE = { 
              "01": 0, "12": 1, "20": 2,
              "00": 3, "11": 4, "22": 5,
              "10": 6, "21": 7, "02": 8
            }
    
    def __init__(self, classifier, states = 3, window = 3, ahead = 1, delay_process = 5, history = -1, randomness = 0, counter = None):
        super().__init__(classifier, states, window, ahead, delay_process, history, counter)
        self.randomness = randomness
        self.data = np.zeros(shape = (1100, self.window)).astype('int64')
    
    def encode(self, me, op):
        return self.TABLE[str(me) + str(op)]
        
    def convert(self, buf):
                
        arr = []
        for index in range(self.window):
            arr.append(self.encode(buf[index], buf[index + self.window]))
        
        return np.array(arr).astype('int64')
    
    def decide(self):
        
        if np.random.uniform(0, 1) <= self.randomness:
            return self.submit(self.random())
        else:
            return super().decide()
 


class MirrorOpponentDecider(BaseAgent):
    
    def __init__(self, states = 3, window = 0, ahead = 0, counter = None):
        super().__init__(states, window, ahead, counter)
        
    def decide(self):
        
        if len(self.opponent) <= 0:
            return self.submit(self.random())
 
        return self.submit((self.opponent[-1].item() + self.ahead) % self.states)

    
class MirrorSelfDecider(BaseAgent):
    
    def __init__(self, states = 3, window = 0, ahead = 0, counter = None):
        super().__init__(states, window, ahead, counter)
        
    def decide(self):
        
        if len(self.mines) <= 0:
            return self.submit(self.random())

        return self.submit((self.mines[-1].item() + self.ahead) % self.states)


class MostCommonDecider(BaseAgent):
    
    def __init__(self, states = 3, window = 0, ahead = 1, counter = None):
        super().__init__(states, window, ahead, counter)
        
    def decide(self):
        
        if len(self.mines) <= 0:
            return self.submit(self.random())

        counts = np.bincount(self.opponent)
        return self.submit((int(np.argmax(counts)) + self.ahead) % self.states)   



class LeastCommonDecider(BaseAgent):
    
    def __init__(self, states = 3, window = 0, ahead = 1, counter = None):
        super().__init__(states, window, ahead, counter)
        
    def decide(self):
        
        if len(self.mines) <= 0:
            return self.submit(self.random())

        counts = np.bincount(self.opponent)
        return self.submit((int(np.argmin(counts)) + self.ahead) % self.states) 
    
    
        
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
        
    def decide(self):
           
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
    
    def decide(self):
        
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

    


class Sharer:
    
    def __init__(self, classifier, states = 3, ahead = 0):
        self.classifier = classifier
        self.states = states
        self.ahead = ahead
        
    def __str__(self):
        return "<" +self.classifier.__str__() + ">{" + str(self.ahead) + "}"
    
    def myQueue(self, index = None):
        return self.classifier.myQueue(index)
    
    def opQueue(self, index = None):
        return self.classifier.opQueue(index)
    
    def reset(self):
        self.classifier.reset()
    
    def deposit(self, token):
        if self.classifier.last == None:
            self.classifier.deposit(token)
        
    def add(self, token):
        if self.classifier.last == None:
            self.classifier.add(token)
            
    def estimate(self):
        if self.classifier.last != None:
            return (self.classifier.last + self.ahead) % self.states
        return self.classifier.estimate()
    


class VoteAgency(BaseAgent):
    
    def __init__(self, agents, states = 3, randomness = 0):
        super().__init__(states, 0, 0, None)
        self.agents = agents
        self.randomness = randomness
        self.totalWin = 0
        self.totalLoss = 0
        self.crazy = False
        self.executor = None
        
    def __str__(self): 
        if self.crazy:
            return "VoteAgency(Crazy)"
        return "VoteAgency(" + self.executor.__str__() + ")"
    
    def add(self, token):
        self.opponent = np.append(self.opponent, token)
        for agent, _, _ in self.agents:
            agent.add(token)
            
    def submit(self, token):
        self.mines = np.append(self.mines, token)
        return token

    def lastmatch(self):
        out = (self.mines[-1] - self.opponent[-1]) % self.states
        if out == 1:
            self.totalWin += 1
        elif out == 2:
            self.totalLoss += 1
              
        for agent, scores, result in self.agents:
            
            agent.reset()
            
            scores[0] = (scores[0] - 1) / 1.2 + 1
            scores[1] = (scores[1] - 1) / 1.2 + 1
           
            res = (result[0] - self.opponent[-1]) % self.states
            if res == 1:
                scores[0] += 3
            elif res == 2:
                scores[1] += 6
            else:
                scores[0] += 3 / 2
                scores[1] += 3 / 2
            
            
    def choose(self):
   
        probabilities = {}
        for agent, scores, result in self.agents:
     
            if result[0] not in probabilities:
                probabilities[result[0]] = [ agent ]
            probabilities[result[0]].append(round(0 if scores[1] == 0 else (scores[0] / (scores[1] + 1)), 4))
        
        
        
        best_agent = None
        best_move = -1
        best_prob = -1
        for key, val in probabilities.items():
            mprob = np.median(val[1:])
            if mprob > best_prob:
                best_agent = val[0]
                best_prob = mprob
                best_move = key
                
        self.executor = best_agent
      
        return best_move
    
    def decide(self):
                         
        if self.mines.size > 0 and self.opponent.size > 0:
            self.lastmatch()
      
        self.lastmoves = np.array([]).astype('int64')
        
        try:    
            for agent, _, result in self.agents:
                result[0] = agent.estimate()
        except:
            pass
            
             
        self.executor = self.agents[0]
        best_move = self.random()
        
        if self.randomness > 0 and np.random.uniform(0, 1) <= self.randomness:
            self.crazy = True
            best_move = self.random()
        else:
            self.crazy = False
            if self.mines.size > 0 and self.opponent.size > 0:
                best_move = self.choose()    
                      
        for agent, _, _ in self.agents:
            agent.deposit(best_move)
        
        return self.submit(best_move)




     
class MetaAgency(BaseAgent):
    
    TABLE = { 
              "01": 0, "12": 1, "20": 2,
              "00": 3, "11": 4, "22": 5,
              "10": 6, "21": 7, "02": 8
            }
    
    def __init__(self, managers, agents, states = 3, history = -1, randomness = 0, random_threshold = -20, window = 10):
        super().__init__(states, window, 0, None)
        self.managers = managers
        self.agents = agents
        self.data = np.zeros(shape = (1100, self.window)).astype('int64')
        self.results = np.zeros(shape = (1100, )).astype('int64')
        self.row = 0
        self.history = history
        self.full = False
        self.randomthreshold = random_threshold
        self.randomness = randomness
        self.lostcontrol = randomness
        self.executor = None
        self.lastmoves = np.array([]).astype('int64')
        self.testdata = np.array([]).astype('int64')
        self.totalWin = 0
        self.totalLoss = 0
        self.crazy = False
        
    def __str__(self): 
        if self.crazy:
            return "MetaAgency(Crazy)"
        return "MetaAgency(" + self.executor.__str__() + ")"
    
    def add(self, token):
        self.opponent = np.append(self.opponent, token)
        for agent in self.agents:
            agent.add(token)
            
    def submit(self, token):
        self.mines = np.append(self.mines, token)
        return token
            
    def encode(self, index):
        return self.TABLE[str(self.mines[index]) + str(self.opponent[index])]
    
    def lastmatch(self):
        out = (self.mines[-1] - self.opponent[-1]) % self.states
        if out == 1:
            self.totalWin += 1
        elif out == 2:
            self.totalLoss += 1
            
        for agent in self.agents:
            agent.reset()
            
        for _, scores, result in self.managers:
            
            scores[0] = (scores[0] - 1) / 1.1 + 1
            scores[1] = (scores[1] - 1) / 1.1 + 1
            
            res = (self.lastmoves[result[0]] - self.opponent[-1]) % self.states
            if res == 1:
                scores[0] += 3
            elif res == 2:
                scores[1] += 5
            else:
                scores[0] += 3 / 2
                scores[1] += 3 / 2
            
            
    def choose(self, last):
        
        best_agent = 0
        
        try :
            
            probabilities = {}
            for manager, scores, result in self.managers:
                manager.fit(self.data[:last], self.results[:last])
                result[0] = manager.predict(self.testdata)[0]
                if result[0] not in probabilities:
                    probabilities[result[0]] = []
                probabilities[result[0]].append(round(scores[0] / (scores[1] + 1), 4))
            
            best_agent = -1
            best_prob = -1
            for key, val in probabilities.items():
                mprob = np.median(val)
                if mprob > best_prob:
                    best_prob = mprob
                    best_agent = key
        except:
            pass
      
        return best_agent
    
    def decide(self):
        
           
        last = 0
      
        if self.mines.size > 0 and self.opponent.size > 0:
            self.lastmatch()
            current = self.totalWin - self.totalLoss
            if current < self.randomthreshold:
                ratio = 0.3 + abs(current) * 0.1
                self.lostcontrol = ratio if ratio < 0.6 else 0.6
            else:
                self.lostcontrol = self.randomness

    
        if self.mines.size > self.window and self.opponent.size > self.window:
            outcomes = []
            for index in range(self.row, self.row + self.window):
                outcomes.append(self.encode(index))
                
            self.data[self.row] = outcomes
            self.results[self.row] = np.where(self.lastmoves == (self.opponent[-1].item() + 1) % self.states)[0][0]
            
            
            if self.full == False:
                if self.history > 0 and self.row + 1 >= self.history:
                    self.full = True
            
            if self.history > 0:        
                self.row = (self.row + 1) % self.history
            else:
                self.row += 1
            
            last = self.row if self.full == False else self.history
            
            
            
            outcomes = []
            for index in range(self.mines.size - self.window, self.mines.size):
                outcomes.append(self.encode(index))
                
            self.testdata = np.array(outcomes).astype('int64').reshape(1, -1)
        
      
        self.lastmoves = np.array([]).astype('int64')
        
        try:
            
            for agent in self.agents:
                self.lastmoves = np.append(self.lastmoves, agent.estimate())
        except:
                self.lastmoves = np.array([0, 1, 2])
            
             
        self.executor = self.agents[0]
        best_move = self.lastmoves[0].item()
        
        if self.randomness > 0 and np.random.uniform(0, 1) <= self.lostcontrol:
            self.crazy = True
            best_move = self.random()
        else:
            self.crazy = False
            if self.testdata.size > 0 and last > 5:
                best_agent = self.choose(last)
                self.executor = self.agents[best_agent]
                best_move = self.lastmoves[best_agent].item()
        
                      
        for agent in self.agents:
            agent.deposit(best_move)
        
        return self.submit(best_move)
        
    
    
class BetaAgency:
    
    def __init__(self, agents, step_size = 3, decay = 1.05):
        self.agents = agents
        self.stepSize = step_size
        self.decay = decay
        self.executor = None
        
    def __str__(self):
        return "BetaAgency(" + self.executor.__str__() + ")"
    
    def add(self, token):
        for agent, _ in self.agents:
            agent.add(token)
    
    def lastgame(self, agent):
        if agent.myQueue().size <= 0 or agent.opQueue().size <= 0:
            return 0
        
        res = (agent.myQueue(-1) - agent.opQueue(-1)) % 3
        if res == 1:
            return 1
        elif res == 2:
            return -1
        
        return 0    
          
    def decide(self):
        for agent, scores in self.agents:
            
            agent.reset()
            
            scores[0] = (scores[0] - 1) / self.decay + 1
            scores[1] = (scores[1] - 1) / self.decay + 1
          
            outcome = self.lastgame(agent)
            if outcome > 0:
                scores[0] += self.stepSize
            elif outcome < 0:
                scores[1] += self.stepSize
            else:
                scores[0] = scores[0] + self.stepSize / 2
                scores[1] = scores[1] + self.stepSize / 2
        
        
        best_prob = -1
        best_agent = None
        best_move = None
        for agent, scores in self.agents:
            prob = np.random.beta(scores[0], scores[1])
            move = agent.estimate()
            if prob > best_prob:
                best_prob = prob
                best_agent = agent
                best_move = move
        
        self.executor = best_agent
        
        for agent, _ in self.agents:
            agent.deposit(best_move)
               
        return best_move
    
