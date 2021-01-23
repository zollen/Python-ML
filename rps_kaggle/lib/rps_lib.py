'''
Created on Dec. 15, 2020

@author: zollen
'''
import numpy as np
import time
import itertools


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
        self.data = np.zeros(shape = (1100, self.window * 3)).astype('int64')
    
    def encode(self, me, op):
        return self.TABLE[str(me) + str(op)]
        
    def convert(self, buf):
                
        arr = []
        for index in range(self.window):
            arr.append(self.encode(buf[index], buf[index + self.window]))
        
        return np.array(buf + arr).astype('int64')
    
    def decide(self):
        
        if np.random.uniform(0, 1) <= self.randomness:
            return self.submit(self.random())
        else:
            return super().decide()
    


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
    


class Scorer:
    
    def __init__(self, agents):
        self.agents = agents
        
    def normalize(self, scores):
        total = np.sum(scores)
        if total <= 0:
            return [ 0 ] * len(self.agents)
        
        return [ x / total for x in scores ]
    
class PopularityScorer(Scorer):
    
    def __init__(self, agents):
        super().__init__(agents)
    
    def calculate(self):
        
        counts = [0, 0, 0]
        for _, output, _ in self.agents:
            counts[output[-1]] += 1
            
        return counts
    
    def normalize(self, scores):
        
        final_scores = []
        for _, predicted, _ in self.agents:
            final_scores.append(scores[predicted[-1]])
            
        return super().normalize(final_scores)
    
class OutComeScorer(Scorer):
    
    WON = 0
    LOST = 1
    
    def __init__(self, agents, step = 3, decay = 1.1):
        super().__init__(agents)
        self.numAgents = len(agents)
        self.score = [ [0, 0] for _ in range(self.numAgents) ] 
        self.step = step
        self.decay = decay
        
    def calculate(self):
        
        idx = 0
        for _, _, outcome in self.agents:
            self.score[idx][self.WON] = (self.score[idx][self.WON] - 1) / self.decay + 1
            self.score[idx][self.LOST] = (self.score[idx][self.LOST] - 1) / self.decay + 1

            if outcome and outcome[-1] == 1:
                self.score[idx][self.WON] += self.step
            elif outcome and outcome[-1] == -1:
                self.score[idx][self.LOST] += (self.step + 1)
            else:
                self.score[idx][self.WON] += self.step / 2
                self.score[idx][self.LOST] += self.step / 2
            idx += 1
                
        return self.score
    
    def normalize(self, scores):
        
        final_scores = []
        for won, lost in scores:
            final_scores.append(won / (won + lost))
            
        return super().normalize(final_scores)
    
class Last5RoundsScorer(Scorer):
    
    def __init__(self, agents):
        super().__init__(agents)
    
    def calculate(self):
        
        final_scores = []
        for _, _, outcome in self.agents:
            final_scores.append(outcome[-5:].count(1))
        
        return final_scores
    
    def normalize(self, scores):
        return super().normalize(scores)
    
class BetaScorer(OutComeScorer):
    
    def __init__(self, agents):
        super().__init__(agents)
        self.history = []
    
    def calculate(self):
        
        scores = super().calculate()
        
        final_scores = []
        for won, lost in scores:
            final_scores.append(np.random.beta(won, lost))
        return final_scores
    
    def normalize(self, scores):
        
        self.history.append(scores)
        
        final_scores = [0] * len(self.agents)
        for score in self.history[-3:]:
            final_scores = [ a + b for a, b in zip(final_scores, score)]
        
        return Scorer.normalize(self, final_scores)
    

class StatsAgency(BaseAgent):
        
    def __init__(self, scorers, agents, states = 3, random_threshold = -10):
        super().__init__(states, 0, 0, None)
        self.scorers = scorers
        self.agents = agents
        self.random_threshold = random_threshold
        self.rnd = 0
        self.combos = self.generate()
        self.executor = None
        self.totalwon = 0
        self.totallost = 0
    
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
        if self.executor == None:
            return "StatsAgency(Crazy)"
        return "StatsAgency(" + self.executor.__str__() + ")"
    
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
        rew = (mymove - opmove) % self.states 
        if rew == 1:
            return 1
        elif rew == 2:
            return -1
        return 0
    
    def lastmatch(self):
        if len(self.mines) > 0 and len(self.opponent) > 0:
            res = self.reward(self.mines[-1], self.opponent[-1])
            if res > 0:
                self.totalwon += 1
            elif res < 0:
                self.totallost += 1
    
    def decide(self):
        
        self.lastmatch()
        
        final_scores = [0] * len(self.agents)
                
        if self.rnd > 0:
            for _, predicted, outcome in self.agents:
                outcome.append(self.reward(predicted[-1], self.opponent[-1]))
             
            for scorer in self.scorers:
                new_scores = scorer.normalize(scorer.calculate())
                final_scores = [ a + b for a, b in zip(final_scores, new_scores) ]
                
        for agent, predicted, _ in self.agents:
            try :
                predicted.append(agent.estimate())
            except:
                predicted.append(np.random.randint(self.states))
            
        
        self.rnd += 1
        
        self.executor = self.agents[np.argmax(final_scores)][0]
        
        stats = self.totalwon - self.totallost
        if stats < self.random_threshold and np.random.uniform(0, 1) < 0.5 + (stats * 0.01):
            self.executor = None
            return self.submit(self.random())
                              
        return self.submit(self.agents[np.argmax(final_scores)][1][-1])




     
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
                scores[1] += 3
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
    
    def __init__(self, agents, states = 3, step_size = 3, decay = 1.05):
        self.agents = agents
        self.states = states
        self.stepSize = step_size
        self.decay = decay
        self.executor = None
        
    def __str__(self):
        return "BetaAgency(" + self.executor.__str__() + ")"
    
    def add(self, token):
        for agent, _, _ in self.agents:
            agent.add(token)
    
    def lastgame(self, agent, result):
        
        if agent.opQueue().size <= 0:
            return 0
        
        res = (result - agent.opQueue(-1)) % self.states
        if res == 1:
            return 1
        elif res == 2:
            return -1
        
        return 0    
          
    def decide(self):
        for agent, scores, result in self.agents:
            
            agent.reset()
            
            scores[0] = (scores[0] - 1) / self.decay + 1
            scores[1] = (scores[1] - 1) / self.decay + 1
          
            outcome = self.lastgame(agent, result[0])
            if outcome > 0:
                scores[0] += self.stepSize
            elif outcome < 0:
                scores[1] += (self.stepSize + 1)
            else:
                scores[0] = scores[0] + self.stepSize / 2
                scores[1] = scores[1] + self.stepSize / 2
        
        
        best_prob = -1
        best_agent = None
        best_move = None
        for agent, scores, result in self.agents:
            prob = np.random.beta(scores[0], scores[1])
            result[0] = agent.estimate()
            if prob > best_prob:
                best_prob = prob
                best_agent = agent
                best_move = result[0]
        
        self.executor = best_agent
        
        for agent, _, _ in self.agents:
            agent.deposit(best_move)
               
        return best_move
    
