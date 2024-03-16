'''
Created on Mar 12, 2024

@author: STEPHEN
@url: https://www.youtube.com/watch?v=783ZtAF4j5g


A round (r) consists of all ants travel from beginning node(s) to the final destination node(s).

k'th ant travels on the edge(s) i to z <- final destination 
L is the fixed cost or length of the path all traveled paths 
--------------------------------------------------------------------
  
        z-1,z
  L(k) =  ∑  fixed_cost(x,y)
       x=i,y=i+1
                 
  all paths(i..z) traveled by the same ant would have the same Δ τ value for that ant  
  Δ τ(i..z,k) = 1/L(k)
            

Pheromone values Without vaporization round (r)
-----------------------------------------------
               m
  τ(r,i,j,k) = ∑  Δ τ(r,i,j,k)       
              k=1
  
  
Pheromone values With vaporization of percentage (p), round (r)
-----------------------------------------------------------------
                                      m
  τ(r+1,i,j,k) = (i - p) * τ(r,i,j) + ∑  Δ τ(r,i,j,k)
                                     k=1
                
Calculating the probabilities
------------------------------
  η(i,j) = 1/fixed_cost(i,j)   <== fixed cost or length of the path(i,j)
  α and β adjust the effectiveness of the components.
  We could use only the Pheromone for decision making by setting β to 0
  
                    τ(r,i,j)^α * η(i,j)^β  
  P(r+1,i,j) = -------------------------
                ∑ ( τ(r,i,j)^α * η(i,j)^β )                                       
                                     
                                     
DaemonActions
--------------
   Once solutions have been constructed, and before updating the pheromone values, often some problem 
   specific actions may be required. These are often called daemon actions, and can be used to implement 
   problem specific and/or centralized actions, which cannot be performed by single ants. The most used 
   daemon action consists in the application of local search to the constructed solutions: the locally 
   optimized solutions are then used to decide which pheromone values to update.  
 
 
  procedure ACO_MetaHeuristic is
    init()
    while not terminated do
        generateAntsSolutions()  // all ants select paths from beginning to the final destination
        daemonActions()          // optional
        pheromoneUpdate()        
    repeat
  end procedure
                                   
                                  
 Matrix solution would need cost matrix, pheromone matrix, probabilities matrix     
 
 ant_matrix     
    i x j   
 fixed_cost_matrix           
    i x j                                         
 delta_pheromone_matrix 
    i x j
 pheromone_matrix
    i x j
 probabilities_matrix
    i x j             
    
    
 init()
    ones(pheromone_matrix)
     
 generateAntsSolutions()
    generate_probabilities()
        pheromone_matrix, fixed_cost_matrix => probabilities_matrix
        -----------------------------------------------------------
            top                       = multiply_element (pheromone_matrix^α, fixed_cost_matrix^β) 
            bottom                    = horizontal_sum(top)
            probabilities_matrix      = horizontal_divide(top, bottom)
            
    generate_ants_solution()
        probabilities_matrix => ants_matrix
        -----------------------------------
            zeros(ant_matrix)
            for k in (1..100):
                i = random([ possible starting locations ])
                j = -1
                while j not in [ possible final destinations ]:
                    j = decide(probabilities_matrix[i])
                    ant_matrix[k,i,j] = fixed_cost_matrix[i,j]
                    i = j
                ant_matrix[k] = 1 / sum(ant_matrix[k])
              
        
 pheromoneUpdate()
    generate_delta_pheromone()
        ants_matrix, fixed_cost_matrix => delta_pheromone_matrix 
        --------------------------------------------------------
        zeros(delta_pheromone_matrix)
        for ant_matrix in ants:
            delta_pheromone_matrix += ant_matrix
            
    update_pheromone()
        pheromone_matrix, delta_pheromone matrix => pheromone_matrix
        ------------------------------------------------------------
        pheromone_matrix = (1 - p) * pheromone_matrix + delta_pheromone_matrix
 '''
import numpy as np
import threading

class ACO_Optimization:
    
    def __init__(self, cost_matrix, start_locs, end_locs, numOfAnts, evaporation = 0.4, alpha_value = 1, beta_value = 0.1):
        self.cost_matrix = cost_matrix
        self.start_locs = start_locs
        self.end_locs = end_locs
        self.numOfAnts = numOfAnts
        self.evaporation = evaporation
        self.alpha_value = alpha_value
        self.beta_value = beta_value
        self.pheromone_matrix = np.zeros(shape=cost_matrix.shape, dtype=float)
        self.pheromone_matrix[np.where(self.cost_matrix > 0)] = 1.0
        
    def start(self, rnds = 5):
        for _ in range(rnds):
            ants_matrix = self.generateAntsSolutions()  
            self.daemonActions(ants_matrix)          
            self.pheromoneUpdate(ants_matrix)   

    def generateAntsSolutions(self):
        probabilities_matrix = self.generate_probabilities()
        return self.generate_ants_solution(probabilities_matrix)
    
    def generate_probabilities(self):
        neta = np.divide(1, self.cost_matrix, where=self.cost_matrix!=0)
        np.nan_to_num(neta, copy=False)
        nominator = np.multiply(np.power(self.pheromone_matrix, self.alpha_value), 
                                np.power(np.abs(neta), self.beta_value))
        denominator = np.sum(nominator, axis=1, keepdims=True)
        return np.nan_to_num(np.divide(nominator, denominator, out=np.zeros_like(nominator), where=denominator!=0, dtype=float), copy=False)
    
    def generate_ants_solution(self, probabilities_matrix):
        
        ants_matrix = np.zeros(tuple(np.insert(list(self.cost_matrix.shape), 0, self.numOfAnts, axis=0)))
        index = range(len(probabilities_matrix[0]))
        threads = []
        
        def thread_handler(k):
            t = threading.Thread(target=generate_ant_solution, args=(k,))
            threads.append(t)
            t.start()
        
        def generate_ant_solution(k):
            i = np.random.choice(self.start_locs)
            j = -1
            while j not in self.end_locs: 
                deadend = False
                if not np.any(probabilities_matrix[i]):
                    if not np.any(self.cost_matrix[i]):
                        j = self.end_locs[0]
                        deadend = True
                    else:
                        j = np.random.choice(np.nonzero(self.cost_matrix[i])[0])
                else:
                    j = np.random.choice(index, 1, p=probabilities_matrix[i])[0]
               
                ants_matrix[k][i, j] = self.cost_matrix[i, j]
                i = j
                
            total = np.sum(ants_matrix[k])
            if total == 0 or deadend == True:
                total = 999999999
            l = 1 / total
            ants_matrix[k][np.where(ants_matrix[k] != 0)] = l
    
        [ thread_handler(k) for k in index ]
        [ t.join() for t in threads ]
        
        return ants_matrix
      
    def daemonActions(self, ants_matrix):
        pass
    
    def pheromoneUpdate(self, ants_matrix):
        delta_pheromone_matrix = self.generate_delta_pheromone(ants_matrix)
        self.update_pheromone(delta_pheromone_matrix)
        
    def generate_delta_pheromone(self, ants_matrix):
        return np.sum(ants_matrix, axis=0, dtype=float)
    
    def update_pheromone(self, delta_pheromone_matrix):
        self.pheromone_matrix = (1 - self.evaporation) * self.pheromone_matrix + delta_pheromone_matrix
    
    def best_start(self):
        cloned = np.array(self.pheromone_matrix)
        ind = np.array(range(self.pheromone_matrix.shape[0]))
        ind = np.delete(ind, self.start_locs)
        cloned[ind] = 0
        return int(np.argmax(cloned) / cloned.shape[1])
            
    def best_path(self, labels):    
        i = self.best_start()
        j = -1
        shortest = "[" + labels[i] + "]"
        while j not in self.end_locs:
            j = np.argmax(self.pheromone_matrix[i])
            shortest += " = " + str(self.cost_matrix[i, j]) + " => [" + labels[j] + "]"
            i = j
                
        return shortest
    
    def best_scores(self):
        i = self.start_locs[0]
        j = -1
        total = 0
        while j not in self.end_locs:
            j = np.argmax(self.pheromone_matrix[i])
            total += self.cost_matrix[i, j]
            i = j
                
        return total
        

    

