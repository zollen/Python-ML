'''
Created on Mar 12, 2024

@author: STEPHEN
 '''
import numpy as np

class ACO_Optimization:
    
    def __init__(self, cost_matrix, start_locs, end_locs, numOfAnts, evaporation = 0.5, alpha_value = 1, beta_value = 0.1):
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
                                np.power(neta, self.beta_value))
        denominator = np.sum(nominator, axis=1, keepdims=True)
        return np.nan_to_num(np.divide(nominator, denominator, out=np.zeros_like(nominator), where=denominator!=0, dtype=float), copy=False)
    
    def generate_ants_solution(self, probabilities_matrix):
        
        index = range(len(probabilities_matrix[0]))
        
        def generate_ant_solution(ant):
            i = np.random.choice(self.start_locs)
            j = -1
            while j not in self.end_locs:
                j = np.random.choice(index, 1, p=probabilities_matrix[i])[0]
                ant[i, j] = self.cost_matrix[i, j]
                i = j
            l = 1 / np.sum(ant)
            ant[np.where(ant != 0)] = l
            return ant
    
        ants_matrix = np.zeros(tuple(np.insert(list(self.cost_matrix.shape), 0, self.numOfAnts, axis=0)))
        return [ generate_ant_solution(k) for k in ants_matrix ]
      
    def daemonActions(self, ants_matrix):
        pass
    
    def pheromoneUpdate(self, ants_matrix):
        delta_pheromone_matrix = self.generate_delta_pheromone(ants_matrix)
        self.update_pheromone(delta_pheromone_matrix)
        
    def generate_delta_pheromone(self, ants_matrix):
        return np.sum(ants_matrix, axis=0, dtype=float)
    
    def update_pheromone(self, delta_pheromone_matrix):
        self.pheromone_matrix = (1 - self.evaporation) * self.pheromone_matrix + delta_pheromone_matrix


    

