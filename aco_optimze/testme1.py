'''
Created on Mar 6, 2024

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

bb = np.array([[1,2,3],[4,5,6]])
cc = np.array([[7,8,9],[10,11,12]])
print(bb)
print(cc)
print("================")
# generate_probabilities
print(bb + cc)
print(np.multiply(bb,cc))
print(np.power(bb, 3))
print(np.sum(bb, axis=1))
print(bb/np.sum(bb, axis=1, keepdims=True))

# generate_ants_solution
aa = np.zeros([5, 4])
aa[...,0] = np.random.choice([2,4,5,6], size=(5))
print(aa)

# generate_delta_pheromone
aa = [
        [
            [1,2,3,4], 
            [3,5,7,8]
        ],
        [
            [6,3,1,5], 
            [8,2,9,10]
        ],
        [
            [4,3,2,1],
            [2,4,5,1]
        ]
      ]
print("============")
print(aa[0])
print(aa[0][0])
print(aa[0][0][0])
print(np.sum(aa, axis=0))

# generate_ants_solution
print("============")
def myfunc(x):
    print(">>> ", x)
    x[0][0] = -1
    return x
    
bb = [ myfunc(x) for x in aa ]
print(bb)


print("=================")
aa = np.array([[1,2,0,3],[0,0,0,0]], dtype=float)
bb = np.sum(aa, axis=1, keepdims=True)
print(np.divide(aa, bb, out=np.zeros_like(aa), where=bb!=0))

print("=================")
aa[np.where(aa != 0)] = 999
print(aa)

print("====================")
x = [0.2, 0.1, 0.5, 0.0, 0.0, 0.2]
samples = np.random.choice(range(len(x)), 1, p=x)
print(samples[0])

print("=================")
aa[np.where(aa != 0)] = 1
print(aa) 

print("==============")
print(np.divide(1, aa, where=aa!=0))