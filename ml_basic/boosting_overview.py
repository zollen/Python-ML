'''
Created on Aug. 14, 2020

@author: zollen
'''
"""
Random Forest
=============
https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&t=335s

Training
--------
for i in 1..N
    Randomly pick training data for a new training sample (resampling)
    Randomly pick a fix size subset of the features for building a full size tree. 

Classification
--------------
The label receives the most votes win out.



Gradient Boost
==============
https://www.youtube.com/watch?v=jxuNLH5dXCs

Training
--------
Each tree utilises all features
Residual = Observed(i) - Predict(i)
eta ε: learning rate
decision threshold: 0.5
Sigmoid(val) = e^(val) / ( 1 + e^(val) )   

Inital Prediction Prob(0): log(odds) = log(num(Yes)/num(No))
Residual of each leaf = Observed(i) - Prob(i) 

Log(i = 0) = log(num(Yes)/num(No))
Prob(i = 0) = Sigmoid(log(i = 0))

for i in 1.. N
    for j in leaf1...leafJ in tree[i]
                                               ∑ (Residuals of leaf(j))
        OutputValue(leaf(j)) =   ----------------------------------------------------------  
                               ( ∑ (Probability(i - 1, j)) * ( 1- Probabilty(i - 1, j)) )  
    for k in 1..K in training data
        odd(k) = Prob(i - 1) + ε * OutputValue(tree[i](k))
        Prob(i, k) = Sigmoid(odd(k)) 
        
            
Classification
--------------
result = log(i = 0) + ε * OutputValue(tree(1).targetNode) + ε * OutputValue(tree(2).targetNode) +
     + .... + ε * OutputValue(tree(N).targetNode)
FinalProbability = Sigmoid(result)    
If finalProbability < threshold of 0.5, then No
If finalProbabilty >= threshold of 0.5, then Yes      

    
 
XgBoost
========
https://www.youtube.com/watch?v=8b1JEDvenQU&t=267s

1. Gradient Boost-(ish) - similar as Graident Boost
2. Regulaization
3. A Unique Regression Tree
4. Approximate Greedy Algorithm
5. Parallel Learning
6. Weighted Quantile Sketch
7. Sparsity-Aware Split Finding
8. Cache-Aware Access
9. Blocks for Out-Of-Core Computation

Training
---------
Each tree limited by a max number of level
Residual = Observed(i) - Predict(i)
lambda λ: regularization, λ > 0 reduces the prediction's sensitivity to isolated observations
gamma γ: for deciding node pruning
Cover: The mimimum residual value in each node: (∑ (Probability(i - 1)) * ( 1- Probabilty(i - 1))) 
min_child_weight = Cover
eta ε: learning rate
Sigmoid(val) = e^(val) / ( 1 + e^(val) )  
Inital Default Prediction: 0.5  (changable)
Residual of each data = Observed(i) - Log(odds) 

Prob(i = 0) = 0.5
log(i = 0) = log(0.5 / (1 - 0.5)) = 0

for i in 1.. N
    for l in level1.. MaxLevel in tree(i)
        for s in splitting_threshold1...splitting_thresholdS from one random feature
            for (k in leaf1.. leafN in level(l)
                                                ( ∑ (Residuals of leaf(l, k)) )^2
                SimilarityScore(leaf(l, k)): -------------------------------------------------  
                                        ( ∑ (Probability(i - 1)) * ( 1- Probabilty(i - 1)) + λ )  
                If Cover(leaf(l, k)) >= Residual(l, k), then remove node(l, k)                         
            Gain = SimilarityScore(left leaf) + SimilarityScore(right leaf) - SimilarityScore(parent)
            If Gain(parent) - γ > 0, then parent stays, 
            If Gain(parent) - γ <= 0, then parent get pruned 
            
        The largest Gain splitting threshold parent is chosen for level(l) - Greedy Algorithm
        Greedy Algorithm makes decision based on the current state without looking ahead, but 
        it builds tree relatively quickly. It uses quantiles for s (all possible splitting
        decisions at level(l)). XgBoost use 33 quantiles by default. 
        
        Using parallel learning to seperate the massive amount of data into segments so 
        each segment can be processed by individual computer at the same time.  
        Weighted Qunatile Sketch merges the data with similar confident level of predictions 
        into an approximate histrogram. Each histogram is further divided into weigthed 
        quantiles that put observations with low confidence predictions into quantiles with 
        fewer observations.
        
        Sparsity-Aware split tells us how to build tree with missing data and how to handle
        new observations where there is missing data.
       
        Cache-Aware access puts Gradients and Hessians into CPU cache so that it can quickly
        calculates Similarity Scores and Output Values.
        
        locks for Out-Of-Core Computation compress the massive data for minimizing the time 
        for accessing the slow hard drives.
        
        XgBoost can speed up building tree by only looking a random subset of features when
        deciding how to split the data.
        
    
    for l in leaf1.. leafN in tree(i)
                                                         ( ∑ Residuals )
        OutputValue(leaf(l)) =  -------------------------------------------------------------  
                                ( ( ∑ (Previous Probability) * ( 1- Previous Probabilty) ) + λ )  
                    
    

Classification
--------------
result = log(i = 0) + ε * OutputValue(tree(1).targetNode) + ε * OutputValue(tree(2).targetNode) +
     + .... + ε * OutputValue(tree(N).targetNode)
FinalProbability = Sigmoid(result)     
     
     
     
     
AdaBoost 
========
https://www.youtube.com/watch?v=LsK-xG1cLYA

1. AdaBoost combines a lot of weak leaners to make classifications. This weak learners are called   
stumps (one parent and two leaves)    
2. Some stumps get more say in classification than other stumps
3. Each stump is made by taking the previous stump's mistakes into account.


     
     
"""