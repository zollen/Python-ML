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
MaxLevel: maximum limited of tree depth
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
        
        for l in leaf1.. leafN in tree(i)
                                                         ( ∑ Residuals )
        OutputValue(leaf(l)) =  -------------------------------------------------------------  
                                ( ( ∑ (Previous Probability) * ( 1- Previous Probabilty) ) + λ )  
                    
    
    
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
        
        Blocks for Out-Of-Core Computation compress the massive data for minimizing the time 
        for accessing the slow hard drives.
        
        XgBoost can speed up building tree by only looking a random subset of features when
        deciding how to split the data.
        

Classification
--------------
result = log(i = 0) + ε * OutputValue(tree(1).targetNode) + ε * OutputValue(tree(2).targetNode) +
     + .... + ε * OutputValue(tree(N).targetNode)
FinalProbability = Sigmoid(result)     
     
     
     
     
AdaBoost 
========
https://www.youtube.com/watch?v=LsK-xG1cLYA

1. Using all features, AdaBoost combines a lot of weak leaners to make classifications. 
This weak learners are called stumps (one parent <with one feature> and two leaves)    
2. Some stumps get more say in classification than other stumps
3. Each stump is made by taking the previous stump's mistakes into account.
4. In each iteration, the training data get resampled based on the sample weights.

Training
---------
Initial Sample Weight for all data
Weight(0, ALL) = 1 / (number of training data)

for i in 1.. N
    for f in feature1.. featureF  (for building stump)
        for d in splitting_decision for feature(f)
            Calculating GiniIndex for each possible stump.         
        Pick the smallest GiniIndex of the splitting_decision of feature(f)
    Pick the smallest GiniIndex of feature amount all possible features
    
    Calculating the AmountOfSay of this stump by determing how well it classifies the data
        The Total Error for a stump is the sum of the weights assoicated with the incorrectly
        classified data. The Total error is always between 0 and 1. 0 means a perfect stump and
        1 is a horrible stump. TotalError should never equal to exactly 0 or 1. A small term
        1e-15 would be added into the Total Error before applying the following formula.
                       1       1 - TotalError
        AmountOfSay = --- log(----------------)
                       2         TotalError
                       
        AmountOfSay is close to 1 if the total error is small (very good predictions)
        AmountOfSay is close to 0 if the total error is 0.5 (no better than random). 
        AmountOfSay is close to -1 is the total error is large (always predicts the opposite).
    
    The incorrect predicted data are going to have heavier weight and other correctly predicted
    data are going to have less weight for the next stump.
        for d in all incorrectly predicted data
            Weight(i, d) = Weight(i - 1, d) x e^(AmountOfSay)
        Weight(i) is large if the AmountOfSay is high (the stump does a good job)
        Weight(i) is small if the AmountOfSay is small (the stump does a poor job)
    
        for d in all correctly predicted data
            Weight(i, d) = Weight(i - 1, d) x e^(-1 * AmountOfSay)    
        Weight(i) is large if the AmountOfSay is small
        Weight(i) is small if the AmountOfSay is high
        
        Normalize all data to ensure all sample weights sum up to 1.0
    
    Creating a new training data by randomly picked the training data(resampling) based on the
    sample weight. This new training data is going to be used for the next iteration. 


Classification
--------------
Stump(1)(Yes) * AmountOfSay(1) + Stump(2)(Yes) * AmountOfSay(2) + Stump(3)(Yes) * AmountOfSay(3) 
   + ... + Stump(N)(Yes) * AmountOfSay(N)
   
Stump(1)(No) * AmountOfSay(1) + Stump(2)(No) * AmountOfSay(2) + Stump(3)(No) * AmountOfSay(3)
   + ... + Stump(N)(No) * AmountOfSay(N)
   
The label with the highest score win   



"""