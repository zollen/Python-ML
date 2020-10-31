'''
Created on Oct. 28, 2020

@author: zollen
@decription: Assoication Rules mining
            https://www.youtube.com/watch?v=VB8KWm8MXss&ab_channel=CSEGURUS
            
 FP-Growth Algorithm

A three-step process is followed:
1. Construct a FP-Tree
    Frequent Pattern Tree is a tree-like structure that is made with the initial itemsets 
    of the database. The purpose of the FP tree is to mine the most frequent pattern. 
    Each node of the FP tree represents an item of the itemset.
    
    The root node represents null while the lower nodes represent the itemsets. 
    The association of the nodes with the lower nodes that is the itemsets with the other 
    itemsets are maintained while forming the tree.
    
2. Create a Conditional Database
    2.1) The first step is to scan the database to find the occurrences of the itemsets in 
        the database. This step is the same as the first step of Apriori. The count of 
        1-itemsets in the database is called support count or frequency of 1-itemset.

    2.2) The second step is to construct the FP tree. For this, create the root of the 
        tree. The root is represented by null.

    2.3) The next step is to scan the database again and examine the transactions. Examine 
        the first transaction and find out the itemset in it. The itemset with the max 
        count is taken at the top, the next itemset with lower count and so on. It means 
        that the branch of the tree is constructed with transaction itemsets in descending 
        order of count.

    2.4) The next transaction in the database is examined. The itemsets are ordered in 
        descending order of count. If any itemset of this transaction is already present in 
        another branch (for example in the 1st transaction), then this transaction branch 
        would share a common prefix to the root.

        This means that the common itemset is linked to the new node of another itemset in 
        this transaction.

    2.5) Also, the count of the itemset is incremented as it occurs in the transactions. 
        Both the common node and new node count is increased by 1 as they are created and 
        linked according to transactions.

    2.6) The next step is to mine the created FP Tree. For this, the lowest node is 
        examined first along with the links of the lowest nodes. The lowest node represents 
        the frequency pattern length 1. From this, traverse the path in the FP Tree. This 
        path or paths are called a conditional pattern base.

    Conditional pattern base is a sub-database consisting of prefix paths in the FP tree 
    occurring with the lowest node (suffix).

    2.7) Construct a Conditional FP Tree, which is formed by a count of itemsets in the 
    path. The itemsets meeting the threshold support are considered in the Conditional FP 
    Tree.

    2.8) Frequent Patterns are generated from the Conditional FP Tree.

'''

import os
from pathlib import Path
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import titanic_kaggle.lib.titanic_lib as tb
import warnings


pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')

def fillAge(src_df, dest_df):
    
    ages = src_df.groupby(['Title', 'Sex', 'SibSp', 'Parch'])['Age'].median()

    for index, value in ages.items():
            dest_df.loc[(dest_df['Age'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Sex'] == index[1]) &
                 (dest_df['SibSp'] == index[2]) &
                 (dest_df['Parch'] == index[3]), 'Age'] = value
                 
    ages = src_df.groupby(['Title', 'Sex', 'SibSp'])['Age'].median()

    for index, value in ages.items():
        dest_df.loc[(dest_df['Age'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Sex'] == index[1]) &
                 (dest_df['SibSp'] == index[2]), 'Age'] = value
               
    ages = src_df.groupby(['Title', 'Sex'])['Age'].median()

    for index, value in ages.items():
        dest_df.loc[(dest_df['Age'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Sex'] == index[1]), 'Age'] = value

## pd.qcut() based boundaries yields better result
def binAge(df):
    df.loc[df['Age'] < 14, 'Age'] = 7
    df.loc[(df['Age'] < 19) & (df['Age'] >= 14), 'Age'] = 17
    df.loc[(df['Age'] < 22) & (df['Age'] >= 19), 'Age'] = 20
    df.loc[(df['Age'] < 25) & (df['Age'] >= 22), 'Age'] = 24
    df.loc[(df['Age'] < 28) & (df['Age'] >= 25), 'Age'] = 27
    df.loc[(df['Age'] < 31.8) & (df['Age'] >= 28), 'Age'] = 30
    df.loc[(df['Age'] < 36) & (df['Age'] >= 31.8), 'Age'] = 34
    df.loc[(df['Age'] < 41) & (df['Age'] >= 36), 'Age'] = 38
    df.loc[(df['Age'] < 50) & (df['Age'] >= 41), 'Age'] = 46
    df.loc[(df['Age'] < 80) & (df['Age'] >= 50), 'Age'] = 70
    df.loc[df['Age'] >= 80, 'Age'] = 80

## pd.qcut() based boundaries yields better result    
def binFare(df):
    df.loc[df['Fare'] < 7.229, 'Fare'] = 4
    df.loc[(df['Fare'] < 7.75) & (df['Fare'] >= 7.229), 'Fare'] = 6
    df.loc[(df['Fare'] < 7.896) & (df['Fare'] >= 7.75), 'Fare'] = 7
    df.loc[(df['Fare'] < 8.05) & (df['Fare'] >= 7.896), 'Fare'] = 8
    df.loc[(df['Fare'] < 10.5) & (df['Fare'] >= 8.05), 'Fare'] = 9
    df.loc[(df['Fare'] < 13) & (df['Fare'] >= 10.5), 'Fare'] = 11
    df.loc[(df['Fare'] < 15.85) & (df['Fare'] >= 13), 'Fare'] = 14
    df.loc[(df['Fare'] < 24) & (df['Fare'] >= 15.85), 'Fare'] = 20
    df.loc[(df['Fare'] < 26.55) & (df['Fare'] >= 24), 'Fare'] = 25
    df.loc[(df['Fare'] < 33.308) & (df['Fare'] >= 26.55), 'Fare'] = 30
    df.loc[(df['Fare'] < 55.9) & (df['Fare'] >= 33.308), 'Fare'] = 45
    df.loc[(df['Fare'] < 83.158) & (df['Fare'] >= 55.9), 'Fare'] = 75
    df.loc[(df['Fare'] < 1000) & (df['Fare'] >= 83.158), 'Fare'] = 100

def calLength(rec):
    return len(rec['antecedents']) + len(rec['consequents'])   

def check0(bag):
    if 'Survived_0' in bag and len(bag) == 1:
        return True
    else:
        return False
    
def check1(bag):
    if 'Survived_1' in bag and len(bag) == 1:
        return True
    else:
        return False
     
PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25
test_df.loc[test_df['Embarked'].isna() == True, 'Embarked'] = 'S' 
test_df.loc[test_df['Fare'].isna() == True, 'Fare'] = 7.25


tb.reeigneeringTitle(train_df)
tb.reeigneeringTitle(test_df)

all_df = pd.concat( [ train_df, test_df ], ignore_index = True )

fillAge(all_df, train_df)
fillAge(all_df, test_df)

binAge(train_df)
binAge(test_df)
binFare(train_df)
binFare(test_df)

train_df.drop(columns = ['PassengerId', 'Title', 'Ticket', 'Name', 'Cabin'], inplace = True)

check_df = train_df.copy()

for name in check_df.columns:

    for val in check_df[name].unique():
        check_df[name + "_" + str(val)] = check_df[name].apply(lambda x : True if x == val else False)
    
    check_df.drop(columns = [name], inplace = True)
    

columns = [ 'antecedents', 'consequents', 'length', 'confidence', 'lift', 'conviction' ]

freq_df = fpgrowth(check_df, min_support=0.02, use_colnames=True)
freq_df = association_rules(freq_df, metric="confidence", min_threshold=0.7)
freq_df['length'] = freq_df.apply(calLength, axis = 1)    
freq_df['Survived_0'] = freq_df['consequents'].apply(check0)
freq_df['Survived_1'] = freq_df['consequents'].apply(check1)
print("== DEAD =============================================")
print(freq_df.loc[(freq_df['Survived_0'] == True) & (freq_df['length'] >= 7), columns])
print("== ALIVE ============================================")
print(freq_df.loc[(freq_df['Survived_1'] == True) & (freq_df['length'] >= 6), columns])
