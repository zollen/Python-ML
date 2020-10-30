'''
Created on Oct. 29, 2020

@author: zollen
@description: Frequent-Pattern growth (FP-growth)
              Association rules mining (apriori alternative )
'''
import pprint


class Node:
    def __init__(self, name):
        
        self.name = name
        self.count = 1
        self.children = {}
        
    def add(self, path):
        
        if len(path) <= 0:
            return
        
        name = path[0]
     
        if name not in self.children:
            current = Node(name)
            self.children[name] = current
        else:
            self.children[name].increment()  
            
        path.pop(0)    
        self.children[name].add(path)
         
    def increment(self):
        self.count += 1    
        
    def format(self, indent):
        
        info = ""
        for _ in range(indent):
            info += " "
          
        return info + "[" + self.name + "] ==> " + str(self.count) + "\n"
    
    def traverse(self, name, allpaths):
        self._traverse(name, allpaths, [])
        
    def _traverse(self, name, allpaths, path):
        
        if self.name != 'Null':
            path.append(self.name)
         
        if name == self.name and path not in allpaths:
            allpaths.append(path)
            return
                
        for child in self.children:
            self.children[child]._traverse(name, allpaths, path.copy())
                  
    def describe(self, indent):
        
        info = self.format(indent)
        
        for name in self.children:
            info += self.children[name].describe(indent + 3)
            
        return info
        
    def __str__(self):
        return self.describe(0)
        
        
transactions = [
    ["L1", "L2", "L5"],
    ["L2", "L4"],
    ["L2", "L3"],
    ["L1", "L2", "L4"],
    ["L1", "L3"],
    ["L2", "L3"],
    ["L1", "L3"],
    ["L1", "L2", "L3", "L5"],
    ["L1", "L2", "L3"]
    ]    

sort_order = {}

for trans in transactions:
    for tt in trans:
        if tt not in sort_order:
            sort_order[tt] = 0
        sort_order[tt] += 1
    
       
for trans in transactions:
    trans.sort(key = lambda x : sort_order[x], reverse = True)


root = Node('Null')

for trans in transactions:
    root.add(trans)


print(root)
print("Starting from lowest to highest support count element => L5, L4, L3, L1")

kk = []
root.traverse('L5', kk)
print(kk)