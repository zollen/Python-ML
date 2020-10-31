'''
Created on Oct. 29, 2020

@author: zollen
@description: Frequent-Pattern growth (FP-growth)
              Association rules mining (apriori alternative. FP-grow use less memory )
@url https://www.youtube.com/watch?v=VB8KWm8MXss&ab_channel=CSEGURUS              
'''
import pprint
import random


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
        self._traverse(name, allpaths, {})
        
    def _traverse(self, name, allpaths, path):
        
        if self.name != 'Null' and self.name != name:
            path[self.name] = self
         
        if name == self.name and len(path) > 0 and path not in allpaths:
            allpaths.append([ path, self.count ] )
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
        
pp = pprint.PrettyPrinter(indent=3) 

        
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


print("First Step: sorting the elememts based on the support counts")
sort_order = {}

for trans in transactions:
    for tt in trans:
        if tt not in sort_order:
            sort_order[tt] = 0
        sort_order[tt] += 1
        
print("Sort Order: ", sort_order)

for trans in transactions:
    trans.sort(key = lambda x : sort_order[x], reverse = True)
    
pp.pprint(transactions)


print("Second Step: Building a FP-Tree")
root = Node('Null')

for trans in transactions:
    root.add(trans.copy())

print(root)

print("Third Step: Building a conditional database")
print("Third Step: Starting from lowest to highest support count element => L5, L4, L3, L1")
print("L2 is the last and also the highest support count element. we can ignore it")
print("Third Step: Conditional Pattern Base")
conditional_pattern_base = {
    "L5": [],
    "L4": [],
    "L3": [],
    "L1": []
    }


for key in conditional_pattern_base:
    _first1 = True
    root.traverse(key, conditional_pattern_base[key])
    out = "{"
    for items, count in conditional_pattern_base[key]:
        out += '{' if _first1 == True else ", {"
        _first1 = False  
        
        _first2 = True
        for id in items:
            out += "" if _first2 == True else ","
            out += items[id].name
            _first2 = False
            
        out += ": " + str(count) + "}"
    out += "}"  
    print(key, " ==> ", out)


print("Fourth Step: Building Conditional FP-Tree")
THRESHOLD = 2
conditional_fp_tree = {
    "L5": {},
    "L4": {},
    "L3": {},
    "L1": {}
    }


objnames = {}


for key in conditional_pattern_base:
    tmp = {}
    names = []
    for items, count in conditional_pattern_base[key]:
        for itm in items:
            obj = items[itm]
            objnames[obj] = items[itm].name
            if obj not in tmp:
                tmp[obj] = 0 
                
            tmp[obj] += count
         
    for obj in tmp:
        if tmp[obj] < THRESHOLD:
            continue
        conditional_fp_tree[key][obj] = tmp[obj]
        
out = ''
for key in conditional_fp_tree:  
    out += key + "  ==> {"
    _first = True
    for obj in conditional_fp_tree[key]:
        if _first == False:
            out += ", "
        _first = False
        out += objnames[obj] + ": " + str(conditional_fp_tree[key][obj])
    out += "}\n" 
print(out)
    


print("Fifth Step: Building Frequent Patterns Generated")
