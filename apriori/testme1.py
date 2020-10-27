'''
Created on Oct. 27, 2020

@author: zollen
@Description: Association Rule mining
@example: https://www.youtube.com/watch?v=UP4ezNZfcH0&ab_channel=CSEGURUS
'''
import pprint

transactions = [
        ['1', '2', '5'],
        ['2', '4'],
        ['2', '3'],
        ['1', '2', '4'],
        ['1', '3'],
        ['2', '3'],
        ['1', '3'],
        ['1', '2', '3', '5'],
        ['1', '2', '3']
    ]

pp = pprint.PrettyPrinter(indent=3, width = 10) 


def join(llist):
    
    val = ''
    for item in llist:
        if len(val) > 0:
            val += ','
        val += item
        
    return val

def display_items_set(itemsset, size):
    
    new_items_set = {}
    for key, val in itemsset.items():
        items = key.split(",")
        if len(items) == size:
            new_items_set[key] = val
        
    return new_items_set
    
def filter_items_set(itemsset, threshold):
    
    new_items_set = {}
    for key, val in itemsset.items():
        if itemsset[key] >= threshold:
            new_items_set[key] = val
        
    return new_items_set
    
def count_items_set(itemsset):
    
    for key in itemsset.keys():
        
        items = str(key).split(",")
        
        for entry in transactions:
            if set(items).issubset(set(entry)):
                itemsset[key] += 1
    
    
def add_items_set(itemsset, size):
    
    targets = []
    for lkey in itemsset.keys():  
        
        llist = lkey.split(",")
        
        for rkey in itemsset.keys():
            
            rlist = rkey.split(",")
            
            tmp = llist + rlist
            
            tmp.sort()
            
            if len(tmp) != size or len(tmp) != len(set(tmp)):
                continue
    
            for items in transactions:
                if set(tmp).issubset(set(items)):
                    targets.append(tmp)
                    break 
    
    for target in targets + list(itemsset.keys()):
        itemsset[join(target)] = 0
        

 
N = 2
items_set = {}


for items in transactions:
    for item in items:
        if item not in items_set:
            items_set[item] = 0
        
        items_set[item] += 1

items_set = filter_items_set(items_set, N)
pp.pprint(items_set) 


                 
for size in range(2, 5):
    
    add_items_set(items_set, size)
    
    count_items_set(items_set)

    items_set = filter_items_set(items_set, N)
 
    pp.pprint(display_items_set(items_set, size))   


print("Final Answer (1,2,3) and (1,2,5)")    