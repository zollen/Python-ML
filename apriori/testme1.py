'''
Created on Oct. 27, 2020

@author: zollen
@Description: Association Rule mining
@example: https://www.youtube.com/watch?v=UP4ezNZfcH0&ab_channel=CSEGURUS

Apriori Algorithm

A two-step process is followed:
1. The join step: To find L(k) a set of candidate k-itemsets is generated by joining L(k-1) 
    with itself. This set of candidates is denoted C(k).
2. The prune step: C(k) is a superset of L(k-1), that is, its member may or may not be 
    frequent, but all of the frequent k-itemsets are included in C(k). To reduce the size of
    C(k) the Aprior property is used.

Understanding: technically the L(k) is from joining L(k-1) only! 
But all subsets of a frequent itemset must be frequent(Apriori propertry). If an itemset 
is infrequent, all its supersets will be infrequent.
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
TOTAL_TRANSACTIONS = 9

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
print()

'''
Generate strong assoication rules from itemsets (where strong association fules satisfy
both mimimum support and minimum confidence)

1. For each frequent itemset l, generate all nonempty subsets of l
2. For every nonemptpy subset s of l, output the rule "s => (l - s)" if:
    support_count(l) / support_count(s) >= min_conf, where min_conf is the minimum
    confidence threshold.
    
    confidence(A => B) = P(B|A) = support_count(A and B) / support_count(A)
    
'''
items_set['1,2,3'] = 2 / TOTAL_TRANSACTIONS
items_set['1,2,5'] = 2 / TOTAL_TRANSACTIONS
items_set['1'] = 6 / TOTAL_TRANSACTIONS
items_set['2'] = 7 / TOTAL_TRANSACTIONS
items_set['3'] = 6 / TOTAL_TRANSACTIONS
items_set['5'] = 2 / TOTAL_TRANSACTIONS
items_set['1,2'] = 4 / TOTAL_TRANSACTIONS
items_set['1,3'] = 4 / TOTAL_TRANSACTIONS
items_set['2,3'] = 2 / TOTAL_TRANSACTIONS
items_set['1,5'] = 2 / TOTAL_TRANSACTIONS
items_set['2,5'] = 2 / TOTAL_TRANSACTIONS

conf, lift, conv = {}, {}, {}

conf['1=>2,3'] = items_set['1,2,3'] / items_set['1']
conf['2=>1,3'] = items_set['1,2,3'] / items_set['2']
conf['3=>1,2'] = items_set['1,2,3'] / items_set['3']
conf['1,2=>3'] = items_set['1,2,3'] / items_set['1,2']
conf['2,3=>1'] = items_set['1,2,3'] / items_set['2,3']
conf['1,3=>2'] = items_set['1,2,3'] / items_set['1,3']
conf['1=>2,5'] = items_set['1,2,5'] / items_set['1']
conf['2=>1,5'] = items_set['1,2,5'] / items_set['2']
conf['5=>1,2'] = items_set['1,2,5'] / items_set['5']
conf['1,2=>5'] = items_set['1,2,5'] / items_set['1,2']
conf['2,5=>1'] = items_set['1,2,5'] / items_set['2,5']
conf['1,5=>2'] = items_set['1,2,5'] / items_set['1,5']

lift['1=>2,3'] = items_set['1,2,3'] / (items_set['1'] * items_set['2,3'])
lift['2=>1,3'] = items_set['1,2,3'] / (items_set['2'] * items_set['1,3'])
lift['3=>1,2'] = items_set['1,2,3'] / (items_set['3'] * items_set['1,2'])
lift['1,2=>3'] = items_set['1,2,3'] / (items_set['1,2'] * items_set['3'])
lift['2,3=>1'] = items_set['1,2,3'] / (items_set['2,3'] * items_set['1'])
lift['1,3=>2'] = items_set['1,2,3'] / (items_set['1,3'] * items_set['2'])
lift['1=>2,5'] = items_set['1,2,5'] / (items_set['1'] * items_set['2,5'])
lift['2=>1,5'] = items_set['1,2,5'] / (items_set['2'] * items_set['1,5'])
lift['5=>1,2'] = items_set['1,2,5'] / (items_set['5'] * items_set['1,2'])
lift['1,2=>5'] = items_set['1,2,5'] / (items_set['1,2'] * items_set['5'])
lift['2,5=>1'] = items_set['1,2,5'] / (items_set['2,5'] * items_set['1'])
lift['1,5=>2'] = items_set['1,2,5'] / (items_set['1,5'] * items_set['2'])

conv['1=>2,3'] = 0 if 1 - conf['1=>2,3'] == 0 else (1 - items_set['2,3']) / (1 - conf['1=>2,3'])
conv['2=>1,3'] = 0 if 1 - conf['2=>1,3'] == 0 else (1 - items_set['1,3']) / (1 - conf['2=>1,3'])
conv['3=>1,2'] = 0 if 1 - conf['3=>1,2'] == 0 else (1 - items_set['1,2']) / (1 - conf['3=>1,2'])
conv['1,2=>3'] = 0 if 1 - conf['1,2=>3'] == 0 else (1 - items_set['3']) / (1 - conf['1,2=>3'])
conv['2,3=>1'] = 0 if 1 - conf['2,3=>1'] == 0 else (1 - items_set['1']) / (1 - conf['2,3=>1'])
conv['1,3=>2'] = 0 if 1 - conf['1,3=>2'] == 0 else (1 - items_set['2']) / (1 - conf['1,3=>2'])
conv['1=>2,5'] = 0 if 1 - conf['1=>2,5'] == 0 else (1 - items_set['2,5']) / (1 - conf['1=>2,5'])
conv['2=>1,5'] = 0 if 1 - conf['2=>1,5'] == 0 else (1 - items_set['1,5']) / (1 - conf['2=>1,5'])
conv['5=>1,2'] = 0 if 1 - conf['5=>1,2'] == 0 else (1 - items_set['1,2']) / (1 - conf['5=>1,2'])
conv['1,2=>5'] = 0 if 1 - conf['1,2=>5'] == 0 else (1 - items_set['5']) / (1 - conf['1,2=>5'])
conv['2,5=>1'] = 0 if 1 - conf['2,5=>1'] == 0 else (1 - items_set['1']) / (1 - conf['2,5=>1'])
conv['1,5=>2'] = 0 if 1 - conf['1,5=>2'] == 0 else (1 - items_set['2']) / (1 - conf['1,5=>2'])

print("============ (1,2,3) ================")
print("        Confidence    Lift    Conviction")
print("1 => 2 and 3: %0.2f    %0.2f    %0.2f" % (conf['1=>2,3'], lift['1=>2,3'], conv['1=>2,3']))
print("2 => 1 and 3: %0.2f    %0.2f    %0.2f" % (conf['2=>1,3'], lift['2=>1,3'], conv['2=>1,3']))
print("3 => 1 and 2: %0.2f    %0.2f    %0.2f" % (conf['3=>1,2'], lift['3=>1,2'], conv['3=>1,2']))
print("1 and 2 => 3: %0.2f    %0.2f    %0.2f" % (conf['1,2=>3'], lift['1,2=>3'], conv['1,2=>3']))
print("2 and 3 => 1: %0.2f    %0.2f    %0.2f" % (conf['2,3=>1'], lift['2,3=>1'], conv['2,3=>1']))
print("1 and 3 => 2: %0.2f    %0.2f    %0.2f" % (conf['1,3=>2'], lift['1,3=>2'], conv['1,3=>2']))

print("Lets say the minimum confidence threshold is 0.6")
print("Strong Assoication Rules: Conf(2 and 3 => 1)")

print()
print("============ (1,2,5) ================")
print("        Confidence    Lift    Conviction")
print("1 => 2 and 5: %0.2f,   %0.2f    %02.f" % (conf['1=>2,5'], lift['1=>2,5'], conv['1=>2,5']))
print("2 => 1 and 5: %0.2f,   %0.2f    %0.2f" % (conf['2=>1,5'], lift['2=>1,5'], conv['2=>1,5']))
print("5 => 1 and 2: %0.2f,   %0.2f    %0.2f" % (conf['5=>1,2'], lift['5=>1,2'], conv['5=>1,2']))
print("1 and 2 => 5: %0.2f,   %0.2f    %0.2f" % (conf['1,2=>5'], lift['1,2=>5'], conv['1,2=>5']))
print("1 and 5 => 2: %0.2f,   %0.2f    %0.2f" % (conf['1,5=>2'], lift['1,5=>2'], conv['1,5=>2']))
print("2 and 5 => 1: %0.2f,   %0.2f    %0.2f" % (conf['2,5=>1'], lift['2,5=>1'], conv['2,5=>1']))

print("Lets say the minimum confidence threshold is 0.6")
print("Strong Assoication Rules: Conf(1 and 2 => 5), Conf(1 and 5 => 2) and Conf(2 and 5 => 1)")