'''
Created on Apr. 15, 2022

@author: zollen
@Description: Association Rule mining
@example: https://www.youtube.com/watch?v=UP4ezNZfcH0&ab_channel=CSEGURUS
'''
import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)


dataset = [
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

if True:
    encoder = TransactionEncoder()
    transactions = pd.DataFrame(encoder.fit(dataset).transform(dataset), 
                            columns=['Item#1', 'Item#2', 'Item#3', 'Item#4', 'Item#5'])
else:
    transactions = pd.DataFrame({ 
                    'Item#1': [ True, False, False, True, True, False, True, True, True ],
                    'Item#2': [ True, True, True, True, False, True, False, True, True ],
                    'Item#3': [ False, False, True, False, True, True, True, True, True ],
                    'Item#4': [ False, True, False, True, False, False, False, False, False ],
                    'Item#5': [ True, False, False, False, False, False, False, True, False ]
    
                    })


print (transactions)


frequent_itemsets = apriori(transactions, min_support= 0.2, use_colnames=True, max_len = 3)
rules = association_rules(frequent_itemsets, metric="lift",  min_threshold = 1.5)
print(rules)
print("Rules identified: ", len(rules))