'''
Created on Apr. 15, 2022

@author: zollen
'''
import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder


pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

basket = pd.read_csv("../data/Groceries_dataset.csv")

basket['itemDescription'] = basket['itemDescription'].transform(lambda x: [x])
basket = basket.groupby(['Member_number','Date'])['itemDescription'].sum().reset_index(drop=True)

encoder = TransactionEncoder()
transactions = pd.DataFrame(encoder.fit(basket).transform(basket), columns=encoder.columns_)


frequent_itemsets = apriori(transactions, min_support= 6/len(basket), use_colnames=True, max_len = 2)
rules = association_rules(frequent_itemsets, metric="lift",  min_threshold = 1.5)
print(rules)
print("Rules identified: ", len(rules))


# One approach that can be proposed is to find out which products drive the sales of Whole Milk and 
# offer discounts on those products instead.
milk_rules = rules[rules['consequents'].astype(str).str.contains('whole milk')]
milk_rules = milk_rules.sort_values(by=['lift'],ascending = [False]).reset_index(drop = True)

print(milk_rules.head())