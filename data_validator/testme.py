'''
Created on Apr. 1, 2022

@author: zollen
@url: https://towardsdatascience.com/validate-your-pandas-dataframe-with-pandera-2995910e564
@desc: That is why in this article we will learn about Pandera, a simple Python library for validating a pandas 
        DataFrame.

'''

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, Check, check_input, check_output


fruits = pd.DataFrame(
    {
        "name": ["apple", "banana", "apple", "orange"],
        "store": ["Aldi", "Walmart", "Walmart", "Aldi"],
        "price": [2, 1, 3, 4],
    }
)


available_fruits = ["apple", "banana", "orange"]
nearby_stores = ["Aldi", "Walmart"]

schema = pa.DataFrameSchema(
    {
        "name": Column(str, Check.isin(available_fruits)),
        "store": Column(str, Check.isin(nearby_stores)),
        "price": Column(int, Check.less_than(4)),
    }
)

if False:
    print(schema.validate(fruits))

'''
We can also create custom checks using lambda . In the code below, 
Check(lambda price: sum(price) < 20) checks if the sum of the column price is less than 20.
'''
schema = pa.DataFrameSchema(
    {
        "name": Column(str, Check.isin(available_fruits)),
        "store": Column(str, Check.isin(nearby_stores)),
        "price": Column(
            int, [Check.less_than(5), Check(lambda price: sum(price) < 20)]
        ),
    }
)

out_schema = pa.DataFrameSchema(
    {
        "price": Column(
            int, [Check.less_than(5)]
        ),
    }
)

if False:
    print(schema.validate(fruits))
    
@check_input(schema)    
def check_total(fruits: pd.DataFrame):
    return fruits

@check_output(out_schema)
def check_me(fruits: pd.DataFrame):
    return fruits


print(check_total(fruits))
print("====")
print(check_me(fruits))
print("====")

fruits = pd.DataFrame(
    {
        "name": ["apple", "banana", "apple", "orange"],
        "store": ["Aldi", "Walmart", "Walmart", np.nan],
        "price": [2, 1, 3, 4],
    }
)

schema = pa.DataFrameSchema(
    {
        "name": Column(str, Check.isin(available_fruits)),
        "store": Column(str, Check.isin(nearby_stores), nullable=True, unique=False),
        "price": Column(int, Check.less_than(5)),
    }
)

print(schema.validate(fruits))
    
