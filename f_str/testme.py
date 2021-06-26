'''
Created on Jun. 26, 2021

@author: zollen
@url: https://towardsdatascience.com/4-tricks-to-use-python-f-strings-more-efficiently-4f389e890514
'''

from datetime import datetime

me = 'Stephen'
print(F'Hello is {me}')

number = 3454353453
print(f"The value of the company is {number:,d}")

today = datetime.today().date()
print(f"Today is {today}")
print(f"Today is {today:%B %d, %Y}")
print(f"Today is {today:%m-%d-%Y}")

a = 4
b = 123
print(f"Product numbers are \n{a:03} \n{b:03}")


mylist = [1, 2, 4, 6, 3]
print(f"The list contains {len(mylist)} items.")
