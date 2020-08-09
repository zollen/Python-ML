'''
Created on Jul. 5, 2020

@author: zollen
'''

from functools import reduce

my_pets = ['alfred', 'tabitha', 'william', 'arla']

uppered_pets = list(map(str.upper, my_pets))

print(uppered_pets)

circle_areas = [3.56773, 5.57668, 4.00914, 56.24241, 9.01344, 32.00013]

result = list(map(round, circle_areas, [ 2, 2, 2, 2, 2, 2 ]))

print(result)

my_strings = ['a', 'b', 'c', 'd', 'e']
my_numbers = [1,2,3,4,5]


print(list(zip(my_strings, my_numbers)))
print(list(map(lambda x, y: (x, y), my_strings, my_numbers)))

scores = [66, 90, 68, 59, 76, 60, 88, 74, 81, 65]

def is_A_student(score):
    return score > 75

print(list(filter(is_A_student, scores)))
print(list(filter(lambda score : score > 75, scores)))

dromes = ("demigod", "rewire", "madam", "freer", "anutforajaroftuna", "kiosk")
print(list(filter(lambda word: word == word[::-1], dromes)))

numbers = [3, 4, 6, 9, 34, 12]

def custom_sum(first, second):
    return first + second

print(reduce(custom_sum, numbers))
print(reduce(lambda x, y : x + y, numbers))