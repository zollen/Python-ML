'''
Created on Jul. 22, 2021

@author: zollen
'''

import pandas as pd

df = pd.DataFrame({
        'key'   : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        'month' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                   1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'prices': [0.1, 0.2, 0.1, 0.5, 0.0, 0.3, 0.2, 0.0, 0.3, 0.1, 
                   0.2, 0.3, 0.2, 0.1, 0.6, 0.2, 0.1, 0.1, 0.0, 0.1],
        'cnt'   : [ 0, 0, 1, 3, 1, 0, 0, 2, 0, 0, 
                   0, 0, 1, 1, 0, 5, 0, 0, 0, 0 ] 
        })



    

grps = df.groupby('key')

for grp in grps.groups:
    members = grps.get_group(grp).sort_values('month')
    print("Group: ", grp)
    print(members)



print("Number of groups: ", grps.ngroups)

print("first group")
print("===========")
print(grps.first())
print("last group")
print("===========")
print(grps.last())
#print(calculate(df['cnt']))
