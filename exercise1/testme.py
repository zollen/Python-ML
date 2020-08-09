'''
Created on Jul. 2, 2020

@author: zollen

'''

def generator():
    i = 1
    a = 1
    b = 1
    
    while(1):
        yield [ i, a ]
        a, b = b, a + b
        i += 1
    
    
for i, n in generator():
    if ( n > 100 ):
        break
    print(i, n)
    
    
for _ in range(3):
    print(_)