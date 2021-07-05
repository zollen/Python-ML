'''
Created on Jul. 5, 2021

@author: zollen
'''

import time
from numba import jit
from  functools import *


@lru_cache
def factorial1(n):
    return n * factorial1(n-1) if n else 1


start = time.time()
res = factorial1(300)
end = time.time()
print("TIME: ", end - start)

@jit(nopython=True)
def factorial2(n):
    return n * factorial2(n-1) if n else 1


start = time.time()
res = factorial2(300)
end = time.time()
print("TIME: ", end - start)

