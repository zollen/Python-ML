'''
Created on Jul. 5, 2020

@author: zollen
'''
import functools
import time
import math


def debug(func):

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)           
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        
        return value

    return wrapper_debug
  
def measureTime(func):
    
    @functools.wraps(func)
    def process(*args, **kwargs):
        
        begin = time.time()
        
        func(*args, **kwargs)
        
        end = time.time()
        
        print("Total Time taken in: [", func.__name__, "]", end - begin)
    
   
    return process
        

@debug
@measureTime          
def sayHello(msg):
    time.sleep(0.5)
    print(msg)
    
def sayMe():
    print("Good Morning!")
 
@debug   
def approx_e(terms=18):
    return sum(1.0 / math.factorial(n) for n in range(terms))

    
sayHello("Greeting!")    
sayHello(msg="cool!")   

help(sayHello)



print("Approx E: %f" % approx_e(50))


