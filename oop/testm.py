'''
Created on Aug. 3, 2021

@author: zollen
@desc: When to use what?

We generally use class method to create factory methods. Factory methods return class 
object ( similar to a constructor ) for different use cases.

We generally use static methods to create utility functions.
'''

class A(object):
    
    def __init__(self, fname, lname):
        self.fname = fname
        self.lname = lname
        
    def foo(self, x):
        print(f"executing foo({self}, {x})")

    @classmethod
    def class_foo(cls, x):
        # class method access/modify the class variable
        print(f"executing class_foo({cls}, {x})")

    @staticmethod
    def static_foo(x):
        # static method cannot access/modify the class variable
        print(f"executing static_foo({x})")
    
    @property
    def name(self):
        return self.fname + " " + self.lname

   
    @name.setter
    def name(self, name):
        self.fname = name
        
    def __add__(self, other_self):
        return f"{self.lname} {other_self.lname}"
        
        
a = A("Stephen", "Kong")

a.foo(12)
a.class_foo(12)
a.static_foo(12)
print(a.name)
a.name = "Dragon"
print(a.name)
print(a.__dict__)

b = A("Sophie", "King")

print(a.name)
print(b.name)

print(a + b)