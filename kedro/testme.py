'''
Created on May 10, 2021

@author: zollen
@url: https://jlgjosue.medium.com/kedro-the-best-python-framework-for-data-science-fda6d8503646
@desc: Kedro Framework, being open-source created by Quantumblack, widely used to code 
        in Python in a reproducible, sustainable and modular way to create “batch” 
        pipelines with several “steps”.
        This Framework has been increasingly gaining space and being adopted by the 
        community, especially when it is necessary to create a sequential execution 
        “mat” with different steps; this fact has happened mainly in the development of 
        codes focused on data science due to its ease of use together Python code, rich 
        documentation, as well as being extremely simple and intuitive.
'''

from kedro.pipeline import node
from kedro.pipeline import Pipeline
from kedro.io import DataCatalog, MemoryDataSet


'''
Node
====
It is a wrapper for a Python function that names the inputs and outputs of 
that function” that is; it is a block of code that can direct the execution of a 
certain sequence of codes or even other blocks.
'''

# Preparing the "nodes"
def return_greeting(name):
    return "Hello! " + name

def return_insulting(name):
    return "Damn You! " + name

#defining the node that will return
return_greeting_node = node(func=return_greeting, inputs="stephen", outputs="my salutation")
print(return_greeting_node)

return_insulting_node = node(func=return_insulting, inputs="jean", outputs="how are you")
print(return_insulting_node)



'''
Pipeline
========
A pipeline organizes the dependencies and order of execution of a collection of nodes 
and connects inputs and outputs, maintaining its modular code. The pipeline determines 
the order of execution of the node by resolving dependencies and does not necessarily 
execute the nodes in the order in which they are transmitted.
'''

pipeline = Pipeline([return_greeting_node, return_insulting_node])
print(pipeline)

'''
DataCatalog
===========
A DataCatalog is a Kedro concept. It is the record of all data sources that the project 
can use. It maps the names of node inputs and outputs as keys in a DataSet, which is a 
Kedro class that can be specialized for different types of data storage. Kedro uses a 
MemoryDataSet for data that is simply stored in memory. 
'''

data_catalog = DataCatalog({"stephen": MemoryDataSet()})

