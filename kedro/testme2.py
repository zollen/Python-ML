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
from kedro.runner import SequentialRunner

'''
Runner
======
The Runner is an object that runs the pipeline. Kedro resolves the order in which the 
nodes are executed.
1. Kedro first performs return_greeting_node. This performs return_greeting, which 
    receives no input, but produces the string “Hello”.
2. The output string is stored in the MemoryDataSet called my_salutation. Kedro then 
    executes the second node, join_statements_node.
3. This loads the my_salutation dataset and injects it into the join_statements function.
4. The function joins the incoming greeting with “Kedro!” to form the output 
    string “Hello Kedro!”
5. The pipeline output is returned in a dictionary with the key my_message.
'''

# Prepare the "data catalog"
data_catalog = DataCatalog({"my_salutation": MemoryDataSet()})

# Prepare the first "node"
def return_greeting():
    return "Hello"
return_greeting_node = node(return_greeting, inputs=None, outputs="my_salutation")

# Prepare the second "node"
def join_statements(greeting):
    return f"{greeting} Kedro!"

join_statements_node = node(
    join_statements, inputs="my_salutation", outputs="my_message"
)

# Assign "nodes" to a "pipeline"
pipeline = Pipeline([return_greeting_node, join_statements_node])

# Create a "runner" to run the "pipeline"
runner = SequentialRunner()

# Execute a pipeline
print(runner.run(pipeline, data_catalog))

'''
Another way to visualize the execution of pipelines in kedro is using the kedro-viz 
plugin.
'''
