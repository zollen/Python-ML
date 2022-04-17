'''
Created on Apr. 17, 2022

@author: zollen
@url: https://towardsdatascience.com/analyze-and-visualize-urls-with-network-graph-ee3ad5338b69
@desc: APi for handling URL and creatin graph
'''
import pandas as pd
from yarl import URL
import graphistry


pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

url = URL("https://github.com/search?q=data+science")
print(url.scheme)
print(url.host)
print(url.path)
print(url.query_string)
print("================================================================")

data = pd.read_csv("../data/URL Classification.csv", names=["url", "Type"], index_col=0)
sample = data.sample(10000, random_state=1)
print(sample.head())
print("================================================================")

sample["url"] = sample["url"].apply(lambda url: URL(url))

processed = sample.assign(
    host=sample.url.apply(lambda url: url.host),
    path=sample.url.apply(lambda url: url.path),
    name=sample.url.apply(lambda url: url.name),
    scheme=sample.url.apply(lambda url: url.scheme),
    query=sample.url.apply(lambda url: url.query_string),
)

print(processed.head())
print("================================================================")


# Group by the columns Type and host and get the count
group = processed.groupby(["Type", "host"]).agg(count=("url", "count"))

# Get the top 5 most popular hosts for each type
sorted_group = group.sort_values(by="count", ascending=False).reset_index()
largest = sorted_group.groupby("Type").head(5).sort_values(by='Type')

# View the bottom 10 rows
print(largest.tail(100))
print("================================================================")

graphistry.register(api=3, username='zollen', password='os2warp123')
edges = largest[["Type", "host"]]
def create_node_df(df: pd.DataFrame, col_name: str):
    """Create a node DataFrame for a specific column"""
    nodes = (
        df[[col_name]]
        .assign(type=col_name)
        .rename(columns={col_name: "node"})
        .drop_duplicates()
    )
    return nodes
    
    
type_nodes = create_node_df(largest, "Type")
url_nodes = create_node_df(largest, "host")
nodes = pd.concat([type_nodes, url_nodes])

g = (
    graphistry
    .edges(edges, "Type", "host")
    .nodes(nodes, "node")
)

g.plot()