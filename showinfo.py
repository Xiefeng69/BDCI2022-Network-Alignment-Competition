import math
from turtle import left
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
inputs = 'data_G1.txt'

# step 1: loading exist graph
nx_graph = nx.read_edgelist(inputs, nodetype = int, comments="%")
print("read in original graph\n")
print(f"edge number: {nx_graph.size()}\n")

d = dict(nx.degree(nx_graph))
x = list(range(max(d.values())+1))
y = [i/sum(nx.degree_histogram(nx_graph)) for i in nx.degree_histogram(nx_graph)]
plt.bar(x, y, width=0.5, color="blue")
plt.xlabel("$degree$")
plt.ylabel("$ratio$")
plt.xlim(left=1)
plt.show()