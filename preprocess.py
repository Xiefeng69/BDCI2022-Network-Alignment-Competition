import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
inputs = 'data_G1.txt'

# step 1: loading exist graph
nx_graph = nx.read_edgelist(inputs, nodetype = int, comments="%")
print("read in original graph\n")
adj = nx.adjacency_matrix(nx_graph, nodelist = range(nx_graph.number_of_nodes()) )
A1=np.array(nx.adjacency_matrix(nx_graph).todense())

# step 2: doing permutation to generate a new graph
I = np.identity(len(A1))
P = np.random.permutation(I)
A2 = np.dot(P, A1)
A2 = np.dot(A2, np.transpose(P))
diag = np.diagonal(P)
diag_one_sum = np.sum(diag)
diag_ratio = (len(A1)-diag_one_sum) / len(A1)
print(f"generate new graph with change ratio {diag_ratio}\n")

# step 3: doing edge removal to add structural noise
pr = 0.05
pa = 0.001
remove_num = 0
add_num = 0
for i in range(len(A2)):
    for j in range(len(A2[0])):
        randN = np.random.rand() #主要用于返回一个或一组0到1之间的随机数或随机数组。
        if A2[i][j] == 1:
            if randN >= 0 and randN < pr:
                A2[i][j] = 0
                remove_num += 1
        else:
            if randN >= 0 and randN < pa:
                A2[i][j] = 1
                add_num += 1

print(f"remove edge numer: {remove_num}\n")
print(f"add edge numer: {add_num}\n")

# step 4: save new graph
nx_graph_p = nx.from_numpy_array(A2)
nx_graph_p.edges(data=True)
nx.write_edgelist(nx_graph_p, 'data_G2.txt', delimiter=' ')

# step 5: save ground truth
PT = np.transpose(P)
ground_truth = list()
with open('ground_truth.txt', 'w', encoding='utf-8') as f:
    for idx1 in range(len(PT)):
        list = PT[idx1]
        loc = np.where(list==1)
        idx2 = loc[0][0]
        f.write(f"{idx1} {idx2}\n")
        ground_truth.append(f'{idx1} {idx2}')

# step 6: give the anchor point
anchor_ratio = 0.3
anchor_num = math.ceil(len(ground_truth)*anchor_ratio)
np.random.shuffle(ground_truth)
anchor = ground_truth[0:anchor_num]
print(anchor)
with open('anchor.txt', 'w', encoding='utf-8') as f:
    for i in anchor:
        f.write(f"{i}\n")


# step 7: visualization
# print('show A1\n', A1)
# nx.draw(nx_graph)
# plt.show()

# print('show A2\n', A2)
# nx.draw(nx_graph_p)
# plt.show()