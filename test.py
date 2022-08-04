import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# step 1: load A1
A1 = np.random.randint(0,2,(5,5))

# step 2: doing permutation to generate a new graph
I = np.identity(len(A1))
P = np.random.permutation(I)
print('P\n', P)
A2 = np.dot(P, A1)
A2 = np.dot(A2, np.transpose(P))
diag = np.diagonal(P)
diag_one_sum = np.sum(diag)
diag_ratio = (len(A1)-diag_one_sum) / len(A1)
print(f"generate new graph with change ratio {diag_ratio}\n")
print('A1\n', A1)
print('A2\n', A2)

# step 3: doing edge removal to add structural noise
pr = 0.2
pa = 0.05
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

# step 4: show the ground truth
PT = np.transpose(P)
ground_truth = list()
for idx1 in range(len(PT)):
    list = PT[idx1]
    loc = np.where(list==1)
    idx2 = loc[0][0]
    print(f'{idx1} {idx2}\n')
    ground_truth.append(f'{idx1} {idx2}')

# step 5: visualization
nx_graph = nx.from_numpy_array(A1)
nx_graph.edges(data=True)
nx_graph_p = nx.from_numpy_array(A2)
nx_graph_p.edges(data=True)

print('show A1\n', A1)
nx.draw(nx_graph, with_labels= True)
plt.show()

print('show A2\n', A2)
nx.draw(nx_graph_p, with_labels= True)
plt.show()

# step 6: give the anchor point
anchor_num = 1
np.random.shuffle(ground_truth)
anchor = ground_truth[0:anchor_num]
print(anchor)