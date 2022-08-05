import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
import networkx as nx
from torch.utils.data import DataLoader
from torch.autograd import Variable
np.set_printoptions(suppress=True)

graph1 = 'data_G1.txt'
graph2 = 'data_G2.txt'
def load_data():
    # load graph adjacent matrix
    A1 = nx.read_edgelist(graph1, nodetype = int, comments="%")
    adj1 = nx.adjacency_matrix(A1, nodelist = range(A1.number_of_nodes()) )
    A1=np.array(nx.adjacency_matrix(A1).todense())
    A2 = nx.read_edgelist(graph2, nodetype = int, comments="%")
    adj2 = nx.adjacency_matrix(A2, nodelist = range(A2.number_of_nodes()) )
    A2=np.array(nx.adjacency_matrix(A2).todense())

    # load anchor point
    anchor = np.loadtxt('anchor.txt', delimiter=' ')

    return A1, A2, anchor

class Dataset(object):
    def __init__(self, anchor):
        self.anchor = torch.from_numpy(anchor)
        self.a1anchor = self.anchor[:,0].numpy()
        self.a2anchor = self.anchor[:,1].numpy()
    def __getitem__(self, index):
        return self.a1anchor[index], self.a2anchor[index]
    def __len__(self):
        return len(self.a1anchor)

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ReLU()

        nn.init.xavier_uniform_(self.weight)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)

class Model(nn.Module):
    def __init__(self, A1, A2, embedding_dim):
        super().__init__()
        self.l = 2
        self.embedding_dim = embedding_dim
        self.num_sr = A1.shape[0]
        self.num_tg = A2.shape[0]
        self.adj1 = A1
        self.adj2 = A2
        self.dropout = nn.Dropout(0.5)
        self.embedding_sr = nn.Parameter(torch.FloatTensor(self.num_sr, self.embedding_dim), requires_grad=True)
        self.embedding_tg = nn.Parameter(torch.FloatTensor(self.num_tg, self.embedding_dim), requires_grad=True)
        # self.embedding_sr = nn.Embedding(self.num_sr, self.embedding_dim, _weight=torch.zeros((self.num_sr, self.embedding_dim), dtype=torch.float))
        # self.embedding_tg = nn.Embedding(self.num_tg, self.embedding_dim, _weight=torch.zeros((self.num_tg, self.embedding_dim), dtype=torch.float))
        self.gcnblocks = nn.ModuleList([GraphConvLayer(in_features=self.embedding_dim, out_features=self.embedding_dim) for i in range(self.l)])
    #     self.init_parameters()
    
    # def init_parameters(self):
    #     nn.init.normal_(self.embedding_sr.weight.data, std=1. / math.sqrt(self.num_sr))
    #     nn.init.normal_(self.embedding_tg.weight.data, std=1. / math.sqrt(self.num_tg))

    def normalize(self):
        self.embedding_sr.weight.data = F.normalize(self.embedding_sr.weight, dim=-1, p=2)
        self.embedding_tg.weight.data = F.normalize(self.embedding_tg.weight, dim=-1, p=2)
    
    def forward(self):
        a1_embeddings = self.embedding_sr
        a2_embeddings = self.embedding_tg
        for layer in self.gcnblocks:
            a1_embeddings = layer(a1_embeddings, self.adj1)
            a1_embeddings = self.dropout(a1_embeddings)
            a2_embeddings = layer(a2_embeddings, self.adj2)
            a2_embeddings = self.dropout(a2_embeddings)
        return a1_embeddings, a2_embeddings

def evaluate(Embedding1, Embedding2, sim_measure="euclidean"):
    if sim_measure == "cosine":
        similarity_matrix = cosine_similarity(Embedding1, Embedding2)
    else:
        similarity_matrix = euclidean_distances(Embedding1, Embedding2)
        similarity_matrix = np.exp(-similarity_matrix)
    alignment_hit1 = list()
    alignment_hit5 = list()
    for line in similarity_matrix:
        idx = np.argmax(line)
        alignment_hit1.append(idx)
        idxs = heapq.nlargest(10, range(len(line)), line.take)
        alignment_hit5.append(idxs)
    return alignment_hit1, alignment_hit5

##############
# building block
epoch = 30
embedding_dim = 1000
batchsize = 16
learning_rate = 0.0001
weight_decay = 1e-5
A1, A2, anchor = load_data()
dataset = Dataset(anchor)
train_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True)
model = Model(Variable(torch.from_numpy(A1).float()), Variable(torch.from_numpy(A2).float()), embedding_dim=embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.L1Loss()
##############

for e in range(epoch):
    for i, data in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        a1_align, a2_align = data
        E1, E2 = model()
        a1_embedding = E1[a1_align.long()]
        a2_embedding = E1[a2_align.long()]
        loss = criterion(a1_embedding, a2_embedding)
        loss.backward()
        optimizer.step()
    print(f"epoch: {e+1}, loss: {loss}\n")

# test
model.eval()
E1, E2 = model()
E1 = E1.detach().numpy()
E2 = E2.detach().numpy()
alignment_hit1, alignment_hit5 = evaluate(Embedding1=E1, Embedding2=E2)

ground_truth = np.loadtxt('ground_truth.txt', delimiter=' ')
hit_1 = 0
hit_5 = 0
for idx in range(len(ground_truth)):
    gt = ground_truth[idx][1]
    if int(gt) == int(alignment_hit1[i]):
        hit_1 += 1
    if int(gt) in alignment_hit5[i]:
        hit_5 += 1
print(f"final score: hit@1: {hit_1/len(ground_truth)}, hit@5: {hit_5/len(ground_truth)}")