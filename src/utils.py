import math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

def evaluate(Embedding1, Embedding2, hitmax=10, sim_measure="cosine"):
    # Embedding1 = F.normalize(Embedding1, p=2, dim=1)
    # Embedding2 = F.normalize(Embedding2, p=2, dim=1)
    Embedding1 = Embedding1.detach().numpy()
    Embedding2 = Embedding2.detach().numpy()
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
        idxs = heapq.nlargest(hitmax, range(len(line)), line.take)
        alignment_hit5.append(idxs)
    return similarity_matrix, alignment_hit1, alignment_hit5

def generate_neg_sample(train_data, neg_samples_size):
    # broadcast ground truth
    t = len(train_data)
    L = np.ones((t, neg_samples_size)) * (train_data[:,0].reshape((t,1)))
    neg1_left = L.reshape((t*neg_samples_size,))
    L = np.ones((t, neg_samples_size)) * (train_data[:,1].reshape((t,1)))
    neg2_right = L.reshape((t*neg_samples_size,))

    # generate random neg-counterparts: can also use np.random.choice
    neg1_right = np.random.randint(low=0, high=1034, size=(t*neg_samples_size))
    neg2_left = np.random.randint(low=0, high=1034, size=(t*neg_samples_size))

    return neg1_left, neg1_right, neg2_left, neg2_right

def load_data(graph1, graph2):
    # load graph adjacent matrix
    A1 = nx.read_edgelist(graph1, nodetype = int, comments="%")
    adj1 = nx.adjacency_matrix(A1, nodelist = range(A1.number_of_nodes()) )
    A1=np.array(nx.adjacency_matrix(A1).todense())
    A2 = nx.read_edgelist(graph2, nodetype = int, comments="%")
    adj2 = nx.adjacency_matrix(A2, nodelist = range(A2.number_of_nodes()) )
    A2=np.array(nx.adjacency_matrix(A2).todense())
    # add self-loop
    I = np.identity(len(A1))
    A1 = A1 + I
    A2 = A2 + I
    # load anchor point
    anchor = np.loadtxt('anchor.txt', delimiter=' ')

    return A1, A2, anchor