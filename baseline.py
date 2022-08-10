from multiprocessing import reduction
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"current device is {device}")

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
    # add self-loop
    I = np.identity(len(A1))
    A1 = A1 + I
    A2 = A2 + I
    # load anchor point
    anchor = np.loadtxt('anchor.txt', delimiter=' ')

    return A1, A2, anchor

class Dataset(object):
    def __init__(self, anchor):
        self.anchor = torch.from_numpy(anchor)
        self.a1anchor = self.anchor[:,0].numpy()
        self.a2anchor = self.anchor[:,1].numpy()
        self.negative = np.random.randint(low=0, high=1034, size=(1034))
        self.negative.tolist()
    def __getitem__(self, index):
        return self.a1anchor[index], self.a2anchor[index], self.negative[index]
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
        # print(self.weight.grad)
        # print(self.weight.grad_fn)
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)

class Model(nn.Module):
    def __init__(self, A1, A2, embedding_dim):
        super(Model, self).__init__()
        self.l = 2
        self.embedding_dim = embedding_dim
        self.num_sr = A1.shape[0]
        self.num_tg = A2.shape[0]
        self.adj1 = A1
        self.adj2 = A2
        self.dropout = nn.Dropout(0.2)
        self.embedding1 = nn.Parameter(torch.FloatTensor(size=(self.num_sr, self.embedding_dim)), requires_grad=True) # [num_node, embedding_dim]
        self.embedding2 = nn.Parameter(torch.FloatTensor(size=(self.num_tg, self.embedding_dim)), requires_grad=True) # [num_node, embedding_dim]
        # self.embedding1 = nn.Embedding(self.num_sr, self.embedding_dim, _weight=torch.zeros((self.num_sr, self.embedding_dim), dtype=torch.float))
        # self.embedding2 = nn.Embedding(self.num_tg, self.embedding_dim, _weight=torch.zeros((self.num_tg, self.embedding_dim), dtype=torch.float))
        self.gcnblocks = nn.ModuleList([GraphConvLayer(in_features=self.embedding_dim, out_features=self.embedding_dim) for i in range(self.l)])
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.normal_(self.embedding1.data, mean=0, std=1./self.embedding_dim)
        nn.init.normal_(self.embedding2.data, mean=0, std=1./self.embedding_dim)
    
    def get_embeddings(self):
        return self.embedding1, self.embedding2

    def forward(self):
        a1_embeddings = self.embedding1
        # print('a1_embeddings', a1_embeddings)
        # print('1', self.embedding1.grad)
        # print('2', self.embedding1.grad_fn)
        a2_embeddings = self.embedding2
        for layer in self.gcnblocks:
            a1_embeddings = layer(a1_embeddings, self.adj1)
            a2_embeddings = layer(a2_embeddings, self.adj2)
        return a1_embeddings, a2_embeddings

def customized_loss(a1_embedding, a2_embedding, a1_align, a2_align, neg1_left, neg1_right, neg2_left, neg2_right, neg_samples_size, pos_margin=0.01, neg_margin=3, neg_param=0.2, only_pos=False):
    a1_embedding = F.normalize(a1_embedding, p=2, dim=1)
    a2_embedding = F.normalize(a2_embedding, p=2, dim=1)
    # process the ground truth
    a1_align = np.array(a1_align)
    a2_align = np.array(a2_align)
    t = len(a1_align)
    L = np.ones((t, neg_samples_size)) * (a1_align.reshape((t,1)))
    a1_align = L.reshape((t*neg_samples_size,))
    L = np.ones((t, neg_samples_size)) * (a2_align.reshape((t,1)))
    a2_align = L.reshape((t*neg_samples_size,))
    # convert to tensor
    a1_align = torch.tensor(a1_align)
    a2_align = torch.tensor(a2_align)
    neg1_left = torch.tensor(neg1_left)
    neg1_right = torch.tensor(neg1_right)
    neg2_left = torch.tensor(neg2_left)
    neg2_right = torch.tensor(neg2_right)
    # positive pair loss computation
    left_x = a1_embedding[a1_align.long()]
    right_x = a2_embedding[a2_align.long()]
    pos_loss = torch.abs(left_x - right_x)
    pos_loss = torch.sum(pos_loss, dim=1)
    '''
    pos_loss = F.relu(pos_loss - pos_margin) # Bootstrapping Entity Alignment with Knowledge Graph Embedding
    pos_loss = torch.sum(pos_loss)
    pos_loss = Variable(pos_loss, requires_grad=True)
    '''
    # negative pair loss computation on neg_1
    left_x = a1_embedding[neg1_left.long()]
    right_x = a2_embedding[neg1_right.long()]
    neg_loss_1 = torch.abs(left_x - right_x)
    neg_loss_1 = torch.sum(neg_loss_1, dim=1)
    '''
    neg_loss_1 = F.relu(neg_margin - neg_loss_1)
    neg_loss_1 = torch.sum(neg_loss_1)
    neg_loss_1 = Variable(neg_loss_1, requires_grad=True)
    neg_loss_1 = neg_param * neg_loss_1
    '''
    # negative pair loss computation on neg_2
    left_x = a1_embedding[neg2_left.long()]
    right_x = a2_embedding[neg2_right.long()]
    neg_loss_2 = torch.abs(left_x - right_x)
    neg_loss_2 = torch.sum(neg_loss_2, dim=1)
    '''
    neg_loss_2 = F.relu(neg_margin - neg_loss_2)
    neg_loss_2 = torch.sum(neg_loss_2)
    neg_loss_2 = Variable(neg_loss_2, requires_grad=True)
    neg_loss_2 = neg_param * neg_loss_2
    return pos_loss * neg_samples_size + neg_loss_1 + neg_loss_2
    '''
    loss1 = F.relu(pos_loss + neg_margin - neg_loss_1)
    loss2 = F.relu(pos_loss + neg_margin - neg_loss_2)
    loss1 = torch.sum(loss1)
    loss2 = torch.sum(loss2)
    return loss1+loss2

def margin_ranking_loss(criterion, E1, E2, neg_samples_size, a1_align, a2_align, neg1_left, neg1_right, neg2_left, neg2_right):
        a1_align = np.array(a1_align)
        a2_align = np.array(a2_align)
        t = len(a1_align)
        L = np.ones((t, neg_samples_size)) * (a1_align.reshape((t,1)))
        a1_align = L.reshape((t*neg_samples_size,))
        L = np.ones((t, neg_samples_size)) * (a2_align.reshape((t,1)))
        a2_align = L.reshape((t*neg_samples_size,))
        # convert to tensor
        a1_align = torch.as_tensor(a1_align, dtype=torch.float)
        a2_align = torch.as_tensor(a2_align, dtype=torch.float)
        neg1_left = torch.as_tensor(neg1_left, dtype=torch.float)
        neg1_right = torch.as_tensor(neg1_right, dtype=torch.float)
        neg2_left = torch.as_tensor(neg2_left, dtype=torch.float)
        neg2_right = torch.as_tensor(neg2_right, dtype=torch.float)
        # using index to get the embeddings
        sr_true = E1[a1_align.long()]
        sr_neg = E1[neg2_left.long()]
        tg_true = E2[a2_align.long()]
        tg_neg = E2[neg1_right.long()]
        # print(sr_true.shape, sr_neg.shape, tg_true.shape, tg_neg.shape)
        # computing loss
        loss = criterion(
            anchor = torch.cat([sr_true, tg_true], dim=0),
            positive = torch.cat([tg_true, sr_true], dim=0),
            negative = torch.cat([tg_neg, sr_neg], dim=0)
        )
        return loss

class RandingbasedLossFunc(nn.Module):
    def __init__(self, neg_samples_size, pos_margin=0.01, neg_margin=3, neg_param=0.2):
        super(RandingbasedLossFunc, self).__init__()
        self.neg_samples_size = neg_samples_size
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.neg_param = neg_param
    def forward(self, a1_embedding, a2_embedding, a1_align, a2_align, neg1_left, neg1_right, neg2_left, neg2_right):
        # process the ground truth
        a1_align = np.array(a1_align)
        a2_align = np.array(a2_align)
        t = len(a1_align)
        L = np.ones((t, self.neg_samples_size)) * (a1_align.reshape((t,1)))
        a1_align = L.reshape((t*self.neg_samples_size,))
        L = np.ones((t, self.neg_samples_size)) * (a2_align.reshape((t,1)))
        a2_align = L.reshape((t*self.neg_samples_size,))
        # convert to tensor
        a1_align = torch.tensor(a1_align)
        a2_align = torch.tensor(a2_align)
        neg1_left = torch.tensor(neg1_left)
        neg1_right = torch.tensor(neg1_right)
        neg2_left = torch.tensor(neg2_left)
        neg2_right = torch.tensor(neg2_right)
        # positive pair loss computation
        left_x = a1_embedding[a1_align.long()]
        right_x = a2_embedding[a2_align.long()]
        pos_loss = torch.abs(left_x - right_x)
        pos_loss = torch.sum(pos_loss, dim=1)
        '''
        pos_loss = F.relu(pos_loss - pos_margin) # Bootstrapping Entity Alignment with Knowledge Graph Embedding
        pos_loss = torch.sum(pos_loss)
        pos_loss = Variable(pos_loss, requires_grad=True)
        '''
        # negative pair loss computation on neg_1
        left_x = a1_embedding[neg1_left.long()]
        right_x = a2_embedding[neg1_right.long()]
        neg_loss_1 = torch.abs(left_x - right_x)
        neg_loss_1 = torch.sum(neg_loss_1, dim=1)
        '''
        neg_loss_1 = F.relu(neg_margin - neg_loss_1)
        neg_loss_1 = torch.sum(neg_loss_1)
        neg_loss_1 = Variable(neg_loss_1, requires_grad=True)
        neg_loss_1 = neg_param * neg_loss_1
        '''
        # negative pair loss computation on neg_2
        left_x = a1_embedding[neg2_left.long()]
        right_x = a2_embedding[neg2_right.long()]
        neg_loss_2 = torch.abs(left_x - right_x)
        neg_loss_2 = torch.sum(neg_loss_2, dim=1)
        '''
        neg_loss_2 = F.relu(neg_margin - neg_loss_2)
        neg_loss_2 = torch.sum(neg_loss_2)
        neg_loss_2 = Variable(neg_loss_2, requires_grad=True)
        neg_loss_2 = neg_param * neg_loss_2
        return pos_loss * neg_samples_size + neg_loss_1 + neg_loss_2
        '''
        loss = F.relu(pos_loss + self.neg_margin - neg_loss_1 - neg_loss_2)
        loss = torch.sum(loss)
        #loss = Variable(loss, requires_grad=True)
        return loss

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

############################
# building block
epoch = 100
embedding_dim = 500
learning_rate = 0.01
weight_decay = 1e-5
neg_samples_size = 10
A1, A2, anchor = load_data()
batchsize = len(anchor[:,0])
dataset = Dataset(anchor)
train_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=False)
model = Model(Variable(torch.from_numpy(A1).float()), Variable(torch.from_numpy(A2).float()), embedding_dim=embedding_dim)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('#params:',pytorch_total_params)
criterion = RandingbasedLossFunc(neg_samples_size=neg_samples_size)
criterion = nn.TripletMarginLoss(margin=3, p=2)
############################

neg1_left, neg1_right, neg2_left, neg2_right = generate_neg_sample(anchor, neg_samples_size=neg_samples_size)

best_E1 = None
best_E2 = None
for e in range(epoch):
    model.train()
    if e % 10 == 0:
        neg1_left, neg1_right, neg2_left, neg2_right = generate_neg_sample(anchor, neg_samples_size=neg_samples_size)
    for _, data in enumerate(train_loader):
        a1_align, a2_align, a2_neg_align = data
        E1, E2 = model()
        
        optimizer.zero_grad()
        loss = customized_loss(E1, E2, a1_align, a2_align, neg1_left, neg1_right, neg2_left, neg2_right, neg_samples_size=neg_samples_size, neg_param=0.3)
        # loss = margin_ranking_loss(criterion, E1, E2, a1_align, a2_align, neg1_left, neg1_right, neg2_left, neg2_right)
        loss.backward()
        # print([x.grad for x in optimizer.param_groups[0]['params']])
        optimizer.step()
    print(f"epoch: {e+1}, loss: {loss}\n")

# test and evaluate 
model.eval()
E1, E2 = model()
similarity_matrix, alignment_hit1, alignment_hit5 = evaluate(Embedding1=E1, Embedding2=E2, hitmax=20)
print(similarity_matrix)

ground_truth = np.loadtxt('ground_truth.txt', delimiter=' ')
hit_1 = 0
hit_5 = 0
for idx in range(len(ground_truth)):
    gt = ground_truth[idx][1]
    if int(gt) == alignment_hit1[idx]:
        hit_1 += 1
    if int(gt) in alignment_hit5[idx]:
        hit_5 += 1

print(f"final score: hit@1: total {hit_1} and ratio {hit_1/len(ground_truth)}, hit@5: total {hit_5} and ratio {hit_5/len(ground_truth)}")