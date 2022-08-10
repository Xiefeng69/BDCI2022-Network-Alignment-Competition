from multiprocessing import reduction
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
np.set_printoptions(suppress=True)

from src.loss import customized_loss, margin_ranking_loss
from src.dataset import Dataset
from src.layers import GraphConvLayer
from src.utils import evaluate, generate_neg_sample, load_data
from src.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"current device is {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument('--hidden', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--negsize', type=int, default=10)
parser.add_argument('--negiter', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--graph', type=str, default="data_G")
args = parser.parse_args()

############################
# parameters
epoch = args.epoch
embedding_dim = args.hidden
learning_rate = args.lr
weight_decay = args.weight_decay
neg_samples_size = args.negsize
negiter = args.negiter
graph_path = args.graph
############################

############################
# preprocess
graph1 = f'{graph_path}1.txt'
graph2 = f'{graph_path}2.txt'
A1, A2, anchor = load_data(graph1=graph1, graph2=graph2)
batchsize = len(anchor[:,0])
dataset = Dataset(anchor)
train_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=False)
model = Model(Variable(torch.from_numpy(A1).float()), Variable(torch.from_numpy(A2).float()), embedding_dim=embedding_dim)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.TripletMarginLoss(margin=3, p=2)
############################

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'#total params: {pytorch_total_params}')

# begin training
best_E1 = None
best_E2 = None
neg1_left, neg1_right, neg2_left, neg2_right = generate_neg_sample(anchor, neg_samples_size=neg_samples_size)
for e in range(epoch):
    model.train()
    if e % negiter == 0:
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