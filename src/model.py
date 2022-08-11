import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import GraphConvLayer

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
    
    @property
    def embedding1(self):
        return self.embedding1
    
    @property
    def embedding2(self):
        return self.embedding2

    def forward(self):
        a1_embeddings = self.embedding1
        a2_embeddings = self.embedding2
        for layer in self.gcnblocks:
            a1_embeddings = layer(a1_embeddings, self.adj1)
            a2_embeddings = layer(a2_embeddings, self.adj2)
        a1_embeddings = F.normalize(a1_embeddings, dim=-1, p=2)
        a2_embeddings = F.normalize(a2_embeddings, dim=-1, p=2)
        return a1_embeddings, a2_embeddings