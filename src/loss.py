import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
