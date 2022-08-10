import math
import torch
import torch.nn as nn
import numpy as np

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