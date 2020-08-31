import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset



class mv_embedding(nn.Module):
    def __init__(self, input_size, kernels_shape,output_size):
        super(mv_embedding, self).__init__()
        self.kernels_shape=kernels_shape
        self.embedding = nn.Linear(
            input_size-kernels_shape[1]+1, 3, bias=False)
        self.fc2 = nn.Linear(3*kernels_shape[0], output_size)

    def forward(self, x):
        # x = apply_kernels(x, kernels)
        y = torch.zeros([x.shape[0],self.kernels_shape[0],3]).to('cuda')        
        # x = self.embedding(x[0, ])
        # print(x.shape[0])
        for j in range(x.shape[0]):
            for i in range(self.kernels_shape[0]):
                # print(self.embedding(x[j,i, ]))
                y[j,i,] = self.embedding(x[j,i, ])
        y = y.view([x.shape[0],3*self.kernels_shape[0]])
        y = self.fc2(y)
        # print(y.shape)
        return y


