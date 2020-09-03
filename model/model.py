# import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import init
# import pandas as pd
# import numpy as np
import torch
# from torch.utils.data import DataLoader
# from torch.utils.data import TensorDataset


class mv_embedding(nn.Module):
    def __init__(self, input_size, kernels_shape, output_size):
        super(mv_embedding, self).__init__()
        self.kernels_shape = kernels_shape
        self.embedding = nn.Linear(input_size, 3, bias=True)
        self.fc2 = nn.Linear(3*kernels_shape[0], output_size, bias=False)
        self.bnor = nn.BatchNorm1d(3)

    def forward(self, x):
        # x = apply_kernels(x, kernels)
        y = torch.zeros([x.shape[0], self.kernels_shape[0], 3]).to('cuda')
        # x = self.embedding(x[0, ])
        # print(x.shape[0])
        for j in range(x.shape[0]):
            for i in range(self.kernels_shape[0]):
                # print(self.embedding(x[j,i, ]))
                y[j, i, ] = self.embedding(x[j, i, ])
        # # layer normalization for each embedding? batch normalization relates to auto grad parameters
        # for i in range(y.shape[2]):
        #     y[:, :, i] = (y[:, :, i]-y[:, :, i].mean())/y[:, :, i].std()
        y = torch.transpose(y, 1, 2)
        y = self.bnor(y)
        y = y.view([x.shape[0], 3*self.kernels_shape[0]])
        # y = torch.sigmoid(y)
        y = self.fc2(y)
        # print(y.shape)
        return y


class stats_embedding(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc0 = nn.Linear(input_size, 30)
        self.fc = nn.Linear(30, output_size)

    def forward(self, x):
        x = self.fc0(x)
        x = F.sigmoid(x)
        x = self.fc(x)
        return x
