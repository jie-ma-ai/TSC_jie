import torch
# import numpy as np
from scipy.stats import skew

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def stats(t):
    out = torch.zeros((t.shape[0], t.shape[1], 4))
    # print(t.shape)
    for j in range(t.shape[0]):
        print(j)
        for i in range(t.shape[1]):
            out[j, i, 0] = torch.max(t[j, i, :])
            out[j, i, 1] = torch.mean(t[j, i, :])
            out[j, i, 2] = torch.var(t[j, i, :])
            out[j, i, 3] = skew(t[j, i, :])
            # out[j, i, 4] = summit_count(t[j, i, :])
    return out.view((t.shape[0], t.shape[1]*4))

# def pca(t):
#     for i in range(t.shape[0]):
#         print(torch.pca_lowrank(t[i]))


def summit_count(arr):
    count = 0
    for i in range(1, arr.shape[0] - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            count += 1
    return count


# t = torch.tensor([[[1.0, 2, 3, 4, 3, 4, 5], [2, 4, 5, 6, 1, 5, 4]]])
# print(pca(t))