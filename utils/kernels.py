import torch
import numpy as np


def generate_kernels(num_kernels=1000, kernel_length=7):
    weights = torch.randn((num_kernels, kernel_length))
    weights = weights.type(torch.float64)
    for i in range(num_kernels):
        weights[i, :] = torch.div(weights[i, :], torch.sum(torch.abs(weights[i, :])))
    return weights


def apply_kernels(time_series, kernels):
    kernels_output = torch.zeros([time_series.shape[0], kernels.shape[0], time_series.shape[1]-kernels.shape[1]+1])
    for j in range(time_series.shape[0]):
        for i in range(kernels.shape[0]):
            kernels_output[j, i, ] = torch.from_numpy(np.convolve(time_series[j, ], kernels[i, ], mode='valid'))
    return kernels_output


# kernels = generate_kernels(7, 2)
# print(apply_kernels(torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]), kernels))
