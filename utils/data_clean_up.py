import numpy as np
import torch


def data_loader(input_path, dataset_name):
    train_dataset = np.loadtxt(f'{input_path}/{dataset_name}/{dataset_name}_TRAIN.txt')
    test_dataset = np.loadtxt(f'{input_path}/{dataset_name}/{dataset_name}_TEST.txt')
    # train dataset clean up
    x_train, y_train = train_dataset[:, 1:], train_dataset[:, 0]
    x_train = fill_nan_0_min(x_train)
    # input_size = x_train.shape[1]

    # test dataset clean up
    x_test, y_test = test_dataset[:, 1:], test_dataset[:, 0]
    x_test = fill_nan_0_min(x_test)

    # category y
    y = np.concatenate((y_train, y_test))
    y = one_hot(y)
    y_train = y[: y_train.shape[0], :]
    y_test = y[-y_test.shape[0]:, :]
    # output_size = y_train.shape[1]

    # numpy to tensor to torch dataloarder
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    # training_set = TensorDataset(x_train, y_train)
    # train_loader = DataLoader(training_set, batch_size, shuffle=True)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    # test_set = TensorDataset(x_test, y_test)
    # test_loader = DataLoader(test_set, batch_size, shuffle=False)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def fill_nan_0_min(arr):
    mask = np.isnan(arr)
    print('#NA: ', len(mask[mask == True]))
    # arr = np.where(~mask, arr, [0])
    # arr = np.where(~mask, arr, min(arr.min(), [0]))
    arr[mask] = 0
    arr[mask] = min(arr.min(), 0)
    return arr


def one_hot(arr):
    uni = np.unique(arr)
    out = np.zeros((arr.shape[0], uni.shape[0]))
    for i in range(uni.shape[0]):
        out[arr == uni[i], i] = 1
    return out


def one_hot_reverse(arr):
    out = torch.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        out[i] = torch.argmax(arr[i, :])
    return out
# a = np.array([[np.nan, -1, 2, np.nan, 3, np.nan], [np.nan, -1, 2, np.nan, 3, np.nan]])
# print(fill_nan_0_min(a))

# b = np.array([0,1, 2, 2,4, 4])
# print(one_hot(b))


# print(data_loader(r'd:/project_git/TSC_jie/data/Univariate_arff', 'Car', 10))
