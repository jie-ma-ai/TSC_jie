import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def generate_kernels(num_kernels=1000, kernel_length=7):
    weights = np.random.normal(1, 1, (num_kernels, kernel_length))
    for i in range(num_kernels):
        weights[i, :] = weights[i, :]/sum(abs(weights[i, :]))

    return weights

# print(generate_kernels(7,2))


def apply_kernels(time_series, kernels):
    kernels_output = torch.zeros([time_series.shape[0],kernels.shape[0],time_series.shape[1]-kernels.shape[1]+1])
    for j in range(time_series.shape[0]):
        for i in range(kernels.shape[0]):
            kernels_output[j, i,] = torch.from_numpy(np.convolve(time_series[j,], kernels[i,], mode='valid'))
    return kernels_output

def one_hot(labels, shift, max, batch_size):
    target=labels.add(shift)
    output = torch.zeros(batch_size, max+1)
    for i in range(batch_size):
        output[i,int(target[i].item())] = 1
    return output
# # print(kernels)
# time_series=np.array([1,2,3])
# print(apply_kernels(time_series,kernels))

# Model
# class sub_model(nn.Module):
#     def __init__(self):
#         super(sub_model, self).__init__()
#         self.embedding = nn.Linear(input_size-kernels.shape[1]+1, 3)
#     def forward(self,x):
#         x=self.embedding(x)
#         return x


class mv_embedding(nn.Module):
    def __init__(self):
        super(mv_embedding, self).__init__()
        self.embedding = nn.Linear(
            input_size-kernels.shape[1]+1, 3, bias=False)
        self.fc2 = nn.Linear(3*kernels.shape[0], output_size)

    def forward(self, x):
        # x = apply_kernels(x, kernels)
        y = torch.zeros([x.shape[0],kernels.shape[0],3])        
        # x = self.embedding(x[0, ])
        # print(x.shape[0])
        for j in range(x.shape[0]):
            for i in range(kernels.shape[0]):
                # print(self.embedding(x[j,i, ]))
                y[j,i,] = self.embedding(x[j,i, ])
        y = y.view([x.shape[0],3*kernels.shape[0]])
        y = self.fc2(y)
        # print(y.shape)
        return y


if __name__ == '__main__':
    input_path = r'./data/Univariate_arff'
    dataset_name = 'Car'
    batch_size = 10

    gpu = 0
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

    training_data = np.loadtxt(
        f"{input_path}/{dataset_name}/{dataset_name}_TRAIN.txt")
    training_data = torch.tensor(training_data)
    Y_training, X_training = training_data[:, 0], training_data[:, 1:]

    training_set = TensorDataset(X_training, Y_training)

    trainloader = DataLoader(
        training_set, batch_size, shuffle=True, num_workers=2)

    test_data = np.loadtxt(
        f"{input_path}/{dataset_name}/{dataset_name}_TEST.txt")

    test_data = torch.tensor(test_data)
    Y_test, X_test = test_data[:, 0], test_data[:, 1:]

    test_set = TensorDataset(X_test, Y_test)
    testloader = DataLoader(
        test_set, batch_size, shuffle=False, num_workers=2)

    kernels = generate_kernels(100, 5)
    input_size = X_training.shape[1]
    output_size = torch.unique(Y_training).shape[0]

    # print(input_size, output_size)

    model = mv_embedding()
    model.to('cuda')

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001,weight_decay=0.01)

    for epoch in range(50):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # print(data)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # labels.to('cuda')
            inputs = apply_kernels(inputs, kernels).to('cuda')
            # print(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            labels = one_hot(labels, -1, 3, batch_size)
            labels.to('cuda')
            
            # print(labels)
            # labels=labels.cuda()
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)
            # l2=0
            # for p in model.parameters():
            #     l2 += p.pow(2).sum() * 0.01
            # loss+=l2
            # print(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss))
                running_loss = 0.0

    print('Finished Training')
    # print(model.parameters())

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = apply_kernels(images, kernels)
            # print(labels[1:10])
            labels = labels.add(-torch.min(labels))
            # print(labels[1:10])
            # imgaes=images.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))