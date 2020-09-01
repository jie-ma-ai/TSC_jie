import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch

from model.model import mv_embedding
from utils.data_clean_up import data_loader, one_hot_reverse
from utils.kernels import generate_kernels, apply_kernels
import os


# train the model
def train_model(train_loader, test_loader, model, learning_rate, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), learning_rate, weight_decay=0.01)
    running_loss = 0.0
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = apply_kernels(inputs, kernels).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            labels = labels.type(torch.float32)
            labels = labels.to(device)
            # print(inputs.dtype,outputs.dtype,labels.dtype)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1 == 0:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
        if epoch % 1 == 0:
            test_model(test_loader, model)
    print('Finished Training')


# test the model
def test_model(test_loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = apply_kernels(images, kernels).to(device)
            # print(labels[1:10])
            labels = one_hot_reverse(labels)
            labels = labels.type(torch.int64)
            labels = labels.add(-torch.min(labels)).to(device)
            # print(labels[1:10])
            # imgaes=images.to('cuda')
            outputs = model(images)
            # print(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            total += labels.size(0)
            # print(predicted.dtype,labels)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %d test cases: %d %%' % (total, 100*correct / total))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    input_path = r'./data/Univariate_arff'
    dataset_name = 'Car'
    batch_size = 20
    kernels_num = 300
    kernels_length = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, input_size, output_size = data_loader(input_path, dataset_name, batch_size)

    kernels = generate_kernels(kernels_num, kernels_length)
    model = mv_embedding(input_size, kernels_shape=kernels.shape, output_size=output_size)
    model = model.to(device)

    train_model(train_loader, test_loader, model, learning_rate=0.0001, epochs=10)

    test_model(test_loader, model)
