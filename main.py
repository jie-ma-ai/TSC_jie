import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from model.model import mv_embedding, stats_embedding
from utils.data_clean_up import data_loader, one_hot_reverse
from utils.kernels import generate_kernels, apply_kernels
from utils.embedding import stats
import os


# train the model
def train_model(train_loader, test_loader, model, learning_rate, epochs):
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), learning_rate, weight_decay=0.01)
    running_loss = 0.0
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # inputs = apply_kernels(inputs, kernels)
            # # print(inputs.shape)
            # inputs = stats(inputs)
            # inputs = inputs.to(device)
            # print(inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            # labels = labels.type(torch.float32)
            labels = one_hot_reverse(labels)
            labels = labels.type(torch.int64)
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
            # images = apply_kernels(images, kernels)
            # images = stats(images).to(device)
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
    kernels_num = 500
    kernels_length = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data loading
    kernels = generate_kernels(kernels_num, kernels_length)
    x_train, y_train, x_test, y_test = data_loader(input_path, dataset_name, batch_size)
    x_train = apply_kernels(x_train, kernels)
    # print(x_train.shape)
    # x_train = stats(x_train)
    x_train = x_train.to(device)
    x_test = apply_kernels(x_test, kernels)
    # x_test = stats(x_test)
    x_test = x_test.to(device)
    input_size = x_train.shape[2]
    output_size = y_train.shape[1]

    training_set = TensorDataset(x_train, y_train)
    train_loader = DataLoader(training_set, batch_size, shuffle=True)

    test_set = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    # model learning
    model = mv_embedding(input_size, kernels_shape=kernels.shape, output_size=output_size)
    # model = stats_embedding(kernels_num*5, output_size)
    model = model.to(device)

    train_model(train_loader, test_loader, model, learning_rate=0.001, epochs=50)

    test_model(test_loader, model)

    #

    # from sklearn.linear_model import RidgeClassifierCV
    # # -- training ----------------------------------------------------------
    # y_train = one_hot_reverse(y_train)
    # y_test = one_hot_reverse(y_test)
    # classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    # classifier.fit(x_train, y_train)

    # # -- test --------------------------------------------------------------

    # print(classifier.score(x_test, y_test))
