
'''
The CNN case for the homework
1. try the NN framework with 2-layers + cross entropy loss function
2. Update the weights based on backward functions -- pending

'''

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
from torchsummary import summary

def get_data():

    plt.rcParams['figure.figsize'] = (7, 5)

    # Fetch the dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    # Class indices of planes and dogs
    plane, dog = 0, 5
    max_size = 2500  # Maximum number of data points

    # Prepare the training set
    labels_train = np.array(trainset.targets)
    selector_train = np.logical_or(labels_train == plane, labels_train == dog)
    X_train = torch.tensor(trainset.data[selector_train][:max_size]).float()
    y_train = torch.tensor((labels_train[selector_train] == plane).astype(dtype = np.float32)[:max_size])

    # Prepare the test set
    labels_test = np.array(testset.targets)
    selector_test = np.logical_or(labels_test == plane, labels_test == dog)
    X_test = torch.tensor(testset.data[selector_test]).float()
    y_test = torch.tensor((labels_test[selector_test] == plane).astype(dtype = np.float32))

    # Get the number of data points and the number of dimensions
    shape = X_train.shape
    n = shape[0]  # Number of data points
    d = shape[1] * shape[2] * shape[3]  # Number of dimensions

    # Flatten the image data
    X_train_1D = torch.reshape(X_train, (-1, d))
    X_test_1D = torch.reshape(X_test, (-1, d))

    # If you have CUDA, copy these tensors to GPU
    cuda = True  # If False CUDA will be disabled
    if cuda and torch.cuda.is_available():
        print("Cuda available")
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()
        X_train_1D = X_train_1D.cuda()
        X_test_1D = X_test_1D.cuda()

    # Short variable names for brevity
    x, x_ = X_train_1D.t(), X_test_1D.t()
    y, y_ = y_train, y_test

    x_train = torch.transpose(x, 0, 1)
    x_test = torch.transpose(x_, 0, 1)
    y_train = y
    y_test = y_

    return X_train, X_test, y_train, y_test

'''
Define the neural network results -- pending and im cases
'''

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()
        # Create the layers
        self.conv = nn.Conv2d(3, 16, 9)
        self.pool = nn.MaxPool2d(3)
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):

        # Forward pass
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(-1, 1024)
        x = torch.sigmoid(self.fc(x))
        return x

if __name__ == "__main__":

    cnn = CNN()  # Construct the network
    cuda = True
    if cuda and torch.cuda.is_available():
        cnn = cnn.cuda()
    print("check the number of paraemeters ", summary(cnn, input_size = (3, 32, 32)))
    input("check")

    cuda = True
    if cuda and torch.cuda.is_available():
        cnn = cnn.cuda()

    weight_scale = 1e-6             # Maximum initial weight
    alpha = 2e-6                    # Learning rate
    T = 2000                        # Number of epochs
    log = True                      # If False the prints will be disabled
    dt = 100                        # Log the loss in every dt epochs
    loss_train = np.empty(T // dt)  # Training loss
    loss_test = np.empty(T // dt)   # Test loss
    accuracy = np.empty(T // dt)    # Test accuracy

    # Short variable names for brevity
    X_train, X_test, y_train, y_test = get_data()
    x, x_ = X_train.reshape(-1, 3, 32, 32), X_test.reshape(-1, 3, 32, 32)
    y, y_ = y_train, y_test

    # The loss function and the optimization method
    criterion = nn.BCELoss()
    # optimizer = optim.SGD(cnn.parameters(), lr = alpha, momentum = 0.9)
    optimizer = optim.Adam(cnn.parameters(), lr = alpha, weight_decay = 0.7)

    # Start training
    start = time.time()
    for t in range(T):

        optimizer.zero_grad()

        # forward + backward
        z = cnn(x)[:, 0]
        loss = criterion(z, y)
        loss.backward()

        # Print the training loss, the test loss and the test accuracy
        if (t + 1) % dt == 0:

            # Store the training loss
            loss_train[t // dt] = loss

            # Predict the classes of the images in the test set
            z = cnn(x_)[:, 0]
            loss_test[t // dt] = F.binary_cross_entropy(z, y_)
            accuracy[t // dt] = 1 - torch.mean(torch.abs((z > 0.5).float() - y_))

            if log:
                print('Epoch:', t + 1)
                print('Training Loss:', loss_train[t // dt])
                print('Test Loss:', loss_test[t // dt])
                print('Accuracy:', accuracy[t // dt])

        # Update the weights
        optimizer.step()

    end = time.time()

    # Print the statistics
    print('Training time:', end - start)
    print('Minimum Training Loss:', np.min(loss_train))
    print('Minimum Test Loss:', np.min(loss_test))
    print('Maximum Accuracy:', np.max(accuracy))

    # Plot the loss
    plt.figure()
    plt.plot(np.arange(dt, T + 1, dt), loss_train, label='Training Loss')
    plt.plot(np.arange(dt, T + 1, dt), loss_test, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # PLot the accuracy
    plt.figure()
    plt.plot(np.arange(dt, T + 1, dt), accuracy, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # The performance with SGB -- 2000 epoch
    # Training time: 60.4261257648468
    # Minimum Training Loss: 0.5267274975776672
    # Minimum Test Loss: 0.7889460921287537
    # Maximum Accuracy: 0.7684999704360962

    # The performance with Adam -- 2000 epoch
    # Training time: 60.07949256896973
    # Minimum Training Loss: 1.3148653507232666
    # Minimum Test Loss: 1.46648371219635
    # Maximum Accuracy: 0.7114999890327454

    # The performance with Adam -- 2000 epoch + weight_decay = 0.7
    # Training time: 59.76921343803406
    # Minimum Training Loss: 1.6122194528579712
    # Minimum Test Loss: 1.5424575805664062
    # Maximum Accuracy: 0.7024999856948853
    # Training time: 60.04321527481079
    # Minimum Training Loss: 1.5550256967544556
    # Minimum Test Loss: 1.7299542427062988
    # Maximum Accuracy: 0.6955000162124634










