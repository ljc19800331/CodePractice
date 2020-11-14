
'''
This is the pytorch implementation of the problem
1.

reference:
1. https://pytorch.org/tutorials/beginner/pytorch_with_examples.html


'''

# -*- coding: utf-8 -*-
import torch
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
# import numpy as np
import matplotlib.pyplot as plt
import random

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

    return x_train, x_test, y_train, y_test

def main(weight_scale = 1.0, n_hidden = 6):

    # Basic definition
    T = 5000                        # Number of epochs
    dt = 100                        # Log the loss in every dt epochs
    loss_train = np.empty(T // dt)  # Training loss
    loss_test = np.empty(T // dt)   # Test loss
    accuracy = np.empty(T // dt)    # Test accuracy
    learning_rate = 7e-2
    dtype = torch.float
    device = torch.device("cuda:0")

    # The homework data
    n_hidden = 6
    x, x_, y, y_ = get_data()
    N, D_in, H, D_out = x.shape[0], x.shape[1], n_hidden, 1
    weight_scale = 1e-5
    # weight_scale = torch.from_numpy( np.asarray([weight_scale]) ).float().to(device)
    # 1e-8 to 1e-3 for scaling

    w1 = weight_scale * np.random.rand(D_in, H)
    w1 = torch.from_numpy(w1).float().to(device)
    w1 = torch.autograd.Variable(w1, requires_grad = True)
    w2 = weight_scale * np.random.rand(H, H)
    w2 = torch.from_numpy(w2).float().to(device)
    w2 = torch.autograd.Variable(w2, requires_grad = True)
    w3 = weight_scale * np.random.rand(H, D_out)
    w3 = torch.from_numpy(w3).float().to(device)
    w3 = torch.autograd.Variable(w3, requires_grad = True)

    # w1 = weight_scale * torch.randn(D_in, H, device = device, dtype = dtype, requires_grad = True)
    # w2 = weight_scale * torch.randn(H, H, device = device, dtype = dtype, requires_grad = True)
    # w3 = weight_scale * torch.randn(H, D_out, device = device, dtype = dtype, requires_grad = True)
    #
    # print("w1 = ", w1)
    # print("w2 = ", w2)
    # input("check")

    # Count the time in the loop
    t1 = time.time()
    for t in range(5000):

        # Forward Pass
        print("check x shape = ", x.shape)
        input("check")
        x1 = x.mm(w1)
        x2 = torch.sigmoid(x1)
        x3 = x2.mm(w2)
        x4 = torch.sigmoid(x3)
        x5 = x4.mm(w3)
        y_pred = torch.sigmoid(x5)

        # Loss function
        y_train_check = torch.reshape(y, y_pred.shape)
        loss = F.binary_cross_entropy(y_pred, y_train_check)

        # setpoint checking
        if (t + 1) % dt == 0:

            # check the loss accuracy
            x1 = x_.mm(w1)
            x2 = torch.sigmoid(x1)
            x3 = x2.mm(w2)
            x4 = torch.sigmoid(x3)
            x5 = x4.mm(w3)
            z = torch.sigmoid(x5)
            y_test_check = torch.reshape(y_, z.shape)

            # Record the result -- pending
            loss_train[t // dt] = loss
            loss_curr = F.binary_cross_entropy(z, y_test_check)
            acc = 1 - torch.mean(torch.abs((z > 0.5).float() - y_test_check))
            loss_test[t // dt] = loss_curr
            accuracy[t // dt] = acc

            print('Epoch:', t + 1)
            print('Training Loss:', loss_train[t // dt])
            print('Test Loss:', loss_test[t // dt])
            print('Accuracy:', accuracy[t // dt])

        # autograd
        loss.backward()

        with torch.no_grad():

            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            w3 -= learning_rate * w3.grad

            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()
            w3.grad.zero_()

    t2 = time.time()

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

    # Report total number of parameters
    N_weights = w1.shape[0] * w1.shape[1] + w2.shape[0] * w2.shape[1] + w3.shape[0] * w3.shape[1]
    print("The total number of weight is defined as ", N_weights)

    # the training time
    print("The training time is ", t2 - t1)

    # the minimum training and test loss
    print('Minimum Training Loss:', np.min(loss_train))
    print('Minimum Test Loss:', np.min(loss_test))

    # maximum accuracy
    print('Maximum Accuracy:', np.max(accuracy))

    # Compute the convergence rate for the NN network
    list_rate = []
    for i in range(len(loss_train) - 1):
        list_rate.append(  np.abs(loss_train[i+1] - loss_train[i])  )
    x_show = np.linspace(0, len(list_rate), len(list_rate))
    y_show = list_rate
    plt.figure()
    plt.plot(x_show, y_show, label = 'Convergennce Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Convergennce Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Compare with the logistic regression result

    # Vary the weight scale by orders of 10 ranging from 1e-8 to 1e-3
    # train the network again with 8000 epochs

if __name__ == "__main__":

    main()

