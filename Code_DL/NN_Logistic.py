

'''
The case of the logistic regression

'''

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,5)

def getdata():

    # Fetch the dataset
    # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    # Class indices of planes and dogs
    plane,dog = 0,5
    max_size = 2500  # Maximum number of data points

    # Prepare the training set
    labels_train = np.array(trainset.targets)
    selector_train = np.logical_or(labels_train==plane, labels_train==dog)
    X_train = torch.tensor(trainset.data[selector_train][:max_size]).float()
    y_train = torch.tensor((labels_train[selector_train]==plane).astype(dtype=np.float32)[:max_size])

    # Prepare the test set
    labels_test = np.array(testset.targets)
    selector_test = np.logical_or(labels_test==plane, labels_test==dog)
    X_test = torch.tensor(testset.data[selector_test]).float()
    y_test = torch.tensor((labels_test[selector_test]==plane).astype(dtype=np.float32))

    # Get the number of data points and the number of dimensions
    shape = X_train.shape
    n = shape[0]                        # Number of data points
    d = shape[1] * shape[2] * shape[3]  # Number of dimensions -- row-wise vector

    # Flatten the image data
    X_train_1D = torch.reshape(X_train, (-1,d))
    X_test_1D = torch.reshape(X_test, (-1,d))

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

    return X_train, X_test, X_train_1D, X_test_1D, y_train, y_test

def NN_Logistric():

    alpha = 3e-8        # Learning rate
    T = 7000            # Number of epochs
    log = False         # If False the prints will be disabled
    dt = 100            # Log the loss in every dt epochs
    loss_train = np.empty(T // dt)  # Training loss
    loss_test = np.empty(T // dt)   # Test loss
    accuracy = np.empty(T // dt)    # Test accuracy

    # Short variable names for brevity
    X_train, X_test, X_train_1D, X_test_1D, y_train, y_test = getdata()
    x, x_ = X_train_1D.t(), X_test_1D.t()
    y, y_ = y_train, y_test

    # Construct the weight vector
    d = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
    weight_scale = 1e-5
    W = weight_scale * torch.rand(1, d)  # Random weights between 0 and 1
    cuda = True
    if cuda and torch.cuda.is_available():  # Copy to GPU if it is available
        W = W.cuda()

    # Enable Autograd. This allows us to automatically compute the gradient of the loss w.r.t this weight vectors
    W.requires_grad_(True)

    # Start training
    start = time.time()
    for t in range(T):

        # Forward Pass
        z = torch.sigmoid(W @ x)[0]
        z_check = torch.sigmoid(W @ x)

        # Calculate the loss
        loss = F.binary_cross_entropy(z, y)

        # Backward pass, (thanks to Autograd) -- loss = loss(W)
        loss.backward()

        # Print the training loss, the test loss and the test accuracy
        if (t + 1) % dt == 0:

            # Store the training loss
            loss_train[t // dt] = loss

            # Predict the classes of the images in the test set
            z = torch.sigmoid(W @ x_)[0]
            loss_test[t // dt] = F.binary_cross_entropy(z, y_)
            accuracy[t // dt] = 1 - torch.mean(torch.abs((z > 0.5).float() - y_))

            if log:
                print('Epoch:', t + 1)
                print('Training Loss:', loss_train[t // dt])
                print('Test Loss:', loss_test[t // dt])
                print('Accuracy:', accuracy[t // dt])

        # Update the weights
        with torch.no_grad():  # Do not calculate the gradient for the following operations
            W -= alpha * W.grad
            W.grad.zero_()  # Clear the gradient

    end = time.time()

    # Print the statistics
    print('Training time:', end - start)
    print('Minimum Training Loss:', np.min(loss_train))
    print('Minimum Test Loss:', np.min(loss_test))
    print('Maximum Accuracy:', np.max(accuracy))

    return loss_train, loss_test, accuracy

# The main function
X_train, X_test, X_train_1D, X_test_1D, y_train, y_test = getdata()
print("The shape of X_train is ", X_train.shape)
print("The shape of X_test is ", X_test.shape)
print("The shape of X_train_1D is ", X_train_1D.shape)
print("The shape of X_test_1D is ", X_test_1D.shape)

# The logistic regressipn
loss_train, loss_test, accuracy = NN_Logistric()

print("The shape of loss_train is ", loss_train.shape)
print("The shape of loss_test is ", loss_test.shape)
print("The shape of accuracy is ", accuracy.shape)

# Training loss VS Epochs
x_show = np.linspace(0, len(loss_train), num = len(loss_train))
y_show = loss_train

plt.plot( x_show, y_show, label = "Training loss" )
plt.title('Training loss VS epochs')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.show()

# Testing loss VS Epochs
x_show = np.linspace(0, len(loss_test), num = len(loss_test))
y_show = loss_test

plt.plot( x_show, y_show, label = "Test loss" )
plt.title('Testing loss VS epochs')
plt.xlabel('Epochs')
plt.ylabel('Testing loss')
plt.show()

# Accuracy over time VS Epochs
x_show = np.linspace(0, len(accuracy), num = len(accuracy))
y_show = accuracy

plt.plot( x_show, y_show, label = "Accuracy" )
plt.title('Accuracy VS epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()



